import math
from dataclasses import dataclass
from typing import List, Tuple
import jax
import flax.linen as nn
import jax.numpy as jnp
from jax import random
from flax.linen import partitioning as nn_partitioning
from einops import rearrange



# Utility functions and modules


def rope(pos, dim, theta):
    assert dim % 2 == 0
    scale = jnp.arange(0, dim, 2) / dim
    omega = 1.0 / (theta ** scale)
    out = jnp.einsum('...n,d->...nd', pos, omega)
    out = jnp.stack([jnp.cos(out), -jnp.sin(out), jnp.sin(out), jnp.cos(out)], axis=-1)
    out = rearrange(out, 'b n d (i j) -> b n d i j', i=2, j=2)
    return out#.astype(jnp.float32)

def apply_rope(xq, xk, freqs_cis):
    xq_ = xq.reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape), xk_out.reshape(*xk.shape)

def attention(q, k, v, pe):
    q, k = apply_rope(q, k, pe)
    x = jax.nn.softmax(jnp.einsum('bhqd, bhkd -> bhqk', q, k) / jnp.sqrt(q.shape[-1]), axis=-1)
    x = jnp.einsum('bhqk, bhvd -> bhqd', x, v)
    x = rearrange(x, 'B H L D -> B L (H D)')
    return x

def timestep_embedding(t, dim, max_period=10000, time_factor=1000.0):
    t = time_factor * t
    half = dim // 2
    freqs = jnp.exp(-math.log(max_period) * jnp.arange(0, half) / half)
    args = jnp.expand_dims(t, -1) * jnp.expand_dims(freqs, 0)
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    if dim % 2:
        embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
    return embedding

class EmbedND(nn.Module):
    dim: int
    theta: int
    axes_dim: List[int]

    @nn.compact
    def __call__(self, ids):
        n_axes = ids.shape[-1]
        emb = jnp.concatenate(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            axis=-3,
        )
        return jnp.expand_dims(emb, 1)

class MLPEmbedder(nn.Module):
    in_dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.silu(x)
        x = nn.Dense(self.hidden_dim)(x)
        return x

class RMSNorm(nn.Module):
    dim: int

    def setup(self):
        self.scale = self.param('scale', nn.initializers.ones, (self.dim,))

    @nn.compact
    def __call__(self, x):
        rrms = jax.lax.rsqrt(jnp.mean(x**2, axis=-1, keepdims=True) + 1e-6)
        return (x * rrms) * self.scale

class QKNorm(nn.Module):
    dim: int

    def setup(self):
        self.query_norm = RMSNorm(self.dim)
        self.key_norm = RMSNorm(self.dim)

    @nn.compact
    def __call__(self, q, k, v):
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q, k

class SelfAttention(nn.Module):
    dim: int
    num_heads: int = 8
    qkv_bias: bool = False

    def setup(self):
        self.head_dim = self.dim // self.num_heads
        self.qkv = nn.Dense(self.dim * 3, use_bias=self.qkv_bias)
        self.norm = QKNorm(self.head_dim)
        self.proj = nn.Dense(self.dim)

    @nn.compact
    def __call__(self, x, pe):
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = self.proj(x)
        return x

@dataclass
class ModulationOut:
    shift: jnp.ndarray
    scale: jnp.ndarray
    gate: jnp.ndarray

class Modulation(nn.Module):
    dim: int
    double: bool

    def setup(self):
        self.multiplier = 6 if self.double else 3
        self.lin = nn.Dense(self.multiplier * self.dim)

    @nn.compact
    def __call__(self, vec):
        out = self.lin(nn.silu(vec)).reshape((-1, self.multiplier, self.dim))
        return (
            ModulationOut(*out[:, :3]),
            ModulationOut(*out[:, 3:]) if self.double else None
        )

class DoubleStreamBlock(nn.Module):
    hidden_size: int
    num_heads: int
    mlp_ratio: float
    qkv_bias: bool = False

    def setup(self):
        self.mlp_hidden_dim = int(self.hidden_size * self.mlp_ratio)
        self.img_mod = Modulation(self.hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm()
        self.img_attn = SelfAttention(self.hidden_size, self.num_heads, self.qkv_bias)
        self.img_norm2 = nn.LayerNorm()
        self.img_mlp = nn.Sequential(
            nn.Dense(self.mlp_hidden_dim),
            nn.gelu,
            nn.Dense(self.hidden_size),
        )
        self.txt_mod = Modulation(self.hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm()
        self.txt_attn = SelfAttention(self.hidden_size, self.num_heads, self.qkv_bias)
        self.txt_norm2 = nn.LayerNorm()
        self.txt_mlp = nn.Sequential(
            nn.Dense(self.mlp_hidden_dim),
            nn.gelu,
            nn.Dense(self.hidden_size),
        )

    @nn.compact
    def __call__(self, img, txt, vec, pe):
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        q = jnp.concatenate([txt_q, img_q], axis=2)
        k = jnp.concatenate([txt_k, img_k], axis=2)
        v = jnp.concatenate([txt_v, img_v], axis=2)

        attn = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn[:, :txt.shape[1]], attn[:, txt.shape[1]:]

        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)

        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        return img, txt

class SingleStreamBlock(nn.Module):
    hidden_size: int
    num_heads: int
    mlp_ratio: float = 4.0
    qk_scale: float = None

    def setup(self):
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.qk_scale or self.head_dim**-0.5
        self.mlp_hidden_dim = int(self.hidden_size * self.mlp_ratio)
        self.linear1 = nn.Dense(self.hidden_size * 3 + self.mlp_hidden_dim)
        self.linear2 = nn.Dense(self.hidden_size + self.mlp_hidden_dim)
        self.norm = QKNorm(self.head_dim)
        self.pre_norm = nn.LayerNorm()
        self.mlp_act = nn.gelu
        self.modulation = Modulation(self.hidden_size, double=False)

    @nn.compact
    def __call__(self, x, vec, pe):
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = jnp.split(self.linear1(x_mod), [3 * self.hidden_size], axis=-1)
        q, k, v = rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        attn = attention(q, k, v, pe=pe)
        output = self.linear2(jnp.concatenate([attn, self.mlp_act(mlp)], axis=2))
        return x + mod.gate * output

class LastLayer(nn.Module):
    hidden_size: int
    patch_size: int
    out_channels: int

    def setup(self):
        self.norm_final = nn.LayerNorm()
        self.linear = nn.Dense(self.patch_size * self.patch_size * self.out_channels)
        self.adaLN_modulation = nn.Sequential(nn.silu, nn.Dense(2 * self.hidden_size))

    @nn.compact
    def __call__(self, x, vec):
        shift, scale = jnp.split(self.adaLN_modulation(vec), 2, axis=1)
        x = (1 + jnp.expand_dims(scale, 1)) * self.norm_final(x) + jnp.expand_dims(shift, 1)
        x = self.linear(x)
        return x


# Dummy data and configurations
BATCH_SIZE = 2
SEQ_LENGTH = 10
HIDDEN_SIZE = 64
NUM_HEADS = 8
MLP_RATIO = 4.0
PATCH_SIZE = 4
OUT_CHANNELS = 3

# Testing the individual functions and modules
def test_functions_and_modules():
    key = random.PRNGKey(0)
    dummy_ids = random.randint(key, (BATCH_SIZE, SEQ_LENGTH,2), 0, 100)
    dummy_t = jnp.linspace(0, 1, SEQ_LENGTH)
    dummy_vec = random.normal(key, (BATCH_SIZE, HIDDEN_SIZE))
    dummy_img = random.normal(key, (BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE,HIDDEN_SIZE))
    dummy_txt = random.normal(key, (BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE))
    dummy_pe = rope(dummy_ids[0], HIDDEN_SIZE, 10000)

    # Test timestep_embedding
    timestep_emb = timestep_embedding(dummy_t, HIDDEN_SIZE)
    print("Timestep embedding:", timestep_emb.shape)

    # Test EmbedND
    embed_nd = EmbedND(dim=HIDDEN_SIZE, theta=10000, axes_dim=[64, 64])
    emb = embed_nd.init(key, dummy_ids)
    print("EmbedND output:", emb)

    # Test MLPEmbedder
    mlp_embedder = MLPEmbedder(in_dim=HIDDEN_SIZE, hidden_dim=HIDDEN_SIZE)
    mlp_emb = mlp_embedder.init(key, dummy_vec)
    print("MLPEmbedder output:", mlp_emb)

    # Test RMSNorm
    rms_norm = RMSNorm(dim=HIDDEN_SIZE)
    rms_out = rms_norm.init(key, dummy_img)
    print("RMSNorm output:", rms_out)

    # Test QKNorm
    qknorm = QKNorm(dim=HIDDEN_SIZE)
    qknorm_out = qknorm.init(key, dummy_img, dummy_txt, dummy_img)
    print("QKNorm output:", qknorm_out)

    # Test SelfAttention
    self_attention = SelfAttention(dim=HIDDEN_SIZE, num_heads=NUM_HEADS)
    self_attention_out = self_attention.init(key, dummy_img, dummy_pe)
    print("SelfAttention output:", self_attention_out)

    # Test Modulation
    modulation = Modulation(dim=HIDDEN_SIZE, double=True)
    modulation_out = modulation.init(key, dummy_vec)
    print("Modulation output:", modulation_out)

    # Test DoubleStreamBlock
    double_stream_block = DoubleStreamBlock(hidden_size=HIDDEN_SIZE, num_heads=NUM_HEADS, mlp_ratio=MLP_RATIO)
    double_stream_out = double_stream_block.init(key, dummy_img, dummy_txt, dummy_vec, dummy_pe)
    print("DoubleStreamBlock output:", double_stream_out)

    # Test SingleStreamBlock
    single_stream_block = SingleStreamBlock(hidden_size=HIDDEN_SIZE, num_heads=NUM_HEADS, mlp_ratio=MLP_RATIO)
    single_stream_out = single_stream_block.init(key, dummy_img, dummy_vec, dummy_pe)
    print("SingleStreamBlock output:", single_stream_out)

    # Test LastLayer
    last_layer = LastLayer(hidden_size=HIDDEN_SIZE, patch_size=PATCH_SIZE, out_channels=OUT_CHANNELS)
    last_layer_out = last_layer.init(key, dummy_img, dummy_vec)
    print("LastLayer output:", last_layer_out)

if __name__ == "__main__":
    test_functions_and_modules()