import math
from dataclasses import dataclass
import jax
import flax.linen as nn
import jax.numpy as jnp
from jax import random
from flax.linen import partitioning as nn_partitioning
from einops import rearrange, repeat

from dataclasses import dataclass

from typing import Dict, List, Tuple, Any, Optional, Callable

from transformers import (FlaxCLIPTextModel, CLIPTokenizer, FlaxT5EncoderModel, T5Tokenizer,T5EncoderModel)

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
        return emb # jnp.expand_dims(emb, 1)

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
        self.img_mlp = nn.Sequential([
            nn.Dense(self.mlp_hidden_dim),
            nn.gelu,
            nn.Dense(self.hidden_size),
        ])
        self.txt_mod = Modulation(self.hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm()
        self.txt_attn = SelfAttention(self.hidden_size, self.num_heads, self.qkv_bias)
        self.txt_norm2 = nn.LayerNorm()
        self.txt_mlp = nn.Sequential([
            nn.Dense(self.mlp_hidden_dim),
            nn.gelu,
            nn.Dense(self.hidden_size),
        ])

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
        self.adaLN_modulation = nn.Sequential([nn.silu, nn.Dense(2 * self.hidden_size)])

    @nn.compact
    def __call__(self, x, vec):
        shift, scale = jnp.split(self.adaLN_modulation(vec), 2, axis=1)
        x = (1 + jnp.expand_dims(scale, 1)) * self.norm_final(x) + jnp.expand_dims(shift, 1)
        x = self.linear(x)
        return x

# Dummy data and configurations
BATCH_SIZE = 4
SEQ_LENGTH = 32
HIDDEN_SIZE = 320
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
    dummy_img = random.normal(key, (BATCH_SIZE, SEQ_LENGTH, SEQ_LENGTH, HIDDEN_SIZE))
    dummy_txt = random.normal(key, (BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE))
    dummy_pe = rope(dummy_ids[0], HIDDEN_SIZE, 10000)
    print("Positional embedding:", dummy_pe.shape)

    param_shape = lambda x: jax.tree.map(lambda y:y.shape, x)

    # Test timestep_embedding
    timestep_emb = timestep_embedding(dummy_t, HIDDEN_SIZE)
    print("Timestep embedding:", timestep_emb.shape)

    # Test EmbedND
    embed_nd = EmbedND(dim=HIDDEN_SIZE, theta=10000, axes_dim=[64, 64])
    emb = embed_nd.init(key, dummy_ids)
    print("EmbedND output:", param_shape(emb))

    # Test MLPEmbedder
    mlp_embedder = MLPEmbedder(in_dim=HIDDEN_SIZE, hidden_dim=HIDDEN_SIZE)
    mlp_emb = mlp_embedder.init(key, dummy_vec)
    print("MLPEmbedder output:", param_shape(mlp_emb))

    # Test RMSNorm
    rms_norm = RMSNorm(dim=HIDDEN_SIZE)
    rms_out = rms_norm.init(key, dummy_img)
    print("RMSNorm output:", param_shape(rms_out))

    # Test QKNorm
    qknorm = QKNorm(dim=HIDDEN_SIZE)
    qknorm_out = qknorm.init(key, dummy_img, dummy_txt, dummy_img)
    print("QKNorm output:", param_shape(qknorm_out))

    # Test SelfAttention
    self_attention = SelfAttention(dim=HIDDEN_SIZE, num_heads=NUM_HEADS)
    self_attention_out = self_attention.init(key, dummy_txt, dummy_pe)
    print("SelfAttention output:", param_shape(self_attention_out))

    # Test Modulation
    # modulation = Modulation(dim=HIDDEN_SIZE, double=True)
    # modulation_out = modulation.init(key, dummy_vec)
    # print("Modulation output:", modulation_out)

    # Test DoubleStreamBlock
    # double_stream_block = DoubleStreamBlock(hidden_size=HIDDEN_SIZE, num_heads=NUM_HEADS, mlp_ratio=MLP_RATIO)
    # double_stream_out = double_stream_block.init(key, dummy_img, dummy_txt, dummy_vec, dummy_pe)
    # print("DoubleStreamBlock output:", double_stream_out)

    # Test SingleStreamBlock
    # single_stream_block = SingleStreamBlock(hidden_size=HIDDEN_SIZE, num_heads=NUM_HEADS, mlp_ratio=MLP_RATIO)
    # single_stream_out = single_stream_block.init(key, dummy_img, dummy_vec, dummy_pe)
    # print("SingleStreamBlock output:", single_stream_out)

    # Test LastLayer
    # last_layer = LastLayer(hidden_size=HIDDEN_SIZE, patch_size=PATCH_SIZE, out_channels=OUT_CHANNELS)
    # last_layer_out = last_layer.init(key, dummy_img, dummy_vec)
    # print("LastLayer output:", last_layer_out)


@dataclass
class FluxParams:
    in_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool



class Flux(nn.Module):
    params: FluxParams

    def setup(self):
        self.in_channels = self.params.in_channels
        self.out_channels = self.in_channels

        if self.params.hidden_size % self.params.num_heads != 0:
            raise ValueError(
                f"Hidden size {self.params.hidden_size} must be divisible by num_heads {self.params.num_heads}"
            )
        pe_dim = self.params.hidden_size // self.params.num_heads
        if sum(self.params.axes_dim) != pe_dim:
            raise ValueError(f"Got {self.params.axes_dim} but expected positional dim {pe_dim}")
        
        self.hidden_size = self.params.hidden_size
        self.num_heads = self.params.num_heads

        self.pe_embedder = EmbedND(dim=pe_dim, theta=self.params.theta, axes_dim=self.params.axes_dim)
        self.img_in = nn.Dense(self.hidden_size)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(self.params.vec_in_dim, self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if self.params.guidance_embed else nn.Identity()
        )
        self.txt_in = nn.Dense(self.hidden_size)

        self.double_blocks = [
            DoubleStreamBlock(
                self.hidden_size,
                self.num_heads,
                mlp_ratio=self.params.mlp_ratio,
                qkv_bias=self.params.qkv_bias,
            ) for _ in range(self.params.depth)
        ]

        self.single_blocks = [
            SingleStreamBlock(
                self.hidden_size, self.num_heads, mlp_ratio=self.params.mlp_ratio
            ) for _ in range(self.params.depth_single_blocks)
        ]

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

    def __call__(
        self,
        img: jnp.ndarray,
        img_ids: jnp.ndarray,
        txt: jnp.ndarray,
        txt_ids: jnp.ndarray,
        timesteps: jnp.ndarray,
        y: jnp.ndarray,
        guidance: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))

        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))

        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = jnp.concatenate((txt_ids, img_ids), axis=1)
        pe = self.pe_embedder(ids)

        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

        img = jnp.concatenate((txt, img), axis=1)

        for block in self.single_blocks:
            img = block(img, vec=vec, pe=pe)

        img = img[:, txt.shape[1]:, ...]

        img = self.final_layer(img, vec)
        return img


class HFEmbedder:

    def __init__(self,version: str,max_length: int):
        self.version = version
        self.max_length = max_length
        self.is_clip = self.version.startswith("openai")
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"

        if self.is_clip:
            self.tokenizer = CLIPTokenizer.from_pretrained(self.version, max_length=self.max_length)
            self.hf_module = FlaxCLIPTextModel.from_pretrained(self.version, dtype=jnp.bfloat16,from_pt=True)
        else:
            self.tokenizer = T5Tokenizer.from_pretrained(self.version, max_length=self.max_length)
            self.hf_module = FlaxT5EncoderModel.from_pretrained(self.version, dtype=jnp.bfloat16,from_pt=True) # 'cfu/t5-v1_1-xxl-encoder-bfloat16-flax'

    def __call__(self, text: List[str]) -> jnp.ndarray:
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="np",
        )

        input_ids = jax.device_put(batch_encoding["input_ids"])

        outputs = self.hf_module(
            input_ids=input_ids,
            attention_mask=None,
            output_hidden_states=False,
        )

        return jax.device_put(outputs[self.output_key])

def get_noise(
    num_samples: int,
    height: int,
    width: int,
    key: jax.random.PRNGKey,
    dtype: jnp.dtype,
):
    key, subkey = jax.random.split(key)
    return jax.random.normal(
        subkey,
        (
            num_samples,
            16,
            2 * math.ceil(height / 16),
            2 * math.ceil(width / 16),
        ),
        dtype=dtype,
    )


def prepare(t5: HFEmbedder, clip: HFEmbedder, img: jnp.ndarray, prompt: str | list[str]) -> dict[str, jnp.ndarray]:
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = jnp.zeros((h // 2, w // 2, 3))
    img_ids = img_ids.at[..., 1].add(jnp.arange(h // 2)[:, None])
    img_ids = img_ids.at[..., 2].add(jnp.arange(w // 2)[None, :])
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = jnp.zeros((bs, txt.shape[1], 3))

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img,
        "img_ids": img_ids,
        "txt": txt,
        "txt_ids": txt_ids,
        "vec": vec,
    }


def time_shift(mu: float, sigma: float, t: jnp.ndarray):
    return jnp.exp(mu) / (jnp.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = jnp.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # estimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def denoise(
    model: Flux,
    # model input
    img: jnp.ndarray,
    img_ids: jnp.ndarray,
    txt: jnp.ndarray,
    txt_ids: jnp.ndarray,
    vec: jnp.ndarray,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
):
    # this is ignored for schnell
    guidance_vec = jnp.full((img.shape[0],), guidance, dtype=img.dtype)
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = jnp.full((img.shape[0],), t_curr, dtype=img.dtype)
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
        )

        img = img + (t_prev - t_curr) * pred

    return img


def unpack(x: jnp.ndarray, height: int, width: int) -> jnp.ndarray:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )

def test_embedder():
    # embedder = HFEmbedder('cfu/t5-v1_1-xxl-encoder-bfloat16-flax', max_length=512)
    # print(embedder(["hihi"]).shape)
    # convert and push
    # embedder = HFEmbedder("google/t5-v1_1-xxl", max_length=512)
    # embedder.hf_module.push_to_hub('t5-v1_1-xxl-encoder-bfloat16-flax',token='', safe_serialization=True) 
    # embedder.tokenizer.push_to_hub('t5-v1_1-xxl-encoder-bfloat16-flax',token='', safe_serialization=True)
    # load it from bflab instead
    embedder = T5EncoderModel.from_pretrained('black-forest-labs/FLUX.1-dev',subfolder='text_encoder_2', use_safetensors=True,token='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')
    # embedder.push_to_hub('t5-v1_1-xxl-encoder-bin',safe_serialization=False,token='')


if __name__ == "__main__":
    # test_functions_and_modules()
    test_embedder()
