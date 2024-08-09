from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

import flax.linen as nn
import jax.numpy as jnp

from ..configuration_utils import ConfigMixin, register_to_config
# from ...loaders import FromOriginalModelMixin, PeftAdapterMixin
from ...models.attention import FeedForward
from ...models.attention_processor import Attention, FluxAttnProcessor2_0, FluxSingleAttnProcessor2_0
from .modeling_flax_utils import FlaxModelMixin
from ...models.normalization import AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormZeroSingle
from ...utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from ...utils.flax_utils import maybe_allow_in_graph
from ..embeddings import CombinedTimestepGuidanceTextProjEmbeddings, CombinedTimestepTextProjEmbeddings
from ..modeling_outputs import Transformer2DModelOutput


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def rope(pos: jnp.ndarray, dim: int, theta: int) -> jnp.ndarray:
    assert dim % 2 == 0, "The dimension must be even."

    scale = jnp.arange(0, dim, 2) / dim
    omega = 1.0 / (theta**scale)

    batch_size, seq_length = pos.shape
    out = jnp.einsum("...n,d->...nd", pos, omega)
    cos_out = jnp.cos(out)
    sin_out = jnp.sin(out)

    stacked_out = jnp.stack([cos_out, -sin_out, sin_out, cos_out], axis=-1)
    out = stacked_out.reshape(batch_size, -1, dim // 2, 2, 2)
    return out.astype(jnp.float32)


class EmbedND(nn.Module):
    dim: int
    theta: int
    axes_dim: List[int]

    def __call__(self, ids: jnp.ndarray) -> jnp.ndarray:
        n_axes = ids.shape[-1]
        emb = jnp.concatenate(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            axis=-3,
        )
        return emb[:, jnp.newaxis, :]


class FluxSingleTransformerBlock(nn.Module):
    dim: int
    num_attention_heads: int
    attention_head_dim: int
    mlp_ratio: float = 4.0

    def setup(self):
        self.mlp_hidden_dim = int(self.dim * self.mlp_ratio)

        self.norm = AdaLayerNormZeroSingle(self.dim)
        self.proj_mlp = nn.Dense(self.mlp_hidden_dim)
        self.act_mlp = nn.gelu
        self.proj_out = nn.Dense(self.dim)

        processor = FluxSingleAttnProcessor2_0()
        self.attn = Attention(
            query_dim=self.dim,
            cross_attention_dim=None,
            dim_head=self.attention_head_dim,
            heads=self.num_attention_heads,
            out_dim=self.dim,
            bias=True,
            processor=processor,
            qk_norm="rms_norm",
            eps=1e-6,
            pre_only=True,
        )

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        temb: jnp.ndarray,
        image_rotary_emb=None,
    ):
        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = jnp.concatenate([attn_output, mlp_hidden_states], axis=-1)
        gate = gate[:, jnp.newaxis, :]
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class FluxTransformerBlock(nn.Module):
    dim: int
    num_attention_heads: int
    attention_head_dim: int
    qk_norm: str = "rms_norm"
    eps: float = 1e-6

    def setup(self):
        self.norm1 = AdaLayerNormZero(self.dim)
        self.norm1_context = AdaLayerNormZero(self.dim)

        if hasattr(nn, "scaled_dot_product_attention"):
            processor = FluxAttnProcessor2_0()
        else:
            raise ValueError(
                "The current PyTorch version does not support the `scaled_dot_product_attention` function."
            )
        self.attn = Attention(
            query_dim=self.dim,
            cross_attention_dim=None,
            added_kv_proj_dim=self.dim,
            dim_head=self.attention_head_dim,
            heads=self.num_attention_heads,
            out_dim=self.dim,
            context_pre_only=False,
            bias=True,
            processor=processor,
            qk_norm=self.qk_norm,
            eps=self.eps,
        )

        self.norm2 = nn.LayerNorm(self.dim, use_bias=False, epsilon=self.eps)
        self.ff = FeedForward(dim=self.dim, dim_out=self.dim, activation_fn="gelu")

        self.norm2_context = nn.LayerNorm(self.dim, use_bias=False, epsilon=self.eps)
        self.ff_context = FeedForward(dim=self.dim, dim_out=self.dim, activation_fn="gelu")

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        encoder_hidden_states: jnp.ndarray,
        temb: jnp.ndarray,
        image_rotary_emb=None,
    ):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )

        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        attn_output = gate_msa[:, jnp.newaxis, :] * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, jnp.newaxis]) + shift_mlp[:, jnp.newaxis]

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp[:, jnp.newaxis, :] * ff_output

        hidden_states = hidden_states + ff_output

        context_attn_output = c_gate_msa[:, jnp.newaxis, :] * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, jnp.newaxis]) + c_shift_mlp[:, jnp.newaxis]

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp[:, jnp.newaxis, :] * context_ff_output

        return encoder_hidden_states, hidden_states


@dataclass
class FluxTransformer2DModelConfig(ConfigMixin):
    patch_size: int = 1
    in_channels: int = 64
    num_layers: int = 19
    num_single_layers: int = 38
    attention_head_dim: int = 128
    num_attention_heads: int = 24
    joint_attention_dim: int = 4096
    pooled_projection_dim: int = 768
    guidance_embeds: bool = False
    axes_dims_rope: List[int] = (16, 56, 56)


class FluxTransformer2DModel(FlaxModelMixin, FromOriginalModelMixin, PeftAdapterMixin):
    config: FluxTransformer2DModelConfig

    def setup(self):
        self.out_channels = self.config.in_channels
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim

        self.pos_embed = EmbedND(dim=self.inner_dim, theta=10000, axes_dim=self.config.axes_dims_rope)
        text_time_guidance_cls = (
            CombinedTimestepGuidanceTextProjEmbeddings if self.config.guidance_embeds else CombinedTimestepTextProjEmbeddings
        )
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=self.inner_dim, pooled_projection_dim=self.config.pooled_projection_dim
        )

        self.context_embedder = nn.Dense(self.inner_dim)
        self.x_embedder = nn.Dense(self.inner_dim)

        self.transformer_blocks = [
            FluxTransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=self.config.num_attention_heads,
                attention_head_dim=self.config.attention_head_dim,
            )
            for _ in range(self.config.num_layers)
        ]

        self.single_transformer_blocks = [
            FluxSingleTransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=self.config.num_attention_heads,
                attention_head_dim=self.config.attention_head_dim,
            )
            for _ in range(self.config.num_single_layers)
        ]

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, use_bias=False, epsilon=1e-6)
        self.proj_out = nn.Dense(self.config.patch_size * self.config.patch_size * self.out_channels)

        self.gradient_checkpointing = False

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        pooled_projections: Optional[jnp.ndarray] = None,
        timestep: Optional[jnp.ndarray] = None,
        img_ids: Optional[jnp.ndarray] = None,
        txt_ids: Optional[jnp.ndarray] = None,
        guidance: Optional[jnp.ndarray] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[jnp.ndarray, Transformer2DModelOutput]:
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)

        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.astype(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.astype(hidden_states.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        ids = jnp.concatenate((txt_ids, img_ids), axis=1)
        image_rotary_emb = self.pos_embed(ids)

        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )

        hidden_states = jnp.concatenate([encoder_hidden_states, hidden_states], axis=1)

        for block in self.single_transformer_blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1]:, :]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return output

        return Transformer2DModelOutput(sample=output)
