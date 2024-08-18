# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Callable, Iterable, Tuple, Union


import flax.linen as nn
import jax
import jax.numpy as jnp
from maxdiffusion.models.act_flax import rcolu, colu, make_conv
# Not sure which initializer to use, ruff was complaining, so added an ignore --I prefer xavier to lecun init... (we care about out_channels in generative models!)
# from jax.nn import initializers # noqa: F811


# Type annotations
Array = jnp.ndarray
DType = jnp.dtype
Dtype = Any  # this could be a real type? --I guess Object is better...
PRNGKey = jnp.ndarray
Shape = Iterable[int]
Activation = Callable[..., Array]
# Parameter initializers.
Initializer = Callable[[PRNGKey, Shape, DType], Array]
InitializerAxis = Union[int, Tuple[int, ...]]
NdInitializer = Callable[[PRNGKey, Shape, DType, InitializerAxis, InitializerAxis], Array]

class FlaxUpsample2D(nn.Module):
    out_channels: int
    dtype: jnp.dtype = jnp.float32
    conv3d: bool = False

    def setup(self):
        self.conv = make_conv('1x1',conv3d=self.conv3d,out_channels=self.out_channels,dtype=self.dtype,)

    @nn.compact
    def __call__(self, hidden_states):
        batch, height, width, channels = hidden_states.shape
        hidden_states = jax.image.resize(
            hidden_states,
            shape=(batch, height * 2, width * 2, channels),
            method="nearest",
        )
        hidden_states = self.conv(hidden_states)
        hidden_states = nn.with_logical_constraint(
            hidden_states,
            ('batch', 'keep_1', 'keep_2', 'out_channels')
        )
        return hidden_states


class FlaxDownsample2D(nn.Module):
    out_channels: int
    dtype: jnp.dtype = jnp.float32
    conv3d: bool = False

    def setup(self):
        self.conv = make_conv('down', conv3d=self.conv3d, out_channels=self.out_channels, dtype=self.dtype,)
    @nn.compact
    def __call__(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = nn.with_logical_constraint(
            hidden_states,
            ('batch', 'keep_1', 'keep_2', 'out_channels')
        )
        return hidden_states


class FlaxResnetBlock2D(nn.Module):
    in_channels: int
    out_channels: int = None
    dropout_prob: float = 0.0
    use_nin_shortcut: bool = None
    dtype: jnp.dtype = jnp.float32
    norm_num_groups: int = 32
    act_fn: str = "relu"
    conv3d: bool = False

    def setup(self):
        out_channels = self.in_channels if self.out_channels is None else self.out_channels

        self.norm1 = nn.GroupNorm(num_groups=self.norm_num_groups, epsilon=1e-5)

        self.norm2 = nn.GroupNorm(num_groups=self.norm_num_groups, epsilon=1e-5)
        self.dropout = nn.Dropout(self.dropout_prob)

        use_nin_shortcut = self.in_channels != out_channels if self.use_nin_shortcut is None else self.use_nin_shortcut

        self.conv_shortcut = None
        if use_nin_shortcut:
            self.conv_shortcut = make_conv('1x1', conv3d=self.conv3d, out_channels=self.out_channels,  dtype=self.dtype,)
        out_channels = self.in_channels if self.out_channels is None else self.out_channels

        self.conv1 = make_conv('3x3', conv3d=self.conv3d, out_channels=self.out_channels, dtype=self.dtype,)

        self.time_emb_proj = make_conv('dense',conv3d=self.conv3d, out_channels=self.out_channels, dtype=self.dtype,) 
        
        self.conv2 = make_conv('3x3', conv3d=self.conv3d, out_channels=self.out_channels, dtype=self.dtype,)

        if self.act_fn == "silu":
            self.act = nn.swish # ldm setting
        elif self.act_fn == "rcolu":
            self.act = rcolu
        elif self.act_fn == "colu":
            self.act = colu
        elif self.act_fn == "relu":
            self.act = nn.relu
        else:
            raise NotImplementedError #ValueError(f"Unsupported activation function: {self.act_fn}")

    def __call__(self, hidden_states, temb, deterministic=True):
        skip = hidden_states # not residual!
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.conv1(hidden_states)
        hidden_states = nn.with_logical_constraint(
            hidden_states,
            ('batch', 'keep_1', 'keep_2', 'out_channels')
        )
        
        temb = self.time_emb_proj(self.act(temb))
        temb = temb[...,None,None,:]
        hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic)
        hidden_states = self.conv2(hidden_states)
        hidden_states = nn.with_logical_constraint(
            hidden_states,
            ('batch', 'keep_1', 'keep_2', 'out_channels')
        )

        if self.conv_shortcut is not None:
            skip = self.conv_shortcut(skip)

        return hidden_states + skip
