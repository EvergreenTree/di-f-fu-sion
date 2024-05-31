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
import flax.linen as nn
import jax
import jax.numpy as jnp
from functools import partial


class FlaxUpsample2D(nn.Module):
    out_channels: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.conv = nn.Conv(
            self.out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

    def __call__(self, hidden_states):
        batch, height, width, channels = hidden_states.shape
        hidden_states = jax.image.resize(
            hidden_states,
            shape=(batch, height * 2, width * 2, channels),
            method="nearest",
        )
        hidden_states = self.conv(hidden_states)
        return hidden_states


class FlaxDownsample2D(nn.Module):
    out_channels: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.conv = nn.Conv(
            self.out_channels,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=((1, 1), (1, 1)),  # padding="VALID",
            dtype=self.dtype,
        )

    def __call__(self, hidden_states):
        # pad = ((0, 0), (0, 1), (0, 1), (0, 0))  # pad height and width dim
        # hidden_states = jnp.pad(hidden_states, pad_width=pad)
        hidden_states = self.conv(hidden_states)
        return hidden_states
    

class FlaxResnetBlock2D(nn.Module):
    in_channels: int
    out_channels: int = None
    dropout_prob: float = 0.0
    use_nin_shortcut: bool = None
    act_fn: str = "silu"
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        out_channels = self.in_channels if self.out_channels is None else self.out_channels

        self.act = nn.swish if self.act_fn == "silu" else group_colu
        self.norm1 = nn.GroupNorm(num_groups=32, epsilon=1e-5)
        self.conv1 = nn.Conv(
            out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

        self.time_emb_proj = nn.Dense(out_channels, dtype=self.dtype)

        self.norm2 = nn.GroupNorm(num_groups=32, epsilon=1e-5) 
        self.dropout = nn.Dropout(self.dropout_prob)
        self.conv2 = nn.Conv(
            out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

        use_nin_shortcut = self.in_channels != out_channels if self.use_nin_shortcut is None else self.use_nin_shortcut

        self.conv_shortcut = None
        if use_nin_shortcut:
            self.conv_shortcut = nn.Conv(
                out_channels,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="VALID",
                dtype=self.dtype,
            )

    def __call__(self, hidden_states, temb, deterministic=True):
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.conv1(hidden_states)

        temb = self.time_emb_proj(nn.swish(temb))
        temb = jnp.expand_dims(jnp.expand_dims(temb, 1), 1)
        hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)

        return hidden_states + residual

# @partial(jax.jit, static_argnames=['channel_axis','variant','eps'],)
# def group_colu(x, channel_axis = -1, variant = "soft", eps = 1e-7):
#     num_channels = x.shape[channel_axis]
#     y, x = x.take(jnp.arange(32), axis=channel_axis), x.take(jnp.arange(32,num_channels), axis=channel_axis)
#     num_channels = x.shape[channel_axis]
#     num_groups = y.shape[channel_axis] # number of cones
#     assert num_channels % num_groups == 0, "Input must be a multiple of number of cones"
#     group_size = num_channels // num_groups # S = C / G

#     assert channel_axis < 0, "channel_axis must be negative" # Comply with broadcasting on first dimensions
 
#     x_old_shape = x.shape
#     y_old_shape = y.shape
#     x_shape = x.shape[:channel_axis] + (num_groups, group_size)
#     y_shape = y.shape[:channel_axis] + (num_groups, 1)
#     if channel_axis < -1:
#         x_shape += x.shape[(channel_axis+1):] # NGSHW if channel_axis = -3
#         y_shape += y.shape[(channel_axis+1):] # NG1HW
#     x = x.reshape(x_shape)
#     y = y.reshape(y_shape)

#     xn = jnp.linalg.norm(x,axis=channel_axis,keepdims=True) # NG1HW, per-group norm, or the S dimension

#     mask = y / (xn + eps) # NG1HW

#     if variant == "soft": # soft project
#         scaled_mask = nn.sigmoid(mask-.5)
#     elif variant == "hard": # hard project
#         scaled_mask = mask.clip(0,1)
#     else:
#         raise NotImplementedError("variant must be soft or hard.")

#     x = scaled_mask * x # NGSHW

#     x = x.reshape(x_old_shape)
#     y = y.reshape(y_old_shape)

#     return jnp.concatenate([y,x],axis=channel_axis)
@partial(jax.jit, static_argnames=['channel_axis','variant','eps','num_groups','project_axes','share_axis'])
def group_colu(input, channel_axis = -1, variant = "soft", eps = 1e-7, num_groups = 32, project_axes = False, share_axis = False):
    """project the input x onto the axes dimension"""
    """output dimension = S = axes + cone sections = [len=(G or 1)] + G * [len=(S-1)]"""
    if num_groups == 0: # trivial case
        return input
    num_channels = input.shape[channel_axis]
    if (share_axis and num_groups == num_channels - 1) or (not share_axis and num_groups * 2 == num_channels): # pointwise case
        return nn.silu(input) if variant == "soft" else nn.relu(input)
    group_size = (num_channels - 1) // num_groups + 1 if share_axis else num_channels // num_groups
        
    # y = axes, x = cone sections
    if share_axis:
        assert (num_channels - 1) % num_groups == 0, "Channel size must be a multiple of number of cones plus one"
        y, x = input.take(jnp.arange(1), axis=channel_axis), input.take(jnp.arange(1,num_channels), axis=channel_axis)
    else:
        assert num_channels % num_groups == 0, "Channel size must be a multiple of number of cones"
        y, x = input.take(jnp.arange(num_groups), axis=channel_axis), input.take(jnp.arange(num_groups,num_channels), axis=channel_axis)
        group_size = num_channels // num_groups # S = C / G

    assert channel_axis < 0, "channel_axis must be negative" # Comply with broadcasting on first dimensions
    x_old_shape = x.shape
    y_old_shape = y.shape
    x_shape = x.shape[:channel_axis] + (num_groups, group_size - 1) # NG(S-1)
    if share_axis:
        y_shape = y.shape[:channel_axis] + (1, 1) # N11
    else: 
        y_shape = y.shape[:channel_axis] + (num_groups, 1) # NG1
    if channel_axis < -1:
        x_shape += x.shape[(channel_axis+1):] # NGSHW if channel_axis = -3
        y_shape += y.shape[(channel_axis+1):] # NG1HW
    x = x.reshape(x_shape)
    y = y.reshape(y_shape)
    xn = jnp.linalg.norm(x,axis=channel_axis,keepdims=True) # NG1HW, norm

    if project_axes:
        assert not share_axis, "shuffle_axes is not compatible with share_axis"
        y0, y1 = y.take(jnp.arange(1), axis=channel_axis-1), y.take(jnp.arange(1,num_groups), axis=channel_axis-1) # N11HW, N(G-1)1HW
        yn = jnp.linalg.norm(y1,axis=channel_axis-1,keepdims=True) # N11HW
        ymask = y0 / (yn + eps) # N11HW
        ymask = nn.sigmoid(ymask-.5) if variant == "soft" else ymask.clip(0,1)
        y1 = ymask * y1 # N(G-1)1HW
        y = jnp.concatenate([y0,y1],axis=channel_axis-1)
    
    mask = y / (xn + eps) # NG1HW
    if variant == "softmax":
        mask = nn.softmax(mask, axis=channel_axis)
    elif variant == "softapprox":
        mask = nn.sigmoid(4*mask-2)
    elif variant == "soft":
        mask = nn.sigmoid(mask - .5)
    elif variant == "hard":
        mask = mask.clip(0,1)
    else:
        raise NotImplementedError("variant must be soft or hard.")

    x = mask * x # NGSHW
    x = x.reshape(x_old_shape)
    y = y.reshape(y_old_shape)
    output = jnp.concatenate([y,x],axis=channel_axis)

    return output