import functools
import jax
import jax.numpy as jnp
from jax import nn


@functools.partial(jax.jit, static_argnames=['channel_axis','variant','eps','num_groups','share_axis'])
def colu(input: jnp.ndarray, 
         channel_axis: int = -1, 
         variant: str = "soft", 
         eps: float = 1e-7, 
         num_groups: int = 1, 
         share_axis: bool = False
         ):
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
        y, x = jnp.split(input, [1], axis=channel_axis)
    else:
        assert num_channels % num_groups == 0, "Channel size must be a multiple of number of cones"
        y, x = jnp.split(input, [num_groups], axis=channel_axis)
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
    xn = jnp.linalg.norm(x,axis=channel_axis,keepdims=True) # NG1HW
    
    mask = y / (xn + eps) # NG1HW
    if variant == "sqrt":
        mask = jnp.sqrt(mask)
    elif variant == "log":
        mask = jnp.log(jnp.max(mask,0)+1)
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

@functools.partial(jax.jit, static_argnames=['scaling','eps'])
def rcolu_(x, scaling="constant",eps=1e-8):
    """x = w + v, v || e"""
    C = x.shape[-1]
    # e = jnp.ones(C) / jnp.sqrt(C)
    vn = jnp.sum(x,axis=-1,keepdims=True) / jnp.sqrt(C) # dot(x, e)
    v = jnp.repeat(vn,C,axis=-1) / jnp.sqrt(C) # outer(v, e)
    w = x - v
    wn = jnp.linalg.norm(w, axis=-1, keepdims=True)
    m = jnp.maximum(vn, 0.) / (wn + eps)
    m = jnp.minimum(m, 1.) 
    w_ = w * m # project onto cone
    x = v + w_
    
    return x

@functools.partial(jax.jit, static_argnames=['dim','num_groups','axis','scaling','eps'])
def rcolu(x,
          dim=8,
          num_groups=None,
          scaling='constant',
          axis=-1,
          eps=1e-7
          ):
    """dim=S, num_groups=S"""
    if len(x.shape) == 0:
        return x # edge case
    assert (dim is not None) ^ (num_groups is not None) # specify one of both
    if axis != -1: # not recommended on TPU
        x = jnp.moveaxis(x, axis, -1)
    shape = x.shape
    if dim is None:
        assert shape[-1] % num_groups == 0
        dim = shape[-1] // num_groups
    if num_groups is None:
        assert shape[-1] % dim == 0
        num_groups = shape[-1] // dim
    new_shape = x.shape[:-1] + (num_groups, dim)
    x = x.reshape(new_shape)
    x = rcolu_(x,scaling,eps)
    x = x.reshape(shape)
    if axis != -1:
        x = jnp.moveaxis(x, -1, axis)
    return x

# # some test
# x = jnp.zeros(6).at[0].set(1)
# y = rcolu(x,dim=3)
# y.shape
# # some assertion 
# y1 = jnp.sum(y,axis=-1,keepdims=True) / jnp.sqrt(3) # dot(y, e)
# jnp.linalg.norm(y) / y1