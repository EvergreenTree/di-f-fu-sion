import functools
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Callable

@functools.partial(jax.jit, static_argnames=['channel_axis','scaling','eps','num_groups','dim','share_axis'])
def colu(input: jax.Array, 
         channel_axis: int = -1, 
         scaling: str = "hard", 
         eps: float = 1e-7, 
         num_groups: Optional[int] = None, 
         dim: Optional[int] = 4, 
         share_axis: bool = False
         ) -> jax.Array:
    """project the input x onto the axes dimension"""
    """G=number of cones, S=dim of cones"""
    """output dimension = C = axes + cone sections = [len=(G or 1)] + G * [len=(S-1)]"""
    """jnp.moveaxis is avoided to optimize speed on TPU"""
    shape = input.shape
    if len(shape) == 0:
        return input # edge case
    assert (dim is not None) ^ (num_groups is not None) # specify one of both, infer the other

    if share_axis:
        if dim is None:
            assert (shape[channel_axis] - 1) % num_groups == 0
            dim = (shape[channel_axis] - 1) // num_groups + 1
        if num_groups is None:
            assert (shape[channel_axis] - 1) % (dim - 1) == 0
            num_groups = (shape[channel_axis] - 1) // (dim - 1)
    else:
        if dim is None:
            assert shape[channel_axis] % num_groups == 0
            dim = shape[channel_axis] // num_groups
        if num_groups is None:
            assert shape[channel_axis] % dim == 0
            num_groups = shape[channel_axis] // dim

    if dim == 2: # pointwise case
        return nn.silu(input) if scaling == "soft" else nn.relu(input)
        
    # y = axes, x = cone sections
    if share_axis:
        y, x = jnp.split(input, [1], axis=channel_axis)
    else:
        y, x = jnp.split(input, [num_groups], axis=channel_axis)

    assert channel_axis < 0, "channel_axis must be negative" # Comply with broadcasting on first dimensions
    x_old_shape = x.shape
    y_old_shape = y.shape
    x_shape = x.shape[:channel_axis] + (num_groups, dim - 1) # NG(S-1)
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
    if scaling == "sqrt":
        mask = jnp.sqrt(mask)
    elif scaling == "log":
        mask = jnp.log(jnp.max(mask,0)+1)
    elif scaling == "soft": # TODO: there's a discontinuity of x1/x_1 at 0
        mask = nn.sigmoid(mask - .5)
    elif scaling == "hard":
        mask = mask.clip(0,1)
    else:
        raise NotImplementedError

    x = mask * x # NGSHW
    x = x.reshape(x_old_shape)
    y = y.reshape(y_old_shape)
    output = jnp.concatenate([y,x],axis=channel_axis)

    return output

@functools.partial(jax.jit, static_argnames=['scaling','eps'])
def rcolu_(x: jax.Array, scaling: str="hard",eps: float=1e-8) -> jax.Array:
    """x = w + v, v || e"""
    C = x.shape[-1]
    # e = jnp.ones(C) / jnp.sqrt(C)
    vn = jnp.sum(x,axis=-1,keepdims=True) / jnp.sqrt(C) # dot(x, e)
    v = jnp.repeat(vn,C,axis=-1) / jnp.sqrt(C) # outer(v, e)
    w = x - v
    wn = jnp.linalg.norm(w, axis=-1, keepdims=True)
    m = jnp.maximum(vn, 0.) / (wn + eps)
    if scaling == 'hard':
        m = jnp.minimum(m, 1.) 
    elif scaling == "soft": # TODO: there's a discontinuity of x1/x_1 at 0
        m = nn.sigmoid(m - .5)
    else:
        raise NotImplementedError
    w_ = w * m # project onto cone
    x = v + w_
    
    return x

@functools.partial(jax.jit, static_argnames=['dim','num_groups','axis','scaling','eps'])
def rcolu(x: jax.Array,
          dim: Optional[int]=4,
          num_groups: Optional[int]=None,
          scaling: str='hard',
          axis: int=-1,
          eps: float=1e-7
          ) -> jax.Array:
    """dim=S, num_groups=G"""
    if len(x.shape) == 0:
        return x
    assert (dim is not None) ^ (num_groups is not None) # specify one and only one of both
    shape = x.shape
    if dim is None:
        assert shape[-1] % num_groups == 0
        dim = shape[-1] // num_groups
    if num_groups is None:
        assert shape[-1] % dim == 0
        num_groups = shape[-1] // dim
    if axis != -1: 
        x = jnp.moveaxis(x, axis, -1)
    new_shape = x.shape[:-1] + (num_groups, dim)
    x = x.reshape(new_shape)
    x = rcolu_(x,scaling,eps)
    x = x.reshape(shape)
    if axis != -1:
        x = jnp.moveaxis(x, -1, axis)
    return x


# Option 1: this should be combined by jitting the resulting function as follows (deprecated)
# jax.jit(functools.partial(apply_conv, conv_fn),static_argnames=['conv3d'])
# def apply_conv(conv_fn,x,conv3d=False):
#     if conv3d:
#         x_shape = x.shape
#         assert x_shape[-1] % 4 == 0
#         x_new_shape = x_shape[:-1] + (x_shape[-1]//4, 4)
#         x = x.reshape(x_new_shape)
#         x = conv_fn(x)
#         x = x.reshape(x_shape)
#         return x
#     else:
#         return conv_fn(x)

# Option 2: use closure (deprecated)
# def make_conv(conv_fn:Callable,conv3d:bool=False):
#     if conv3d:
#         @jax.jit
#         def conv(x):
#             x_shape = x.shape
#             assert x_shape[-1] % 4 == 0
#             x_new_shape = x_shape[:-1] + (x_shape[-1]//4, 4)
#             x = x.reshape(x_new_shape)
#             x = conv_fn(x)
#             x = x.reshape(x_shape)
#             return x
#     else:
#         @jax.jit
#         def conv(x):
#             return conv_fn(x)
#     return conv 

# Option 3: inherit nn.Module class
class PolyConv(nn.Conv):
    conv3d: bool = False
    dim: Optional[int] = 4 # input tangent space dimension
    # note: `features` is the output tangent space dimension
    down: int = 1

    def __call__(self,x,**kwargs):
        if self.conv3d:
            x_shape = x.shape
            assert x_shape[-1] % self.dim == 0
            x_new_shape = x_shape[:-1] + (x_shape[-1] // self.dim, self.dim)
            x = x.reshape(x_new_shape) # imagine a tangent space...
            x = super().__call__(x,**kwargs)
            if self.down != 1: # int(stride): downsample rate
                assert x_shape[-3] % self.down == 0 and x_shape[-2] % self.down == 0 # make sure we are on the same page
                x_shape = x_shape[:-3] + (x_shape[-3]//self.down,x_shape[-2]//self.down,x_shape[-1],)
            if self.dim != self.features: # channel resampling!
                x_shape = x_shape[:-1] + (x_shape[-1] // self.dim * self.features,)
            x = x.reshape(x_shape) # go back to reality...
            return x
        else:
            return super().__call__(x,**kwargs)

class WrappedDense(nn.Dense):
    conv3d: bool = False
    dim: Optional[int] = None

# preset configs
def make_conv(method: str, conv3d: bool, out_channels: int, in_channels: Optional[int] = None, **kwargs) -> nn.Module:
    # if in_channels and in_channels != out_channels: # in_channels checking dropped! thanks to follow ups on channel resizing
    #     conv3d = False
    if method == '3x3':
        if conv3d:
            return PolyConv(
                features=4,
                kernel_size=(3, 3, 3),
                strides=(1, 1, 1),
                padding=((1, 1), (1, 1), (1, 1)),
                kernel_init = nn.with_logical_partitioning(
                    nn.initializers.glorot_normal(),
                    ('keep_1', 'keep_2', 'keep_3', 'conv_in', 'conv_out')
                ),
                conv3d=conv3d,
                dim=4,
                **kwargs
            )
        else: 
            return PolyConv(
                features=out_channels,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding=((1, 1), (1, 1)),
                kernel_init = nn.with_logical_partitioning(
                    nn.initializers.glorot_normal(),
                    ('keep_1', 'keep_2', 'conv_in', 'conv_out')
                ),
                conv3d=conv3d,
                dim=None,
                **kwargs
            )
    elif method == 'down':
        if conv3d:
            return PolyConv(
                features=4,
                kernel_size=(3, 3, 3),
                strides=(2, 2, 1),
                padding=((1, 1), (1, 1), (1, 1)),  # padding="VALID",
                kernel_init = nn.with_logical_partitioning(
                    nn.initializers.glorot_normal(),
                    ('keep_1', 'keep_2', 'keep_3', 'conv_in', 'conv_out')
                ),
                conv3d=conv3d,
                dim=4,
                down=2,
                **kwargs
            )
        else:
            return PolyConv(
                out_channels,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding=((1, 1), (1, 1)),  # padding="VALID",
                kernel_init = nn.with_logical_partitioning(
                    nn.initializers.glorot_normal(),
                    ('keep_1', 'keep_2', 'conv_in', 'conv_out')
                ),
                conv3d=conv3d,
                dim=None,
                **kwargs
            )
    elif method == '1x1': # used in attention
        if conv3d:
            return PolyConv(
                features=4,
                kernel_size=(1, 1, 3),
                strides=(1, 1, 1),
                padding=(0, 0, 1),  # padding="VALID",
                kernel_init = nn.with_logical_partitioning(
                    nn.initializers.glorot_normal(),
                    ('keep_1', 'keep_2', 'keep_3', 'conv_in', 'conv_out')
                ),
                conv3d=conv3d,
                dim=4,
                **kwargs
            )
        else:
            return PolyConv(
                out_channels,
                kernel_size=(1, 1),
                strides=(1, 1),
                kernel_init = nn.with_logical_partitioning(
                    nn.initializers.glorot_normal(),
                    ('keep_1', 'keep_2', 'conv_in', 'conv_out')
                ),
                conv3d=conv3d,
                dim=None,
                **kwargs
            )
    elif method == 'dense':
        if conv3d:
            return PolyConv(
                features=4,
                kernel_size=(3,),
                strides=(1,),
                padding=((1, 1),),
                kernel_init=nn.with_logical_partitioning(
                    nn.initializers.glorot_normal(),
                    ('keep_1', "conv_in", "conv_out")
                ),
                use_bias=False,
                conv3d=conv3d,
                dim=4,
                **kwargs
            )
        else:
            return WrappedDense(
                out_channels,
                kernel_init=nn.with_logical_partitioning(
                    nn.initializers.glorot_normal(),
                    ("conv_in", "conv_out") # don't use ("embed", "heads"). unify!
                ),
                use_bias=False,
                conv3d=conv3d,
                dim=None,
                **kwargs
            )
    elif method == 'concave': # channel x8 for GEGLU in MLP in Transformer
        if conv3d:
            return PolyConv(
                features=32,
                kernel_size=(3,),
                strides=(1,),
                padding=((1, 1),),
                kernel_init=nn.with_logical_partitioning(
                    nn.initializers.glorot_normal(),
                    ('keep_1', "conv_in", "conv_out")
                ),
                use_bias=False,
                conv3d=conv3d,
                dim=4, 
                **kwargs
            )
        else:
            return WrappedDense(
                out_channels,
                kernel_init=nn.with_logical_partitioning(
                    nn.initializers.glorot_normal(),
                    ("conv_in", "conv_out") 
                ),
                use_bias=False,
                conv3d=conv3d,
                dim=None,
                **kwargs
            )
    elif method == 'convex': # channel /4
        if conv3d:
            return PolyConv(
                features=4,
                kernel_size=(3,),
                strides=(1,),
                padding=((1, 1),),
                kernel_init=nn.with_logical_partitioning(
                    nn.initializers.glorot_normal(),
                    ('keep_1', "conv_in", "conv_out")
                ),
                use_bias=False,
                conv3d=conv3d,
                dim=16, 
                **kwargs
            )
        else:
            return WrappedDense(
                out_channels,
                kernel_init=nn.with_logical_partitioning(
                    nn.initializers.glorot_normal(),
                    ("conv_in", "conv_out") 
                ),
                use_bias=False,
                conv3d=conv3d,
                dim=None,
                **kwargs
            )
    else: raise NotImplementedError