import os
import inspect
from typing import Optional, Any
import pickle
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn


Array = jax.Array


class NetLinLayer(nn.Module):
    features: int = 1
    use_dropout: bool = False
    training: bool = False

    @nn.compact
    def __call__(self, x):
        if self.use_dropout:
            x = nn.Dropout(rate=0.5)(x, deterministic=not self.training)
        x = nn.Conv(self.features, (1, 1), padding=0, use_bias=False)(x)
        return x 


class AlexNet(nn.Module):

    @nn.compact
    def __call__(self, x: Array):
        x = nn.Conv(64, (11, 11), strides=(4, 4), padding=(2, 2))(x)
        x = nn.relu(x)
        relu_1 = x
        x = nn.max_pool(x, (3, 3), strides=(2, 2))

        x = nn.Conv(192, (5, 5), padding=2)(x)
        x = nn.relu(x)
        relu_2 = x
        x = nn.max_pool(x, (3, 3), strides=(2, 2))

        x = nn.Conv(384, (3, 3), padding=1)(x)
        x = nn.relu(x)
        relu_3 = x
        x = nn.Conv(256, (3, 3), padding=1)(x)
        x = nn.relu(x)
        relu_4 = x
        x = nn.Conv(256, (3, 3), padding=1)(x)
        x = nn.relu(x)
        relu_5 = x

        return [relu_1, relu_2, relu_3, relu_4, relu_5]

        
class VGG16(nn.Module):
    act: bool = True
    
    @nn.compact
    def __call__(self, x: Array):
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 
               'M', 512, 512, 512, 'M', 512, 512, 512]

        layer_ids = [1, 4, 8, 12, 16]
        out = []
        for i, v in enumerate(cfg):
            if v == 'M':
                x = nn.max_pool(x, (2, 2,), strides=(2, 2))
            else:
                x = nn.Conv(v, (3, 3), padding=(1, 1))(x)
                if self.act:
                    x = nn.relu(x)
                if i in layer_ids:
                    out.append(x)
                if not self.act:
                    x = nn.relu(x)
        return out


Dtype = Any


class LPIPSEvaluator:
    def __init__(self, replicate=True, pretrained=True, net='alexnet', lpips=True,
                 use_dropout=True, dtype=jnp.float32):
        self.lpips = LPIPS(pretrained, net, lpips, use_dropout,
                           training=False, dtype=dtype)
        model_path = os.path.abspath(os.path.join(inspect.getfile(self.__init__), '..', f'weights/{net}.ckpt'))
        self.params = pickle.load(open(model_path, 'rb'))
        if replicate:
            self.params = flax.jax_utils.replicate(self.params)
        self.params = dict(params=self.params)
        
        self.replicate = replicate
    
    def __call__(self, images_0, images_1):
        fn = jax.pmap(self.lpips.apply) if self.replicate else self.lpips.apply
        return fn(
            self.params,
            images_0,
            images_1
        ) 


class LPIPS(nn.Module):
    pretrained: bool = True
    net_type: str = 'vgg16'
    lpips: bool = True
    use_dropout: bool = True
    training: bool = False
    dtype: Optional[Dtype] = jnp.float32

    @nn.compact
    def __call__(self, images_0, images_1):
        shift = jnp.array([-0.030, -0.088, -0.188], dtype=self.dtype)
        scale = jnp.array([0.458, 0.448, 0.450], dtype=self.dtype)
        images_0 = (images_0 - shift) / scale
        images_1 = (images_1 - shift) / scale
        
        if self.net_type == 'alexnet':
            net = AlexNet()
        elif self.net_type == 'vgg16':
            net = VGG16()
        else:
            raise ValueError(f'Unsupported net_type: {self.net_type}. Must be in [alexnet, vgg16]')
        
        outs_0, outs_1 = net(images_0), net(images_1)
        diffs = []
        for feat_0, feat_1 in zip(outs_0, outs_1):
            diff = (normalize(feat_0) - normalize(feat_1)) ** 2
            diffs.append(diff)
        
        res = []
        for d in diffs:
            if self.lpips:
                d = NetLinLayer(use_dropout=self.use_dropout)(d)
            else:
                d = jnp.sum(d, axis=-1, keepdims=True)
            d = spatial_average(d, keepdims=True)
            res.append(d)

        val = sum(res)
        return val


def spatial_average(feat, keepdims=True):
    return jnp.mean(feat, axis=[1, 2], keepdims=keepdims)


def normalize(feat, eps=1e-10):
    norm_factor = jnp.sqrt(jnp.sum(feat ** 2, axis=-1, keepdims=True))
    return feat / (norm_factor + eps)

if __name__ == '__main__':
    lpips = LPIPSEvaluator()
    a = jnp.ones((8,32,32,3))
    a = flax.jax_utils.replicate(a)
    l = lpips(a,a)
    print(l.shape)