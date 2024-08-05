import functools
import time
import os

from einops import repeat

from absl import logging
from clu import metric_writers
from clu import periodic_actions
from typing import Any, Optional, Callable
from tqdm import tqdm, trange

import flax
from flax.training import train_state
from flax.training import common_utils
from flax.training import dynamic_scale as dynamic_scale_lib
from flax.training import checkpoints
from flax import jax_utils

import optax

import jax.numpy as jnp
import numpy as np
import jax 

import ml_collections
import tensorflow as tf
import tensorflow_datasets as tfds

import wandb

from maxdiffusion import FlaxUNet2DConditionModel


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


def noise_to_x0(noise, xt, batched_t, ddpm):
    assert batched_t.shape[0] == xt.shape[0] == noise.shape[0] # make sure all has batch dimension
    sqrt_alpha_bar = ddpm['sqrt_alphas_bar'][batched_t, None, None, None]
    alpha_bar= ddpm['alphas_bar'][batched_t, None, None, None]
    x0 = 1. / sqrt_alpha_bar * xt -  jnp.sqrt(1./alpha_bar-1) * noise
    return x0


def x0_to_noise(x0, xt, batched_t, ddpm):
    assert batched_t.shape[0] == xt.shape[0] == x0.shape[0] # make sure all has batch dimension
    sqrt_alpha_bar = ddpm['sqrt_alphas_bar'][batched_t, None, None, None]
    alpha_bar= ddpm['alphas_bar'][batched_t, None, None, None]
    noise = (1. / sqrt_alpha_bar * xt - x0) /jnp.sqrt(1./alpha_bar-1)
    return noise


def get_posterior_mean_variance(img, t, x0, v, ddpm_params):

    beta = ddpm_params['betas'][t, None,None,None]
    alpha = ddpm_params['alphas'][t, None,None,None]
    alpha_bar = ddpm_params['alphas_bar'][t, None,None,None]
    alpha_bar_last = ddpm_params['alphas_bar'][t-1, None,None,None]
    sqrt_alpha_bar_last = ddpm_params['sqrt_alphas_bar'][t-1, None,None,None]

    # only needed when t > 0
    coef_x0 = beta * sqrt_alpha_bar_last / (1. - alpha_bar)
    coef_xt = (1. - alpha_bar_last) * jnp.sqrt(alpha) / ( 1- alpha_bar)        
    posterior_mean = coef_x0 * x0 + coef_xt * img
        
    posterior_variance = beta * (1 - alpha_bar_last) / (1. - alpha_bar)
    posterior_log_variance = jnp.log(jnp.clip(posterior_variance, a_min = 1e-20))

    return posterior_mean, posterior_log_variance


# called by p_loss and ddpm_sample_step - both use pmap
def model_predict(state, x, x0, t, ddpm_params, self_condition, is_pred_x0, use_ema=True):
    if use_ema:
        variables = {'params': state.params_ema}
    else:
        variables = {'params': state.params}
    
    if self_condition:
        pred = state.apply_fn(variables, jnp.concatenate([x, x0],axis=-3), t).sample # fixed for HF unet
    else:
        pred = state.apply_fn(variables, x, t).sample # fixed for HF unet

    if is_pred_x0: # if the objective is is_pred_x0, pred == x0_pred
        x0_pred = pred
        noise_pred =  x0_to_noise(pred, x, t, ddpm_params)
    else:
        noise_pred = pred
        x0_pred = noise_to_x0(pred, x, t, ddpm_params)
    
    return x0_pred, noise_pred


def ddpm_sample_step(state, rng, x, t, x0_last, ddpm_params, self_condition=False, is_pred_x0=False):

    batched_t = jnp.ones((x.shape[0],), dtype=jnp.int32) * t
    
    if self_condition:
        x0, v = model_predict(state, x, x0_last, batched_t, ddpm_params, self_condition, is_pred_x0, use_ema=True) 
    else:
        x0, v = model_predict(state, x, None, batched_t,ddpm_params, self_condition, is_pred_x0, use_ema=True)
    
    # make sure x0 between [-1,1]
    x0 = jnp.clip(x0, -1., 1.)

    posterior_mean, posterior_log_variance = get_posterior_mean_variance(x, t, x0, v, ddpm_params)
    x = posterior_mean + jnp.exp(0.5 *  posterior_log_variance) * jax.random.normal(rng, x.shape) 

    return x, x0


def sample_loop(rng, state, shape, p_sample_step, timesteps):
    
    # shape include the device dimension: (device, per_device_batch_size, H,W,C)
    rng, x_rng = jax.random.split(rng)
    list_x0 = []
    # generate the initial sample (pure noise)
    x = jax.random.normal(x_rng, shape)
    x0 = jnp.zeros_like(x) # initialize x0 for self-conditioning
    # sample step
    for t in reversed(jnp.arange(timesteps)):
        rng, *step_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
        step_rng = jnp.asarray(step_rng)
        x, x0 = p_sample_step(state, step_rng, x, jax_utils.replicate(t), x0)
        list_x0.append(x0)
    # normalize to [0,1]
    img = unnormalize_to_zero_to_one(jnp.asarray(x0))

    return img



def flatten(x):
  return x.reshape(x.shape[0], -1)

def l2_loss(logit, target):
    return (logit - target)**2

def l1_loss(logit, target): 
    return jnp.abs(logit - target)

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def crop_resize(image, resolution):
  crop = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
  h, w = tf.shape(image)[0], tf.shape(image)[1]
  image = image[(h - crop) // 2:(h + crop) // 2,
                (w - crop) // 2:(w + crop) // 2]
  image = tf.image.resize(
      image,
      size=(resolution, resolution),
      antialias=True,
      method=tf.image.ResizeMethod.BICUBIC)
  return tf.cast(image, tf.uint8)


def get_dataset(rng, config):
    
    if config.data.batch_size % jax.device_count() > 0:
        raise ValueError('Batch size must be divisible by the number of devices')
    
    batch_size = config.data.batch_size //jax.process_count()

    platform = jax.local_devices()[0].platform
    if config.training.half_precision:
        if platform == 'tpu':
            input_dtype = tf.bfloat16
        else:
            input_dtype = tf.float16
    else: input_dtype = tf.float32

    dataset_builder = tfds.builder(config.data.dataset)
    dataset_builder.download_and_prepare()

    def preprocess_fn(d):
        img = d['image']
        img = crop_resize(img, config.data.image_size)
        img = tf.image.flip_left_right(img)
        img= tf.image.convert_image_dtype(img, input_dtype)
        return({'image':img})
    
    # create split for current process 
    train_examples = dataset_builder.info.splits['train'].num_examples
    split_size = train_examples // jax.process_count()
    start = jax.process_index() * split_size
    split = f'train[{start}:{start + split_size}]'

    ds = dataset_builder.as_dataset(split=split)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 48
    ds.with_options(options)

    if config.data.cache:
        ds= ds.cache()

    ds = ds.repeat()
    ds = ds.shuffle(16 * batch_size , seed=0)

    ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # (local_devices * device_batch_size), height, width, c
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(10)
    # multihost training
    ds = ds.shard(jax.process_count(), jax.process_index())

    def scale_and_reshape(xs):
        local_device_count = jax.local_device_count()
        def _scale_and_reshape(x):
           # Use _numpy() for zero-copy conversion between TF and NumPy.
           x = x._numpy()  # pylint: disable=protected-access
           # normalize to [-1,1]
           x = normalize_to_neg_one_to_one(x)
           # move axis for diffuser unet
           x = jnp.moveaxis(x,-1,-3) 
           # reshape (batch_size, height, width, channels) to
           # (local_devices, device_batch_size, 3, height, width)
           return x.reshape((local_device_count, -1) + x.shape[1:])

        return jax.tree_map(_scale_and_reshape, xs)

    it = map(scale_and_reshape, ds)
    it = jax_utils.prefetch_to_device(it, 2)

    return it


def create_model(*, model_cls, half_precision, **kwargs):
  platform = jax.local_devices()[0].platform
  if half_precision:
    if platform == 'tpu':
      model_dtype = jnp.bfloat16
    else:
      model_dtype = jnp.float16
  else:
    model_dtype = jnp.float32
  return model_cls(dtype=model_dtype, **kwargs)


def initialized(key, model):
  return jax.jit(model.init_weights)(key)


class TrainState(train_state.TrainState):
  params_ema: Any = None
  dynamic_scale: Optional[dynamic_scale_lib.DynamicScale] = None


def create_train_state(rng, config: ml_collections.ConfigDict):
  """Creates initial `TrainState`."""

  dynamic_scale = None
  platform = jax.local_devices()[0].platform

  if config.training.half_precision and platform == 'gpu':
    dynamic_scale = dynamic_scale_lib.DynamicScale()
  else:
    dynamic_scale = None

  model = create_model(
      model_cls=FlaxUNet2DConditionModel, 
      half_precision=config.training.half_precision,
      conditional=False, # added to adapt to HF implementation
      sample_size=config.data.image_size,
      in_channels=config.data.channels * 2 if config.ddpm.self_condition else config.data.channels,
      out_channels=config.data.channels,
      block_out_channels=config.model.block_out_channels,
      layers_per_block=config.model.layers_per_block,
      act_fn=config.model.act_fn,
      )

  params = initialized(rng, model)

  tx = create_optimizer(config.optim)

  state = TrainState.create(
      apply_fn=model.apply, 
      params=params, 
      tx=tx, 
      params_ema=params,
      dynamic_scale=dynamic_scale)

  return state


def create_optimizer(config):

    if config.optimizer == 'Adam':
        optimizer = optax.adam(
            learning_rate = config.lr , b1=config.beta1, b2 = config.beta2, 
            eps=config.eps)
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!')

    return optimizer


def get_loss_fn(config):

    if config.training.loss_type == 'l1' :
        loss_fn = l1_loss
    elif config.training.loss_type == 'l2':
        loss_fn = l2_loss
    else:
        raise NotImplementedError(
           f'loss_type {config.training.loss_tyoe} not supported yet!')

    return loss_fn


def create_ema_decay_schedule(config):

    def ema_decay_schedule(step):
        count = jnp.clip(step - config.update_after_step - 1, a_min = 0.)
        value = 1 - (1 + count / config.inv_gamma) ** - config.power 
        ema_rate = jnp.clip(value, a_min = config.min_value, a_max = config.beta)
        return ema_rate

    return ema_decay_schedule


def q_sample(x, t, noise, ddpm_params):

    sqrt_alpha_bar = ddpm_params['sqrt_alphas_bar'][t, None, None, None]
    sqrt_1m_alpha_bar = ddpm_params['sqrt_1m_alphas_bar'][t,None,None,None]
    x_t = sqrt_alpha_bar * x + sqrt_1m_alpha_bar * noise

    return x_t


# train step
def p_loss(rng, state, batch, ddpm_params, loss_fn, self_condition=False, is_pred_x0=False, pmap_axis='batch'):
    
    # run the forward diffusion process to generate noisy image x_t at timestep t
    x = batch['image']
    assert x.dtype in [jnp.float32, jnp.float64]
    
    # create batched timesteps: t with shape (B,)
    B, H, W, C = x.shape
    rng, t_rng = jax.random.split(rng)
    batched_t = jax.random.randint(t_rng, shape=(B,), dtype = jnp.int32, minval=0, maxval= len(ddpm_params['betas']))
   
    # sample a noise (input for q_sample)
    rng, noise_rng = jax.random.split(rng)
    noise = jax.random.normal(noise_rng, x.shape)
    # if is_pred_x0 == True, the target for loss calculation is x, else noise
    target = x if is_pred_x0 else noise

    # generate the noisy image (input for denoise model)
    x_t = q_sample(x, batched_t, noise, ddpm_params)
    
    # if doing self-conditioning, 50% of the time first estimate x_0 = f(x_t, 0, t) and then use the estimated x_0 for Self-Conditioning
    # we don't backpropagate through the estimated x_0 (exclude from the loss calculation)
    # this technique will slow down training by 25%, but seems to lower FID significantly  
    if self_condition:

        rng, condition_rng = jax.random.split(rng)
        zeros = jnp.zeros_like(x_t)

        # self-conditioning 
        def estimate_x0(_):
            x0, _ = model_predict(state, x_t, zeros, batched_t, ddpm_params, self_condition, is_pred_x0, use_ema=False)
            return x0

        x0 = jax.lax.cond(
            jax.random.uniform(condition_rng, shape=(1,))[0] < 0.5,
            estimate_x0,
            lambda _ :zeros,
            None)
                
        x_t = jnp.concatenate([x_t, x0], axis=-3) # fixed for HF unet
    
    p2_loss_weight = ddpm_params['p2_loss_weight']

    def compute_loss(params):
        pred = state.apply_fn({'params':params}, x_t, batched_t).sample
        loss = loss_fn(flatten(pred),flatten(target))
        loss = jnp.mean(loss, axis= 1)
        assert loss.shape == (B,)
        loss = loss * p2_loss_weight[batched_t]
        return loss.mean()
    
    dynamic_scale = state.dynamic_scale

    if dynamic_scale:
        grad_fn = dynamic_scale.value_and_grad(compute_loss, axis_name=pmap_axis)
        dynamic_scale, is_fin, loss, grads = grad_fn(state.params)
        # dynamic loss takes care of averaging gradients across replicas
    else:
        grad_fn = jax.value_and_grad(compute_loss)
        loss, grads = grad_fn(state.params)
        #  Re-use same axis_name as in the call to `pmap(...train_step,axis=...)` in the train function
        grads = jax.lax.pmean(grads, axis_name=pmap_axis)
    
    loss = jax.lax.pmean(loss, axis_name=pmap_axis)
    loss_ema = jax.lax.pmean(compute_loss(state.params_ema), axis_name=pmap_axis)

    metrics = {'loss': loss,
               'loss_ema': loss_ema}

    new_state = state.apply_gradients(grads=grads)

    if dynamic_scale:
    # if is_fin == False the gradients contain Inf/NaNs and optimizer state and
    # params should be restored (= skip this step).
        new_state = new_state.replace(
            opt_state=jax.tree_map(
                functools.partial(jnp.where, is_fin),
                new_state.opt_state,
                state.opt_state),
            params=jax.tree_map(
                functools.partial(jnp.where, is_fin),
                new_state.params,
                state.params),
            dynamic_scale=dynamic_scale)
        metrics['scale'] = dynamic_scale.scale
    
     
    return new_state, metrics



def copy_params_to_ema(state):
    state = state.replace(params_ema = state.params)
    return state

def apply_ema_decay(state, ema_decay):
    params_ema = jax.tree_map(lambda p_ema, p: p_ema * ema_decay + p * (1. - ema_decay), state.params_ema, state.params)
    state = state.replace(params_ema = params_ema)
    return state


def load_wandb_model(state, workdir, wandb_artifact):
    artifact = wandb.run.use_artifact(wandb_artifact, type='ddpm_model')
    artifact_dir = artifact.download(workdir)
    return checkpoints.restore_checkpoint(artifact_dir, state)


def restore_checkpoint(state, workdir):
    return checkpoints.restore_checkpoint(workdir, state)


def save_checkpoint(state, workdir):
    # get train state from the first replica
    state = jax.device_get(jax_utils.unreplicate(state))
    step = int(state.step)
    checkpoints.save_checkpoint_multiprocess(workdir, state, step, keep=1)


# utils
import jax.numpy as jnp
import numpy as np
import jax 
import math
from PIL import Image
import wandb
from ml_collections import ConfigDict


def cosine_beta_schedule(timesteps):
    """Return cosine schedule 
    as proposed in https://arxiv.org/abs/2102.09672 """
    s=0.008
    max_beta=0.999
    ts = jnp.linspace(0, 1, timesteps + 1)
    alphas_bar = jnp.cos((ts + s) / (1 + s) * jnp.pi /2) ** 2
    alphas_bar = alphas_bar/alphas_bar[0]
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    return(jnp.clip(betas, 0, max_beta))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    betas = jnp.linspace(
        beta_start, beta_end, timesteps, dtype=jnp.float64)
    return(betas)

def get_ddpm_params(config):
    schedule_name = config.beta_schedule
    timesteps = config.timesteps
    p2_loss_weight_gamma = config.p2_loss_weight_gamma
    p2_loss_weight_k = config.p2_loss_weight_gamma

    if schedule_name == 'linear':
        betas = linear_beta_schedule(timesteps)
    elif schedule_name == 'cosine':
        betas = cosine_beta_schedule(timesteps)
    else:
        raise ValueError(f'unknown beta schedule {schedule_name}')
    assert betas.shape == (timesteps,)
    alphas = 1. - betas
    alphas_bar = jnp.cumprod(alphas, axis=0)
    sqrt_alphas_bar = jnp.sqrt(alphas_bar)
    sqrt_1m_alphas_bar = jnp.sqrt(1. - alphas_bar)
    
    # calculate p2 reweighting
    p2_loss_weight = (p2_loss_weight_k + alphas_bar / (1 - alphas_bar)) ** - p2_loss_weight_gamma

    return {
      'betas': betas,
      'alphas': alphas,
      'alphas_bar': alphas_bar,
      'sqrt_alphas_bar': sqrt_alphas_bar,
      'sqrt_1m_alphas_bar': sqrt_1m_alphas_bar,
      'p2_loss_weight': p2_loss_weight
  }



def make_grid(samples, n_samples, padding=2, pad_value=0.0):

  ndarray = samples.reshape((-1, *samples.shape[2:]))[:n_samples]
  nrow = int(np.sqrt(ndarray.shape[0]))

  if not (isinstance(ndarray, jnp.ndarray) or
          (isinstance(ndarray, list) and
           all(isinstance(t, jnp.ndarray) for t in ndarray))):
    raise TypeError("array_like of tensors expected, got {}".format(
        type(ndarray)))

  ndarray = jnp.asarray(ndarray)

  if ndarray.ndim == 4 and ndarray.shape[-1] == 1:  # single-channel images
    ndarray = jnp.concatenate((ndarray, ndarray, ndarray), -1)

  # make the mini-batch of images into a grid
  nmaps = ndarray.shape[0]
  xmaps = min(nrow, nmaps)
  ymaps = int(math.ceil(float(nmaps) / xmaps))
  height, width = int(ndarray.shape[1] + padding), int(ndarray.shape[2] +
                                                       padding)
  num_channels = ndarray.shape[3]
  grid = jnp.full(
      (height * ymaps + padding, width * xmaps + padding, num_channels),
      pad_value).astype(jnp.float32)
  k = 0
  for y in range(ymaps):
    for x in range(xmaps):
      if k >= nmaps:
        break
      grid = grid.at[y * height + padding:(y + 1) * height,
                     x * width + padding:(x + 1) * width].set(ndarray[k])
      k = k + 1
  return grid


def save_image(samples, n_samples, fp, padding=2, pad_value=0.0, format=None):
  """Make a grid of images and Save it into an image file.

  Args:
    ndarray (array_like): 4D mini-batch images of shape (B x H x W x C).
    fp: A filename(string) or file object.
    nrow (int, optional): Number of images displayed in each row of the grid.
      The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
    padding (int, optional): amount of padding. Default: ``2``.
    pad_value (float, optional): Value for the padded pixels. Default: ``0``.
    format(Optional):  If omitted, the format to use is determined from the
      filename extension. If a file object was used instead of a filename, this
      parameter should always be used.
  """

  grid = make_grid(samples, n_samples, padding, pad_value)
  # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
  ndarr = jnp.clip(grid * 255.0 + 0.5, 0, 255).astype(jnp.uint8)
  ndarr = np.array(ndarr)
  im = Image.fromarray(ndarr)
  im.save(fp, format=format)

  return ndarr

def wandb_log_image(samples_array, step):
    sample_images = wandb.Image(samples_array, caption = f"step {step}")
    wandb.log({'samples':sample_images })

def wandb_log_model(workdir, step):
    artifact = wandb.Artifact(name=f"model-{wandb.run.id}", type="ddpm_model")
    artifact.add_file( f"{workdir}/checkpoint_{step}")
    wandb.run.log_artifact(artifact)


def to_wandb_config(d: ConfigDict, parent_key: str = '', sep: str ='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, ConfigDict):
            items.extend(to_wandb_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)



  

