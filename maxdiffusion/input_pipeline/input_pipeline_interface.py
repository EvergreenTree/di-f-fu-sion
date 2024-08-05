"""
 Copyright 2024 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

import os
import math
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
from datasets import load_dataset, load_from_disk
import jax
import jax.numpy as jnp
import numpy as np

from maxdiffusion import multihost_dataloading

AUTOTUNE = tf.data.experimental.AUTOTUNE

def vae_apply(images, sample_rng, vae, vae_params):
  vae_outputs = vae.apply(
    {"params" : vae_params}, images,
      deterministic=True, method=vae.encode
  )
  latents = vae_outputs.latent_dist.sample(sample_rng)
  latents = jnp.transpose(latents, (0, 3, 1, 2))
  latents = latents * vae.config.scaling_factor

  return latents

def encode(input_ids, text_encoder, text_encoder_params):
  return text_encoder(
    input_ids,
    params=text_encoder_params,
    train=False
  )[0]

# TODO - https://github.com/google/array_record/blob/main/beam/examples/example_gcs_conversion.py
def make_laion400m_train_iterator(
    config,
    mesh,
    global_batch_size,
):
  """Iterator for Laion dataset.
  To see how to prepare this dataset, look at
  maxdiffusion/pedagogical_examples/to_tfrecords.py
  """
  feature_description = {
    "latents" : tf.io.FixedLenFeature([], tf.string),
    "hidden_states" : tf.io.FixedLenFeature([], tf.string)
  }

  def _parse_tfrecord_fn(example):
    return tf.io.parse_single_example(example, feature_description)

  def prepare_sample(features):
    latents = tf.io.parse_tensor(tnp.asarray(features["latents"]), out_type=tf.float32)
    hidden_states = tf.io.parse_tensor(tnp.asarray(features["hidden_states"]), out_type=tf.float32)
    return {"pixel_values" : latents, "input_ids" : hidden_states}

  filenames = tf.io.gfile.glob(os.path.join(config.train_data_dir,"*"))
  train_ds = (
    tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
      .map(_parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)
      .map(prepare_sample, num_parallel_calls=AUTOTUNE)
      .shuffle(global_batch_size * 10) 
      .batch(global_batch_size // jax.process_count(), drop_remainder=True) # per-host batch size. must divide process_count()
      .prefetch(AUTOTUNE)
      .repeat(100000000)
  )

  train_ds = train_ds.shard(num_shards = jax.process_count(), index = jax.process_index()) # shard() will take different batches of the size in batch() on each host

  train_iter = multihost_dataloading.get_batch_sharded_data_pipeline(train_ds, mesh)
  return train_iter

def make_pokemon_train_iterator(
    config,
    mesh,
    global_batch_size,
    tokenize_fn,
    image_transforms_fn):

  captions_column = config.caption_column
  image_column = config.image_column
  cache_latents_text_encoder_outputs = config.cache_latents_text_encoder_outputs
  dataset_save_location = config.dataset_save_location
  if os.path.isdir(dataset_save_location):
    train_ds = load_from_disk(dataset_save_location)
  else:
    train_ds = load_dataset(config.dataset_name,split="train")
    train_ds = train_ds.map(
      function=tokenize_fn,
      batched=True,
      remove_columns=[captions_column],
      num_proc=1 if cache_latents_text_encoder_outputs else 4,
      desc="Running tokenizer on train dataset",
    )
    # need to do it before load_as_tf_dataset
    # since raw images are different sizes
    # will break from_tensor_slices
    train_ds = train_ds.map(
      function=image_transforms_fn,
      batched=True,
      remove_columns=[image_column],
      num_proc=1 if cache_latents_text_encoder_outputs else config.transform_images_num_proc,
      desc="Transforming images",
    )
    train_ds.save_to_disk(dataset_save_location)
    train_ds.cleanup_cache_files()

  # taken from https://github.com/huggingface/transformers/blob/abbffc4525566a48a9733639797c812301218b83/examples/tensorflow/contrastive-image-text/run_clip.py#L225
  def load_as_tf_dataset(dataset, batch_size, shuffle):
    dataset = dataset.with_format("tensorflow")[:]
    tf_dataset = tf.data.Dataset.from_tensor_slices(dataset)

    if shuffle:
      tf_dataset = tf_dataset.shuffle(len(tf_dataset))
    tf_dataset = tf_dataset.batch(batch_size // jax.process_count(), drop_remainder=True)
    tf_dataset = tf_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    repeats = math.ceil((config.max_train_steps * batch_size) / len(tf_dataset))
    tf_dataset = tf_dataset.repeat(repeats)

    return tf_dataset

  train_ds = load_as_tf_dataset(
    train_ds, global_batch_size, True
  )
  train_ds = train_ds.shard(num_shards = jax.process_count(), index = jax.process_index())

  train_iter = multihost_dataloading.get_batch_sharded_data_pipeline(train_ds, mesh)
  return train_iter

def make_tf_train_iterator(
    config,
    mesh,
    per_device_batch_size):
  
  if config.dataset_name == 'cifar10':
    data = CIFAR10(class_conditional = False, randflip = False)
  elif config.dataset_name == 'imagenet':
    data = ImageNet(class_conditional = False, randflip = False)
  elif config.dataset_name == 'lsun':
    data = LSUN(class_conditional = False, randflip = False)
  def normalize(examples):
    img = tf.cast(examples['image'], tf.float32) / 255.0 * 2 - 1
    return {'pixel_values': img,'input_ids': tnp.array(0.)}
  
  train_ds = (data._load_tfds(split='train', shuffle_seed=None)
        .map(normalize,num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(16 * per_device_batch_size, seed=0)
        .batch(per_device_batch_size,drop_remainder=True)
        .shard(num_shards=jax.process_count(),index=jax.process_index())
  )
  train_iter = multihost_dataloading.get_batch_sharded_data_pipeline(train_ds, mesh)
  return train_iter

# The following code is from diffusion_distillation. Thanks!

"""TF Datasets.

The general design philosophy of these dataset loaders is to keep them as simple
as possible. Data processing or manipulation of conditioning information should
be kept in an experiment's main.py, not here.

When data augmentation is enabled, nondeterministic behavior is expected.
"""

# pylint: disable=logging-format-interpolation
# pylint: disable=g-long-lambda

import functools
from absl import logging
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


def batch_dataset(dataset, batch_shape):
  for b in reversed(batch_shape):
    dataset = dataset.batch(b, drop_remainder=True)
  return dataset


class Dataset:
  """Generic dataset."""

  @property
  def info(self):
    raise NotImplementedError

  @property
  def data_shape(self):
    return self.info['data_shape']

  @property
  def num_train(self):
    return self.info['num_train']

  @property
  def num_eval(self):
    return self.info['num_eval']

  @property
  def num_classes(self):
    return self.info['num_classes']

  def _load_tfds(self, *, split, shuffle_seed):
    raise NotImplementedError

  def _preprocess(self, x, *, split, augment):
    """Preprocess one example."""
    raise NotImplementedError

  def _shuffle_buffer_size(self, split):
    del split
    return 50000
  

class CIFAR10(Dataset):
  """CIFAR10 dataset."""

  def __init__(self, *, class_conditional = False, randflip = False):
    self._class_conditional = class_conditional
    self._randflip = randflip
    self._info = {
        'data_shape': (32, 32, 3),
        'num_train': 50000,
        'num_eval': 10000,
        'num_classes': 10 if self._class_conditional else 1
    }

  @property
  def info(self):
    return self._info

  def _load_tfds(self, *, split, shuffle_seed):
    return tfds.load(
        'cifar10',
        split={'train': 'train', 'eval': 'test'}[split],
        shuffle_files=shuffle_seed is not None,
        read_config=None if shuffle_seed is None else tfds.ReadConfig(
            shuffle_seed=shuffle_seed))

  def _preprocess(self, x, *, split, augment):
    del split
    img = tf.cast(x['image'], tf.float32)
    if augment:  # NOTE: this makes training nondeterministic
      if self._randflip:
        aug_img = tf.image.flip_left_right(img)
        aug = tf.random.uniform(shape=[]) > 0.5
        img = tf.where(aug, aug_img, img)
    out = {'image': img}
    if self._class_conditional:
      out['label'] = tf.cast(x['label'], tf.int32)
    return out


def central_square_crop(img):
  """Crop to square along the long edge."""
  h, w, _ = tf.unstack(tf.shape(img))
  box = tf.where(h > w, [h // 2 - w // 2, 0, w, w], [0, w // 2 - h // 2, h, h])
  offset_height, offset_width, target_height, target_width = tf.unstack(box)
  return tf.image.crop_to_bounding_box(
      img, offset_height, offset_width, target_height, target_width)


def decode_and_central_square_crop(img):
  """Crop to square along the long edge."""
  h, w, _ = tf.unstack(tf.io.extract_jpeg_shape(img))
  box = tf.where(h > w, [h // 2 - w // 2, 0, w, w], [0, w // 2 - h // 2, h, h])
  return tf.image.decode_and_crop_jpeg(img, box, channels=3)


class ImageNet(Dataset):
  """ImageNet dataset."""

  def __init__(self,
               *,
               class_conditional,
               image_size,
               randflip,
               extra_image_sizes=()):
    """ImageNet dataset.

    Args:
      class_conditional: bool: class conditional generation problem; if True,
        generated examples will contain a label.
      image_size: int: size of image to model
      randflip: bool: random flip augmentation
      extra_image_sizes: Tuple[int]: also provide image at these resolutions
    """
    self._class_conditional = class_conditional
    self._image_size = image_size
    self._randflip = randflip
    self._extra_image_sizes = extra_image_sizes
    self._info = {
        'data_shape': (self._image_size, self._image_size, 3),
        'num_train': 1281167,
        'num_eval': 50000,
        'num_classes': 1000 if self._class_conditional else 1
    }

  @property
  def info(self):
    return self._info

  def _load_tfds(self, *, split, shuffle_seed):
    return tfds.load(
        'imagenet2012',
        split={'train': 'train', 'eval': 'validation'}[split],
        shuffle_files=shuffle_seed is not None,
        read_config=None if shuffle_seed is None else tfds.ReadConfig(
            shuffle_seed=shuffle_seed),
        decoders={'image': tfds.decode.SkipDecoding()})

  def _preprocess(self, x, *, split, augment):
    del split  # unused
    out = {}

    # Decode the image and resize
    img = tf.cast(decode_and_central_square_crop(x['image']), tf.float32)

    if augment:
      # NOTE: this makes training nondeterministic
      if self._randflip:
        logging.info('ImageNet: randflip=True')
        img = tf.image.random_flip_left_right(img)

    # Standard area resizing
    out['image'] = tf.clip_by_value(
        tf.image.resize(img, [self._image_size, self._image_size], 'area'),
        0, 255)

    # Optionally provide the image at other resolutions too
    for s in self._extra_image_sizes:
      assert isinstance(s, int)
      out[f'extra_image_{s}'] = tf.clip_by_value(
          tf.image.resize(img, [s, s], 'area'), 0, 255)

    # Class label
    if self._class_conditional:
      out['label'] = tf.cast(x['label'], tf.int32)

    return out


class LSUN(Dataset):
  """LSUN dataset."""

  def __init__(self, *, subset, image_size, randflip,
               extra_image_sizes=()):
    """LSUN datasets.

    Args:
      subset: str: 'church' or 'bedroom'
      image_size: int: size of image to model, 64 or 128
      randflip: bool: random flip augmentation
      extra_image_sizes: optional extra image sizes
    """
    self._subset = subset
    self._image_size = image_size
    self._randflip = randflip
    self._extra_image_sizes = extra_image_sizes

    self._info = {
        'data_shape': (self._image_size, self._image_size, 3),
        'num_train': {'bedroom': 3033042, 'church': 126227}[self._subset],
        'num_eval': 300,
        'num_classes': 1,
    }

  @property
  def info(self):
    return self._info

  def _load_tfds(self, *, split, shuffle_seed):
    tfds_name = {'church': 'lsun/church_outdoor',
                 'bedroom': 'lsun/bedroom'}[self._subset]
    return tfds.load(
        tfds_name,
        split={'train': 'train', 'eval': 'validation'}[split],
        shuffle_files=shuffle_seed is not None,
        read_config=None if shuffle_seed is None else tfds.ReadConfig(
            shuffle_seed=shuffle_seed),
        decoders={'image': tfds.decode.SkipDecoding()})

  def _preprocess(self, x, *, split, augment):
    del split  # unused

    # Decode the image and resize
    img = tf.cast(decode_and_central_square_crop(x['image']), tf.float32)
    if augment:  # NOTE: nondeterministic
      if self._randflip:
        aug_img = tf.image.flip_left_right(img)
        aug = tf.random.uniform(shape=[]) > 0.5
        img = tf.where(aug, aug_img, img)

    out = {}
    out['image'] = tf.clip_by_value(tf.image.resize(
        img, [self._image_size, self._image_size], antialias=True), 0, 255)

    # Optionally provide the image at other resolutions too
    for s in self._extra_image_sizes:
      assert isinstance(s, int)
      out[f'extra_image_{s}'] = tf.clip_by_value(
          tf.image.resize(img, [s, s], antialias=True), 0, 255)

    return out
