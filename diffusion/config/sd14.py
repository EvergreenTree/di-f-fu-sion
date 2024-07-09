# coding=utf-8
# Copyright 2024 The TRC Authors.
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

import ml_collections

def D(**kwargs):
  return ml_collections.ConfigDict(initial_dictionary=kwargs)

def get_config():
    config = ml_collections.ConfigDict()

    config.pretrained_model_name_or_path = "duongna/stable-diffusion-v1-4-flax"
    config.dataset_name = "diffusers/pokemon-gpt4-captions"
    config.revision = None
    config.variant = None
    config.dataset_config_name = None
    config.train_data_dir = None
    config.image_column = None
    config.caption_column = None
    config.max_train_samples = None
    config.output_dir = 'sd-model-finetuned'
    config.cache_dir = None
    config.seed = 102
    config.resolution = 512
    config.center_crop = False
    config.random_flip = False
    config.train_batch_size = 1
    config.num_train_epochs = 100
    config.max_train_steps = None
    config.learning_rate = 1e-6
    config.scale_lr = False
    config.lr_scheduler = 'constant'
    config.adam_beta1 = 0.9
    config.adam_beta2 = 0.999
    config.adam_weight_decay = 1e-2
    config.adam_epsilon = 1e-8
    config.max_grad_norm = 1e4
    config.push_to_hub = False
    config.hub_token = None
    config.hub_model_id = None
    config.logging_dir = 'sd-logs'
    config.unet_config_path = None
    config.mixed_precision = 'no'
    config.from_pt = False
    config.from_scratch = False
    config.ema = True
    config.prompt = None
    config.plot_freq = 1000
    config.fid_freq = 10000
    config.precomputed_fid_stats = True
    config.unconditional = False
    config.toy = False

    return config