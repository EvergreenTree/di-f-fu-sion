# di-f-fu-sion
Self-contained and (mostly) minimal training and model fusion of latent diffusion models, with flax!

- codebase credit to https://github.com/huggingface/diffusers and others
- Research supported with Cloud TPUs from Google's TPU Research Cloud (TRC)

## Example training command
```
export MODEL_NAME="$HOME/.cache/huggingface/hub/models--duongna--stable-diffusion-v1-4-flax/snapshots/6f9644eae775b7b50d0031a74b7d8d974f398e26/"
export DATASET_NAME="tglcourse/lsun_church_train"

python3 train_unconditional_local.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --resolution=256 --center_crop --random_flip --seed 23\
  --max_train_steps=3000 \
  --learning_rate=1e-8 --scale_lr \
  --output_dir="sd-church-unconditional" 
 # --max_grad_norm=.05 
  #--learning_rate=1e-4, 1e-6
 # --unet_config_path="unet-config-unconditional.json"

```

## Quickly train a toy
```
export DATASET_NAME="mnist"

python3 train_toy_local.py \
  --dataset_name=$DATASET_NAME \
  --resolution=32 --seed 23\
  --max_train_steps=7500 \
  --learning_rate=1e-4 --scale_lr \
  --output_dir="sd-mnist" \
   --unet_config_path="unet-config-small.json"
```