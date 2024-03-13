# di-f-fu-sion
Self-contained and (mostly) minimal training and model fusion of latent diffusion models, with flax!

- codebase credit to https://github.com/huggingface/diffusers and others
- Research supported with Cloud TPUs from Google's TPU Research Cloud (TRC)

## Example training command
```
export MODEL_NAME="$HOME/.cache/huggingface/hub/models--duongna--stable-diffusion-v1-4-flax/snapshots/6f9644eae775b7b50d0031a74b7d8d974f398e26/"
export DATASET_NAME="Norod78/microsoft-fluentui-emoji-512-whitebg"
export DATASET_NAME="imagenet-1k"

python3 train_unconditional_local.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --resolution=512 --center_crop --random_flip --seed 37 \
  --train_batch_size=1 \
  --mixed_precision="bf16" \
  --max_train_steps=50000 \
  --learning_rate=1e-6 --scale_lr \
  --output_dir="sd-imagenet" 
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