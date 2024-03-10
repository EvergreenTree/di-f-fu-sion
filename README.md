# di-f-fu-sion
Self-contained and (mostly) minimal training and model fusion of latent diffusion models, with flax!

- codebase credit to https://github.com/huggingface/diffusers and others
- Research supported with Cloud TPUs from Google's TPU Research Cloud (TRC)

## Example training command
```
export MODEL_NAME="$HOME/.cache/huggingface/hub/models--duongna--stable-diffusion-v1-4-flax/snapshots/6f9644eae775b7b50d0031a74b7d8d974f398e26/"
export DATASET_NAME="Bingsu/Cat_and_Dog"

python3 train_unconditional_local.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --resolution=256 --center_crop --random_flip --seed 23\
  --train_batch_size=1 \
  --mixed_precision="no" \
  --max_train_steps=15000 \
  --learning_rate=1e-05 --scale_lr\
  --max_grad_norm=.05 \
  --output_dir="sd-cats-unconditional" \
  --unet_config_path="unet-config-unconditional.json" \
  --image_column="Images"

```