# di-f-fu-sion
Self-contained and (mostly) minimal training and model fusion of latent diffusion models, with flax!

- codebase credit to https://github.com/huggingface/diffusers and others
- Research supported with Cloud TPUs from Google's TPU Research Cloud (TRC)

## Example training command
Selected datasets: `conorcl/portraits-512` `teticio/audio-diffusion-512` `Norod78/microsoft-fluentui-emoji-512-whitebg` `poorguys/TW-Kai_2_Chong_Xi_Small_Seal_all_512`
Example pre-trained model path: "$HOME/.cache/huggingface/hub/models--duongna--stable-diffusion-v1-4-flax/snapshots/6f9644eae775b7b50d0031a74b7d8d974f398e26/" 
```
python3 train_unconditional_local.py \
  --pretrained_model_name_or_path="sd-portraits" \
  --dataset_name="conorcl/portraits-512" \
  --resolution=512 --center_crop --random_flip --seed 35 \
  --train_batch_size=1 \
  --max_train_steps=50000 \
  --learning_rate=2e-7 --scale_lr \
  --output_dir="sd-portraits" \
  --unet_config_path="configs/unet-config-medium-conic.json"
```

## Quickly train a toy
```
python3 train_toy_local.py \
  --dataset_name="mnist" \
  --resolution=32 --seed 23\
  --max_train_steps=15000 \
  --learning_rate=1e-6 --scale_lr \
  --output_dir="sd-mnist" \
  --unet_config_path="configs/unet-config-mnist.json"
```