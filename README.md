# Di-f-fu-sion
Mechanics-Inspired diffusion model structures with huggingface🤗 support!

This project is supported by the Google TRC program. Many thanks!

Based on: [diffusers](https://github.com/huggingface/diffusers/tree/main/examples/text_to_image), [maxdiffusion](https://github.com/google/maxdiffusion), [ddpm-flax](https://github.com/yiyixuxu/denoising-diffusion-flax), [LDM](https://github.com/CompVis/latent-diffusion), [diffusion_distillation](https://github.com/google-research/google-research/tree/master/diffusion_distillation). Thanks for open-sourcing!

## Training
```
python maxdiffusion/main.py --config=maxdiffusion/configs/imagenet.py
```

# Demos:
[diffusion_lens.ipynb](diffusion_lens.ipynb)
[parameter_count.ipynb](parameter_count.ipynb)

# Contributions:
- Orthogonal symmetry in general-purpose model design for acceleration
- Translation symmetry in general-purpose model design for compression