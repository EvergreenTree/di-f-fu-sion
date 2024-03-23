import argparse
import logging
import math
import os
from PIL import Image
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
import torch.utils.checkpoint
import transformers
from datasets import load_dataset
from flax import jax_utils
from flax.training import train_state
from flax.training.common_utils import shard
from huggingface_hub import create_repo, upload_folder
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPTokenizer, FlaxCLIPTextModel, set_seed

from src.pipeline_flax_stable_diffusion import (
    FlaxAutoencoderKL,
    FlaxPNDMScheduler,
    FlaxStableDiffusionPipeline,
    FlaxUnconditionalStableDiffusionPipeline,
    FlaxUNet2DConditionModel,
)
from src.schedulers.scheduling_ddpm_flax import FlaxDDPMScheduler

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that  Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup", "warmup_exponential_decay"]'
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--from_pt",
        action="store_true",
        default=False,
        help="Flag to indicate whether to convert models from PyTorch.",
    )
    parser.add_argument(
        "--unet_config_path",
        default=None,
        help="If specified, re-initialize the unet with the given config (Optional)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Do not train, save.",
    )
    parser.add_argument(
        "--from_scratch",
        action="store_true",
        help="Clear model memory.",
    )
    parser.add_argument(
        "--ema",
        action="store_true",
        help="EMA weight decay.",
    )


    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    return args


dataset_name_mapping = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}


def get_params_to_save(params):
    return jax.device_get(jax.tree_util.tree_map(lambda x: x[0], params))


def main():
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)
    if jax.process_index() == 0:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
    if args.seed is not None:
        set_seed(args.seed)
    if jax.process_index() == 0:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id
    if args.dataset_name is not None:
        dataset = load_dataset(
            args.dataset_name, args.dataset_config_name, cache_dir=args.cache_dir, data_dir=args.train_data_dir
        )
    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files["train"] = os.path.join(args.train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=args.cache_dir,
        )
    column_names = dataset["train"].column_names
    dataset_columns = dataset_name_mapping.get(args.dataset_name, None)
    if args.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )   
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        return examples
    if args.max_train_samples is not None:
        dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
    train_dataset = dataset["train"].with_transform(preprocess_train)
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        batch = {
            "pixel_values": pixel_values,
        }
        batch = {k: v.numpy() for k, v in batch.items()}
        return batch
    total_train_batch_size = args.train_batch_size * jax.local_device_count()
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=total_train_batch_size, drop_last=True
    )

    weight_dtype = jnp.float32
    if args.mixed_precision == "fp16":
        weight_dtype = jnp.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = jnp.bfloat16
    # tokenizer = text_encoder = None
    vae, vae_params = FlaxAutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        from_pt=args.from_pt,
        revision=args.revision,
        subfolder="vae",
        dtype=weight_dtype,
    )
    rng = jax.random.PRNGKey(args.seed)

    # load parameters
    if args.unet_config_path is None or not args.from_scratch:
        unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path,
            from_pt=args.from_pt,
            revision=args.revision,
            subfolder="unet",
            dtype=weight_dtype,
        )

    # reload unet
    if args.unet_config_path:
        config = FlaxUNet2DConditionModel.load_config(args.unet_config_path)
        unet = FlaxUNet2DConditionModel.from_config(
            config,
            revision=args.revision,
            dtype=weight_dtype,
        )

    # reload parameters
    if args.from_scratch:
        # Reinitialize weights
        rng, key = jax.random.split(rng)
        unet_params = unet.init_weights(key)
    else:
        # Roughly adapt to pre-trained weights. In the cross-attention case with shape [768, C], prune it to shape [C,C].
        # in this case, unet_config_path must be the same as pre-trained model.
        def prune(x):
            if len(x.shape) > 1:
                if x.shape[-2] == 768:
                    return jnp.eye(x.shape[-1]) # or return x[...,:x.shape[-1],:], which does not work for x.shape[-1] > C
            return x
        if unet.cross_attention_dim > 0:
            print("Pruning cross-attention matrices to force the conditional model to be unconditional")
            unet.config.cross_attention_dim = 0 
            unet.cross_attention_dim = 0 # save config to be compatible for loading later
            unet_params = jax.tree_map(prune, unet_params)
        

    # Optimization
    if args.scale_lr:
        args.learning_rate = args.learning_rate * total_train_batch_size
    if args.lr_scheduler == "constant":
        warmup_steps = 1000
        warmup_schedule = optax.linear_schedule(
            init_value=0.0,
            end_value=args.learning_rate, 
            transition_steps=warmup_steps, 
        )
        constant_schedule = optax.constant_schedule(args.learning_rate)
        schedule = optax.join_schedules(
            schedules=[warmup_schedule, constant_schedule],
            boundaries=[warmup_steps]
        )
    else:
        raise NotImplementedError
    
    adamw = optax.adamw(
        learning_rate=schedule,
        b1=args.adam_beta1,
        b2=args.adam_beta2,
        eps=args.adam_epsilon,
        weight_decay=args.adam_weight_decay,
    )
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        adamw,
    )
    state = train_state.TrainState.create(apply_fn=unet.__call__, params=unet_params, tx=optimizer)
    noise_scheduler = FlaxPNDMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
    )
    noise_scheduler_state = noise_scheduler.create_state()
    
    # Initialization
    train_rngs = jax.random.split(rng, jax.local_device_count())

    def validate(checkpoint=False):
        params = {
            "vae": vae_params,
            "unet": ema_params if args.ema else state.params,
            "scheduler": noise_scheduler_state_p
        }
        sampling_seed = jax.random.PRNGKey(0)
        num_inference_steps = 50
        num_devices = jax.local_device_count()
        sampling_seed = jax.random.split(sampling_seed, num_devices)
        batch_size = 1
        images = pipeline(batch_size, params, sampling_seed, num_inference_steps, jit=True).images
        if jax.process_index() == 0:
            if args.output_dir is not None:
                os.makedirs(args.output_dir+'/samples', exist_ok=True)
            images = np.asarray(images.reshape((batch_size * num_devices,) + images.shape[-3:]))
            images = pipeline.numpy_to_pil(images)
            for index, image in enumerate(images):
                image.save(args.output_dir+'/samples/epoch'+str(epoch)+'_'+str(index)+".png")
            if checkpoint:
                pipeline.save_pretrained(
                    args.output_dir,
                    params={
                        "vae": get_params_to_save(vae_params),
                        "unet": get_params_to_save(state.params),
                    },
                )

    def train_step(state, text_encoder_params, vae_params, batch, train_rng):
        dropout_rng, sample_rng, new_train_rng = jax.random.split(train_rng, 3)

        def compute_loss(params):
            # Convert images to latent space
            vae_outputs = vae.apply(
                {"params": vae_params}, batch["pixel_values"], deterministic=True, method=vae.encode
            )
            latents = vae_outputs.latent_dist.sample(sample_rng)
            # (NHWC) -> (NCHW)
            latents = jnp.transpose(latents, (0, 3, 1, 2))
            latents = latents * vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise_rng, timestep_rng = jax.random.split(sample_rng)
            noise = jax.random.normal(noise_rng, latents.shape)
            # Sample a random timestep for each image
            bsz = latents.shape[0]
            timesteps = jax.random.randint(
                timestep_rng,
                (bsz,),
                0,
                noise_scheduler.config.num_train_timesteps,
            )

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(noise_scheduler_state, latents, noise, timesteps)

            encoder_hidden_states = None

            # Predict the noise residual and compute loss
            model_pred = unet.apply(
                {"params": params}, noisy_latents, timesteps, encoder_hidden_states, train=True
            ).sample

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(noise_scheduler_state, latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            loss = (target - model_pred) ** 2
            loss = loss.mean()

            return loss

        grad_fn = jax.value_and_grad(compute_loss)
        loss, grad = grad_fn(state.params)
        grad = jax.lax.pmean(grad, "batch")

        new_state = state.apply_gradients(grads=grad)

        metrics = {"loss": loss}
        metrics = jax.lax.pmean(metrics, axis_name="batch")

        return new_state, metrics, new_train_rng

    # Create parallel version of the train step
    p_train_step = jax.pmap(train_step, "batch", donate_argnums=(0,))

    # EMA weights
    if args.ema:
        ema_decay = 0.99
        ema_params = jax.tree_map(lambda x: x+0, unet_params)

    # Replicate the train state on each device
    noise_scheduler_state_p = jax_utils.replicate(noise_scheduler_state)
    state = jax_utils.replicate(state)
    if args.ema:
        ema_params = jax_utils.replicate(ema_params)
    text_encoder_params = None
    vae_params = jax_utils.replicate(vae_params)


    # Train!
    num_update_steps_per_epoch = math.ceil(len(train_dataloader))

    # Scheduler and math around the number of training steps.
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel & distributed) = {total_train_batch_size}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    pipeline = FlaxUnconditionalStableDiffusionPipeline(
        vae=vae,
        unet=unet,
        scheduler=noise_scheduler,
    )
                
    epochs = tqdm(range(args.num_train_epochs), desc="Epoch ... ", position=0)
    for epoch in epochs:
        train_metrics = []
        steps_per_epoch = len(train_dataset) // total_train_batch_size
        train_step_progress_bar = tqdm(total=steps_per_epoch, desc="Training...", position=1, leave=False)
        for batch in train_dataloader:
            batch = shard(batch)
            state, train_metric, train_rngs = p_train_step(state, text_encoder_params, vae_params, batch, train_rngs)
            train_metrics.append(train_metric)
            train_step_progress_bar.update(1)
            if global_step >= args.max_train_steps:
                break
            if global_step % 1000 == 0:
                validate()
            global_step += 1
            if args.ema and global_step % 1000 == 0:
                ema_params = jax.tree_util.tree_map(lambda ema, new: ema * ema_decay + (1 - ema_decay) * new, ema_params,state.params)

        train_metric = jax_utils.unreplicate(train_metric)
        train_step_progress_bar.close()
        epochs.write(f"Epoch... ({epoch + 1}/{args.num_train_epochs} | Loss: {train_metric['loss']})")
        # validate()

    logger.info("***** Saving model *****")
    if jax.process_index() == 0:
        pipeline.save_pretrained(
            args.output_dir,
            params={
                "vae": get_params_to_save(vae_params),
                "unet": get_params_to_save(ema_params = jax.tree_util.tree_map(lambda ema, new: ema * ema_decay + (1 - ema_decay) * new, ema_params,state.params)
.params),
                # "scheduler": get_params_to_save(scheduler.params),
            },
        )
        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    

if __name__ == "__main__":
    main()
