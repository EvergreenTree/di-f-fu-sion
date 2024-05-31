import argparse
import logging
import math
import os
import random
from pathlib import Path
import functools
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from torchvision import transforms
from datasets import load_dataset
from flax import jax_utils
from flax.training import train_state
from flax.training.common_utils import shard
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
import transformers
from transformers import CLIPTokenizer, FlaxCLIPTextModel, set_seed #, CLIPImageProcessor
from tensorboardX import SummaryWriter

from src.models.unet_2d_condition_flax import FlaxUNet2DConditionModel
from src.pipeline_flax_stable_diffusion import (
    FlaxAutoencoderKL,
    FlaxPNDMScheduler,
    FlaxStableDiffusionPipeline,
    FlaxUNet2DConditionModel,
    FlaxToyDiffusionPipeline,
    FlaxUnconditionalStableDiffusionPipeline
)
from src.fid import inception

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=False,
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
            " or to a folder containing files that HuggingFace Datasets can understand."
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
        "--image_column", type=str, default=None, help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default=None,
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
    parser.add_argument("--seed", type=int, default=103, help="A seed for reproducible training.")
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
        default=1e-6,
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
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1e4, type=float, help="Max gradient norm.")
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
        default="sd-logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--unet_config_path",
        default=None,
        help="If specified, re-initialize the unet with the given config (Optional)",
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
        "--from_scratch",
        action="store_true",
        help="Train from scratch.",
    )
    parser.add_argument(
        "--ema",
        type=bool,
        default=True,
        help="Exponential moving average (long-term momentum conservation) of model parameters (True by default).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Validation prompt.",
    )
    parser.add_argument(
        "--plot_freq",
        type=int,
        default=1000,
        help="Log plots at this frequency (1000 by default).",
    )
    parser.add_argument(
        "--fid_freq",
        type=int,
        default=10000,
        help="Log FID frequency.",
    )
    parser.add_argument(
        "--precomputed_fid_stats",
        type=bool,
        default=True,
        help="Precomputed FID stats.",
    )
    parser.add_argument(
        "--unconditional",
        action="store_true",
        default=False,
        help="Unconditional Generation.",
    )
    parser.add_argument(
        "--toy",
        action="store_true",
        default=False,
        help="Small dataset like CIFAR10.",
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
    "poloclub/diffusiondb": ("image", "prompt"),
    "cifar10": ("img", ),
}


def get_params_to_save(params):
    return jax.device_get(jax.tree_util.tree_map(lambda x: x[0], params))



def main():
    args = parse_args()

    writer = SummaryWriter(args.logging_dir)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Setup logging, we only want one process per machine to log things on the screen.
    logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)
    if jax.process_index() == 0:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if jax.process_index() == 0:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name == "imagenet":
        import requests
        headers = {"Authorization": f"Bearer {args.hub_token}"}
        API_URL = "https://datasets-server.huggingface.co/parquet?dataset=imagenet-1k"
        data = requests.get(API_URL, headers=headers).json()
        data_files = [k["url"] for k in data["parquet_files"]]
        dataset = load_dataset("parquet", data_files=data_files, split="train", token=args.hub_token)    
    elif args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
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
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    dataset_columns = dataset_name_mapping.get(args.dataset_name, None)
    if args.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )

    if not args.toy and not args.unconditional:
        if args.caption_column is None:
            caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
        else:
            caption_column = args.caption_column
            if caption_column not in column_names:
                raise ValueError(
                    f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
                )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(captions, max_length=tokenizer.model_max_length, padding="do_not_pad", truncation=True)
        input_ids = inputs.input_ids
        return input_ids
    
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
        if not args.toy:
            examples["input_ids"] = tokenize_captions(examples)

        return examples

    if args.max_train_samples is not None:
        dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
    train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        if args.toy:
            batch = {
                "pixel_values": pixel_values,
            }
        else:
            input_ids = [example["input_ids"] for example in examples]
            padded_tokens = tokenizer.pad(
                {"input_ids": input_ids}, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt"
            )
            batch = {
                "pixel_values": pixel_values,
                "input_ids": padded_tokens.input_ids,
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

    # Initialize model parameters
    rng = jax.random.PRNGKey(args.seed)

    if args.pretrained_model_name_or_path:
        unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path,
            from_pt=args.from_pt,
            revision=args.revision,
            subfolder="unet",
            dtype=weight_dtype,
        )
    # Redefine unet
    if args.unet_config_path:
        config = FlaxUNet2DConditionModel.load_config(args.unet_config_path)
        unet = FlaxUNet2DConditionModel.from_config(
            config,
            revision=args.revision,
            dtype=weight_dtype,
        )
    # Reload unet parameters
    if args.toy or args.from_scratch:
        # Reinitialize weights
        rng, key = jax.random.split(rng)
        unet_params = unet.init_weights(key)

    # Load text encoder
    if args.unconditional or args.toy:
        tokenizer = text_encoder = None
    else:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            from_pt=args.from_pt,
            revision=args.revision,
            subfolder="tokenizer",
        )
        text_encoder = FlaxCLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path,
            from_pt=args.from_pt,
            revision=args.revision,
            subfolder="text_encoder",
            dtype=weight_dtype,
        )

    # Load VAE
    if args.toy:
        vae = vae_params = None
    else:
        vae, vae_params = FlaxAutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path,
            from_pt=args.from_pt,
            revision=args.revision,
            subfolder="vae",
            dtype=weight_dtype,
        )

    # Load noise scheduler
    noise_scheduler = FlaxPNDMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
    )
    noise_scheduler_state = noise_scheduler.create_state()

    # Optimization settings
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

    # Plotting configurations
    plot_freq = args.plot_freq
    safety_checker = None
    if args.toy:
        pipeline = FlaxToyDiffusionPipeline(
            unet=unet,
            scheduler=noise_scheduler,
        )
    else:
        pipeline = FlaxStableDiffusionPipeline(
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            tokenizer=tokenizer,
            scheduler=noise_scheduler,
            safety_checker=safety_checker,
            feature_extractor=None
        )
    num_inference_steps = 30
    H = args.resolution
    C = 3
    num_samples = jax.device_count() # 8 
    prng_seed = jax.random.PRNGKey(104) # fixed seed for plotting
    prng_seed = jax.random.split(prng_seed, num_samples)
    if args.unconditional or args.toy:
        prompt_ids = 1 # batch_size
    else:
        prompt = num_samples * [args.prompt] if args.prompt is not None \
                    else [ "A pikachu",
                            "A cat",
                            "A church",
                            "A painting of a beautiful sunset over a calm lake.",
                            "geodesic landscape, john chamberlain, christopher balaskas, tadao ando, 4 k",
                            "hibiscus rosa - sinensis, hand painted",
                            "A surreal dreamscape featuring floating islands, bizarre creatures, and a river that flows into the sky.",
                            "A dome."]
        prompt_ids = pipeline.prepare_inputs(prompt)
        prompt_ids = shard(prompt_ids)

    # FID configurations
    if args.fid_freq:
        fid_steps = 375 # fid_steps * 8 samples in total
        rng, key = jax.random.split(rng)
        model = inception.InceptionV3(pretrained=True)
        fid_fn = functools.partial(model.apply, train=False)
        fid_fn_p = jax.pmap(fid_fn)
        resize_fn = functools.partial(jax.image.resize,shape=(1, 256, 256, 3),method="bicubic") # 256x256 as is in most FID implementations for no obvious reasons
        resize_fn_p = jax.pmap(resize_fn)
        init_params = model.init(key, jnp.ones((1, H, H, C)))
        init_params_p = jax_utils.replicate(init_params)

        stats_path = os.path.join(args.output_dir, 'fid_stats.npz')
        if os.path.isfile(stats_path) and args.precomputed_fid_stats:
            stats = np.load(stats_path)
            mu0, sigma0 = stats["mu"], stats["sigma"]
            print('Loaded pre-computed statistics at:', stats_path)
        else:
            def preprocess_fid(examples):
                images = [image.convert("RGB") for image in examples[image_column]]
                examples["pixel_values"] = [train_transforms(image) for image in images]
                return examples
            
            def fid_collate_fn(examples):
                return torch.stack([example["pixel_values"] for example in examples]).numpy()
            
            fid_dataset = dataset["train"].with_transform(preprocess_fid)
            sampler = torch.utils.data.RandomSampler(fid_dataset, replacement=True, num_samples=int(1e10)) # Make it infinite
            fid_loader = torch.utils.data.DataLoader(fid_dataset, batch_size=num_samples, collate_fn=fid_collate_fn, sampler=sampler)
            fid_loader = iter(fid_loader)
            procs = []
            i = 0
            for x in tqdm(range(fid_steps),desc="Pre-Computing FID stats...", position=0):
                x = next(fid_loader)
                if x.shape[0] != num_samples:
                    continue
                i += 1
                if i >= fid_steps:
                    break
                x = np.moveaxis(x,-3,-1) # (8, H, H, C) -1<=x<=1
                x = shard(x)
                x = resize_fn_p(x)
                proc = fid_fn_p(init_params_p, jax.lax.stop_gradient(x))
                procs.append(proc.squeeze(axis=1).squeeze(axis=1).squeeze(axis=1))
            procs = jnp.concatenate(procs, axis=0)
            mu0 = np.mean(procs, axis=0)
            sigma0 = np.cov(procs, rowvar=False)
            np.savez(stats_path, mu=mu0, sigma=sigma0)
            print('Saved pre-computed statistics at:', stats_path, '. Set --precomputed_fid_stats flag to skip it next time!')
            del procs
    
    # Training function
    def train_step(state, text_encoder_params, vae_params, batch, train_rng):
        sample_rng, new_train_rng = jax.random.split(train_rng)

        def compute_loss(params):
            latents = batch["pixel_values"]
            if not args.toy:
                # Convert images to latent space
                vae_outputs = vae.apply(
                    {"params": vae_params}, latents, deterministic=True, method=vae.encode
                )
                latents = vae_outputs.latent_dist.sample(sample_rng)
                latents = latents * vae.config.scaling_factor
                # (NHWC) -> (NCHW)
                latents = jnp.transpose(latents, (0, 3, 1, 2))

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

            if not args.toy:
                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(
                    batch["input_ids"],
                    params=text_encoder_params,
                    train=False,
                )[0]
            else:
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

    # Replicate the train state on each device
    noise_scheduler_state_p = jax_utils.replicate(noise_scheduler_state)
    state = jax_utils.replicate(state)
    if args.toy:
        text_encoder_params = vae_params = None
    else:
        text_encoder_params = jax_utils.replicate(text_encoder.params)
        vae_params = jax_utils.replicate(vae_params)

    
    # EMA configurations
    if args.ema: 
        ema_decay = 0.99
        ema_freq = 100
        ema_apply_freq = 1000
        ema_params = jax.device_get(unet_params.copy())


    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader))
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)


    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel & distributed) = {total_train_batch_size}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    rng, key = jax.random.split(rng)
    train_rngs = jax.random.split(key, jax.local_device_count())

    epochs = tqdm(range(args.num_train_epochs), desc="Epoch ... ", position=0)
    for epoch in epochs:
        # ======================== Training ================================
        steps_per_epoch = len(train_dataset) // total_train_batch_size
        train_step_progress_bar = tqdm(total=steps_per_epoch, desc="Training...", position=1, leave=False)
        # train
        for batch in train_dataloader:
            batch = shard(batch) # CHW
            
            state, train_metric, train_rngs = p_train_step(state, text_encoder_params, vae_params, batch, train_rngs)

            train_step_progress_bar.update(1)
            global_step += 1

            writer.add_scalar("Loss/train", train_metric['loss'][0], global_step)

            if global_step >= args.max_train_steps:
                break
            if args.ema:
                if (global_step - 2) % ema_freq == 0: # do it off-device to save HBF memory
                    ema_params = jax.tree_util.tree_map(lambda ema, new: ema * ema_decay + (1 - ema_decay) * new, ema_params, get_params_to_save(state.params))
                # if (global_step - 2) % ema_apply_freq == 0: # copy back to state
                #     state.replace(params = jax_utils.replicate(ema_params)) 

            if global_step % plot_freq == 0 or global_step == 100:
                if args.toy:
                    params = {
                        "vae": vae_params,
                        "unet": state.params,
                        "scheduler": noise_scheduler_state_p,
                    }
                else:
                    params = {
                        "text_encoder": text_encoder_params,
                        "vae": vae_params,
                        "unet": state.params,
                        "scheduler": noise_scheduler_state_p,
                    }
                images = pipeline(prompt_ids, params, prng_seed, num_inference_steps,height=H,width=H,jit=True).images
                images = images.reshape(num_samples, H, H, C) # (8,1,...) -> (8,...), unreplicate
                images = images.transpose(0, 3, 1, 2)
                writer.add_images('Images', images, global_step)
                del images

            if args.fid_freq and global_step % args.fid_freq == 0:
                if args.toy:
                    params={
                        "vae": vae_params,
                        "unet": state.params,
                        "scheduler": noise_scheduler_state_p,
                    }
                else:
                    params={
                        "text_encoder": text_encoder_params,
                        "vae": vae_params,
                        "unet": state.params,
                        "scheduler": noise_scheduler_state_p,
                    }
                fid_bar = tqdm(desc="Computing FID stats...", total=fid_steps)
                procs = []
                for i in range(fid_steps): # fid_steps * 8 (num_devices) samples
                    rng, key = jax.random.split(rng)
                    keys = jax.random.split(key, num_samples)
                    images = pipeline(prompt_ids, params, keys, num_inference_steps,height=H,width=H,jit=True).images # on-device (8,1,H,H,C) 
                    images = resize_fn_p(images)
                    images = 2 * images - 1
                    proc = fid_fn_p(init_params_p, jax.lax.stop_gradient(images)) # Inception-Net States
                    procs.append(proc.squeeze(axis=1).squeeze(axis=1).squeeze(axis=1))
                    fid_bar.update(1)

                procs = jnp.concatenate(procs, axis=0)
                mu = np.mean(procs, axis=0)
                sigma = np.cov(procs, rowvar=False)
                stats_path = os.path.join(args.output_dir, f'fid_stats_step_{global_step}.npz')
                np.savez(stats_path, mu=mu0, sigma=sigma0)
                print('Saved statistics at:', stats_path, '.')
                
                fid_score = inception.fid_score(mu0,mu,sigma0,sigma)

                writer.add_scalar("FID/train", fid_score, global_step)
                del procs, images
                fid_bar.close()

        train_metric = jax_utils.unreplicate(train_metric)

        train_step_progress_bar.close()
        epochs.write(f"Epoch... ({epoch + 1}/{args.num_train_epochs} | Loss: {train_metric['loss']})")

    # Create the pipeline using using the trained modules and save it.
    if jax.process_index() == 0:
        state_snapshot = get_params_to_save(state.params.copy())
        if args.ema:
            ema_params = jax.tree_util.tree_map(lambda ema, new: ema * ema_decay + (1 - ema_decay) * new, ema_params, state_snapshot)
        else:
            ema_params = state_snapshot # no ema

        if args.toy:
            pipeline.save_pretrained(
                args.output_dir,
                params={
                    "vae": get_params_to_save(vae_params),
                    "unet": ema_params,
                },
            )
        else:
            pipeline.save_pretrained(
                args.output_dir,
                params={
                    "text_encoder": get_params_to_save(text_encoder_params),
                    "vae": get_params_to_save(vae_params),
                    "unet": ema_params,
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
