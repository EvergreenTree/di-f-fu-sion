import sys
sys.path.append('.')

import argparse
import functools
from typing import Any
from jax._src.dtypes import dtype
import jax.numpy as jnp
import jax

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import flax.linen as nn
import optax
from flax.training import train_state
import numpy as np
import wandb
import flax
from maxdiffusion.models.act_flax import colu, rcolu, make_conv

ACT = colu
CONV3D = False

############# Utils #############
from typing import Sequence
from torch.utils import data


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


class NumpyLoader(data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    ):
        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )


class FlattenAndCast(object):
    def __call__(self, pic):
        return np.array(pic.permute(1, 2, 0), dtype=jnp.float32)


def create_cos_anneal_schedule(base_lr, min_lr, max_steps):
    def learning_rate_fn(step):
        cosine_decay = (0.5) * (1 + jnp.cos(jnp.pi * step / max_steps))
        decayed = (1 - min_lr) * cosine_decay + min_lr
        return base_lr * decayed

    return learning_rate_fn


def compute_weight_decay(params):
    """Given a pytree of params, compute the summed $L2$ norm of the params.
    
    NOTE: For our case with SGD, weight decay ~ L2 regularization. This won't always be the 
    case (ex: Adam vs. AdamW).
    """
    param_norm = 0

    weight_decay_params_filter = flax.traverse_util.ModelParamTraversal(
        lambda path, _: ("bias" not in path and "scale" not in path)
    )

    weight_decay_params = weight_decay_params_filter.iterate(params)

    for p in weight_decay_params:
        if p.ndim > 1:
            param_norm += jnp.sum(p ** 2)

    return param_norm


############# Models #############
import jax
from typing import Any, Callable, Sequence, Optional
from jax import numpy as jnp
import flax
from flax import linen as nn
from functools import partial
import numpy as np

ModuleDef = Any
dtypedef = Any


class ResidualBlock(nn.Module):
    # Define collection of datafields here
    in_channels: int

    # For batchnorm, you can pass it as a ModuleDef
    norm: ModuleDef

    # dtype for fp16/32 training
    dtype: dtypedef = jnp.float32

    # define init for conv layers
    kernel_init: Callable = nn.initializers.kaiming_normal()

    @nn.compact
    def __call__(self, x):
        residual = x

        x = nn.Conv(
            kernel_size=(3, 3),
            strides=1,
            features=self.in_channels,
            padding="SAME",
            use_bias=False,
            kernel_init=self.kernel_init,
            dtype=self.dtype,
        )(x) if not CONV3D else make_conv('3x3',conv3d=True,features=self.in_channels,use_bias=False,dtype=self.dtype,)(x)
        x = self.norm()(x)
        x = ACT(x)
        x = nn.Conv(
            kernel_size=(3, 3),
            strides=1,
            features=self.in_channels,
            padding="SAME",
            use_bias=False,
            kernel_init=self.kernel_init,
            dtype=self.dtype,
        )(x) if not CONV3D else make_conv('3x3',conv3d=True,features=self.in_channels,use_bias=False,dtype=self.dtype,)(x)
        x = self.norm()(x)

        x = x + residual

        return ACT(x)


class DownSampleResidualBlock(nn.Module):
    # Define collection of datafields here
    in_channels: int
    out_channels: int

    # For batchnorm, you can pass it as a ModuleDef
    norm: ModuleDef

    # dtype for fp16/32 training
    dtype: dtypedef = jnp.float32

    # define init for conv layers
    kernel_init: Callable = nn.initializers.kaiming_normal()

    @nn.compact
    def __call__(self, x):
        residual = x

        x = nn.Conv(
            kernel_size=(3, 3),
            strides=1,
            features=self.in_channels,
            padding="SAME",
            use_bias=False,
            kernel_init=self.kernel_init,
            dtype=self.dtype,
        )(x) if not CONV3D else make_conv('3x3',conv3d=True,features=self.in_channels,use_bias=False,dtype=self.dtype,)(x)
        x = self.norm()(x)
        x = ACT(x)
        x = nn.Conv(
            kernel_size=(3, 3),
            strides=(2, 2),
            features=self.out_channels,
            padding=((1, 1), (1, 1)),
            use_bias=False,
            kernel_init=self.kernel_init,
            dtype=self.dtype,
        )(x) if not CONV3D else make_conv('down',conv3d=True,features=self.in_channels,use_bias=False,dtype=self.dtype,)(x)
        x = self.norm()(x)

        x = x + self.pad_identity(residual)

        return ACT(x)

    @nn.nowrap
    def pad_identity(self, x):
        # Pad identity connection when downsampling
        return jnp.pad(
            x[:, ::2, ::2, ::],
            ((0, 0), (0, 0), (0, 0), (self.out_channels // 4, self.out_channels // 4)),
            "constant",
        )


class ResNet(nn.Module):
    # Define collection of datafields here
    filter_list: Sequence[int]
    N: int
    num_classes: int

    # dtype for fp16/32 training
    dtype: dtypedef = jnp.float32

    # define init for conv and linear layers
    kernel_init: Callable = nn.initializers.kaiming_normal()

    # For train/test differences, want to pass “mode switches” to __call__
    @nn.compact
    def __call__(self, x, train):

        norm = partial(
            nn.BatchNorm,
            use_running_average=not train,
            momentum=0.1,
            epsilon=1e-5,
            dtype=self.dtype,
        )
        x = nn.Conv(
            kernel_size=(3, 3),
            strides=1,
            features=self.filter_list[0],
            padding="SAME",
            use_bias=False,
            kernel_init=self.kernel_init,
            dtype=self.dtype,
        )(x) #if not CONV3D else make_conv('3x3',conv3d=True,features=self.filter_list[0],use_bias=False,dtype=self.dtype,)(x)

        x = norm()(x)
        x = ACT(x)

        # First stage
        for _ in range(0, self.N - 1):
            x = ResidualBlock(
                in_channels=self.filter_list[0], norm=norm, dtype=self.dtype
            )(x)

        x = DownSampleResidualBlock(
            in_channels=self.filter_list[0],
            out_channels=self.filter_list[1],
            norm=norm,
            dtype=self.dtype,
        )(x)

        # Second stage
        for _ in range(0, self.N - 1):
            x = ResidualBlock(
                in_channels=self.filter_list[1], norm=norm, dtype=self.dtype
            )(x)

        x = DownSampleResidualBlock(
            in_channels=self.filter_list[1],
            out_channels=self.filter_list[2],
            norm=norm,
            dtype=self.dtype,
        )(x)

        # Third stage
        for _ in range(0, self.N):
            x = ResidualBlock(
                in_channels=self.filter_list[2], norm=norm, dtype=self.dtype
            )(x)

        # Global pooling
        x = jnp.mean(x, axis=(1, 2))

        x = x.reshape(x.shape[0], -1)
        x = nn.Dense(
            features=self.num_classes, kernel_init=self.kernel_init, dtype=self.dtype
        )(x)

        return x


def _resnet(layers, N, dtype=jnp.float32, num_classes=10):
    model = ResNet(filter_list=layers, N=N, dtype=dtype, num_classes=num_classes)
    return model


def ResNet20(
    dtype=jnp.float32,
):
    return _resnet(layers=[16, 32, 64], N=3, dtype=dtype, num_classes=10)


def ResNet32(
    dtype=jnp.float32,
):
    return _resnet(layers=[16, 32, 64], N=5, dtype=dtype, num_classes=10)


def ResNet44(
    dtype=jnp.float32,
):
    return _resnet(layers=[16, 32, 64], N=7, dtype=dtype, num_classes=10)


def ResNet56(
    dtype=jnp.float32,
):
    return _resnet(layers=[16, 32, 64], N=9, dtype=dtype, num_classes=10)


def ResNet110(
    dtype=jnp.float32,
):
    return _resnet(layers=[16, 32, 64], N=18, dtype=dtype, num_classes=10)





class TrainState(train_state.TrainState):
    batch_stats: Any = None
    weight_decay: Any = None


def parse():
    parser = argparse.ArgumentParser(description="Flax CIFAR10 Training")

    parser.add_argument(
        "-data",
        "--data",
        default="wandb/",
        type=str,
        metavar="DIR",
        help="path to dataset",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )

    parser.add_argument(
        "--epochs",
        default=180,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )

    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=128,
        type=int,
        metavar="N",
        help="mini-batch size per process (default: 128)",
    )

    parser.add_argument(
        "--weight-decay",
        "--wd",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
    )

    # My additional args
    parser.add_argument("--model", type=str, default="ResNet56")
    parser.add_argument("--CIFAR10", type=bool, default=True)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--base-lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--dtype", type=str, default="fp32")

    args = parser.parse_args()
    return args


def main():
    global best_prec1, args

    args = parse()

    model_dtype = jnp.float32 if args.dtype == "fp32" else jnp.float16

    if args.model == "ResNet20":
        model = ResNet20(dtype=model_dtype)

    elif args.model == "ResNet32":
        model = ResNet32(dtype=model_dtype)

    elif args.model == "ResNet44":
        model = ResNet44(dtype=model_dtype)

    elif args.model == "ResNet56":
        model = ResNet56(dtype=model_dtype)

    elif args.model == "ResNet110":
        model = ResNet110(dtype=model_dtype)

    # --------- Data Loading ---------#
    if args.CIFAR10:
        assert args.num_classes == 10, "Must have 10 output classes for CIFAR10"
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomCrop(
                    (32, 32),
                    padding=4,
                    fill=0,
                    padding_mode="constant",
                ),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
                ),
                FlattenAndCast(),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
                ),
                FlattenAndCast(),
            ]
        )

        train_dataset = CIFAR10(
            root="./CIFAR", train=True, download=True, transform=transform_train
        )
        train_dataset, validation_dataset = data.random_split(
            train_dataset, [45000, 5000]
        )

        test_dataset = CIFAR10(
            root="./CIFAR", train=False, download=True, transform=transform_test
        )

        train_loader = NumpyLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=False,
        )

        validation_loader = NumpyLoader(
            validation_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=False,
        )

        test_loader = NumpyLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
        )

    # --------- Create Train State ---------#
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    learning_rate_fn = optax.piecewise_constant_schedule(
        init_value=args.base_lr, boundaries_and_scales={32000: 0.1, 48000: 0.1}
    )

    state = create_train_state(
        init_rng,
        momentum=args.momentum,
        learning_rate_fn=learning_rate_fn,
        weight_decay=args.weight_decay,
        model=model,
    )
    del init_rng

    # Setup WandB logging here
    wandb_run = wandb.init(project="Flax Torch")
    wandb.config.max_epochs = args.epochs
    wandb.config.batch_size = args.batch_size
    wandb.config.weight_decay = args.weight_decay

    wandb.config.ModelName = args.model
    wandb.config.Dataset = "CIFAR10"
    wandb.config.Package = "Flax"

    # --------- Training ---------#
    for epoch in range(0, args.epochs):
        state, train_epoch_metrics_np = train_epoch(state, train_loader, epoch)

        print(
            f"train epoch: {epoch}, loss: {train_epoch_metrics_np['loss']:.4f}, accuracy:{train_epoch_metrics_np['accuracy']*100:.2f}%"
        )

        # Get LR:
        step = epoch * args.batch_size
        lr = learning_rate_fn(step)
        lr_np = jax.device_get(lr)

        # Validation set metrics:
        validation_loss, _ = eval_model(state, validation_loader)

        if epoch % 10 == 0:
            _, test_accuracy = eval_model(state, test_loader)

            wandb.log(
                {
                    "acc@1": test_accuracy * 100,
                    "Learning Rate": lr_np,
                    "Training Loss": train_epoch_metrics_np["loss"],
                    "Validation Loss": validation_loss,
                }
            )

        else:
            wandb.log(
                {
                    "Learning Rate": lr_np,
                    "Training Loss": train_epoch_metrics_np["loss"],
                    "Validation Loss": validation_loss,
                }
            )


# --------- Helper Functions: Loss, Train Step, Eval, Etc ---------#
@jax.jit
def cross_entropy_loss(*, logits, labels):
    """
    Softmax + CE Loss
    """
    one_hot_labels = jax.nn.one_hot(labels, num_classes=10)
    return -jnp.mean(jnp.sum(one_hot_labels * nn.log_softmax(logits, axis=-1), axis=-1))


def compute_metrics(*, logits, labels):
    loss = cross_entropy_loss(logits=logits, labels=labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {
        "loss": loss,
        "accuracy": accuracy,
    }
    return metrics


def initialized(key, image_size, model):
    input_shape = (1, image_size, image_size, 3)

    @jax.jit
    def init(rng, shape):
        return model.init(rng, shape, train=True)

    variables = init(rng=key, shape=jnp.ones(input_shape, dtype=model.dtype))
    return variables["params"], variables["batch_stats"]


def create_train_state(rng, momentum, learning_rate_fn, weight_decay, model):
    """Creates initial `TrainState`."""
    params, batch_stats = initialized(rng, 32, model)
    tx = optax.sgd(learning_rate=learning_rate_fn, momentum=momentum, nesterov=True)
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats,
        weight_decay=weight_decay,
    )
    return state


@jax.jit
def train_step(state, batch, labels):
    """Train for a single step."""

    def loss_fn(params):
        logits, new_state = state.apply_fn(
            {"params": params, "batch_stats": state.batch_stats},
            batch,
            mutable=["batch_stats"],
            train=True,
        )
        loss = cross_entropy_loss(logits=logits, labels=labels)

        loss = loss + 0.5 * state.weight_decay * compute_weight_decay(params)

        return loss, (logits, new_state)


    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(state.params)

    logits, new_state = aux[1]

    state = state.apply_gradients(
        grads=grads,
        batch_stats=new_state["batch_stats"],
    )
    metrics = compute_metrics(logits=logits, labels=labels)


    return state, metrics


@jax.jit
def eval_step(state, batch, labels):
    logits = state.apply_fn(
        {"params": state.params, "batch_stats": state.batch_stats},
        batch,
        mutable=False,
        train=False,
    )
    return compute_metrics(logits=logits, labels=labels)


def train_epoch(state, dataloader, epoch):
    """Train for a single epoch."""
    batch_metrics = []

    for images, labels in dataloader:
        state, metrics = train_step(state, images, labels)
        batch_metrics.append(metrics)

    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np])
        for k in batch_metrics_np[0]
    }
    return state, epoch_metrics_np


def eval_model(state, dataloader):
    batch_metrics = []
    for images, labels in dataloader:
        metrics = eval_step(state, images, labels)
        batch_metrics.append(metrics)
    batch_metrics_np = jax.device_get(batch_metrics)
    validation_metrics_np = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np])
        for k in batch_metrics_np[0]
    }

    return validation_metrics_np["loss"], validation_metrics_np["accuracy"]


if __name__ == "__main__":
    main()
