import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    config.workdir = '/home/evergreen/nfs_share/di-f-fu-sion/oxford_flowers102-colu-e1'
    config.wandb_artifact = None

    # wandb
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.entity = None
    wandb.project = "ddpm-flax-flower102-colu-e1"
    wandb.job_type = "training"
    wandb.name = None 
    wandb.log_train = True
    wandb.log_sample = True
    wandb.log_model = True

    # training
    config.training = training = ml_collections.ConfigDict()
    training.num_train_steps = 700000
    training.log_every_steps = 100
    training.loss_type = 'l1'
    training.half_precision = False
    training.save_and_sample_every = 1000
    training.num_sample = 64

    # ema
    config.ema = ema = ml_collections.ConfigDict()
    ema.beta = 0.995
    ema.update_every = 10
    ema.update_after_step = 100
    ema.inv_gamma = 1.0
    ema.power = 2 / 3
    ema.min_value = 0.0

    # ddpm 
    config.ddpm = ddpm = ml_collections.ConfigDict()
    ddpm.beta_schedule = 'cosine'
    ddpm.timesteps = 1000
    ddpm.p2_loss_weight_gamma = 1. # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
    ddpm.p2_loss_weight_k = 1
    ddpm.self_condition = True # not tested yet
    ddpm.pred_x0 = False # by default, the model will predict noise, if True predict x0

    # data
    config.data = data = ml_collections.ConfigDict()
    data.dataset ='oxford_flowers102'
    data.batch_size = 16
    data.cache = False
    data.image_size = 64
    data.channels = 3

    # model
    config.model = model = ml_collections.ConfigDict()
    model.block_out_channels = (64,128,256,256)
    model.layers_per_block = 1
    model.act_fn = 'rcolu'

    # optim
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = 'Adam'
    optim.lr = 1e-4
    optim.beta1 = 0.9
    optim.beta2 = 0.99
    optim.eps = 1e-8

    config.seed = 42

    return config


