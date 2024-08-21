import sys
sys.path.append('.') # no need to install module
from absl import app
from absl import flags
from ml_collections import config_flags
from fusion.train import *
from jax_smi import initialise_tracking
initialise_tracking()

FLAGS = flags.FLAGS

logging.set_verbosity(logging.WARNING)

config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training or sampling hyperparameter configuration.',
    lock_config=True)

def train(config):
    workdir = config.workdir
    # create writer 
    writer = metric_writers.create_default_writer(
        logdir=workdir, just_logging=jax.process_index() != 0)
    # set up wandb run
    if jax.process_index() == 0:
        wandb_config = to_wandb_config(config)
        wandb.init(
                entity=config.wandb.entity,
                project=config.wandb.project,
                job_type=config.wandb.job_type,
                name=config.wandb.name,
                config=wandb_config)
    # set default x-axis as 'train/step'
    #wandb.define_metric("*", step_metric="train/step")

    sample_dir = os.path.join(workdir, "samples")

    rng = jax.random.PRNGKey(config.seed)

    rng, d_rng = jax.random.split(rng) 
    train_iter = get_dataset(d_rng, config)
    
    num_steps = config.training.num_train_steps
    
    rng, state_rng = jax.random.split(rng)
    state = create_train_state(state_rng, config)
    params_sizes = jax.tree_util.tree_map(jax.numpy.size, state.params)
    num_model_parameters = jax.tree_util.tree_reduce(lambda x, y: x + y, params_sizes)
    logging.info(f"number parameters: {num_model_parameters/10**9:.3f} billion")
    print(f"number parameters: {num_model_parameters/10**9:.3f} billion")
    wandb_artifact = config.wandb_artifact
    if wandb_artifact is not None:
        logging.info(f'loading model from wandb: {wandb_artifact}')
        state = load_wandb_model(state, workdir, wandb_artifact)
    else:
        state = restore_checkpoint(state, workdir)
    # step_offset > 0 if restarting from checkpoint
    step_offset = int(state.step)
    state = jax_utils.replicate(state)
    
    loss_fn = get_loss_fn(config)
    
    ddpm_params = get_ddpm_params(config.ddpm)
    ema_decay_fn = create_ema_decay_schedule(config.ema)
    train_step = functools.partial(p_loss, ddpm_params=ddpm_params, loss_fn =loss_fn, self_condition=config.ddpm.self_condition, is_pred_x0=config.ddpm.pred_x0, pmap_axis ='batch')
    p_train_step = jax.pmap(train_step, axis_name = 'batch')
    p_apply_ema = jax.pmap(apply_ema_decay, in_axes=(0, None), axis_name = 'batch')
    p_copy_params_to_ema = jax.pmap(copy_params_to_ema, axis_name='batch')

    train_metrics = []
    hooks = []

    sample_step = functools.partial(ddpm_sample_step, ddpm_params=ddpm_params, self_condition=config.ddpm.self_condition, is_pred_x0=config.ddpm.pred_x0)
    p_sample_step = jax.pmap(sample_step, axis_name='batch')

    if jax.process_index() == 0:
        hooks += [periodic_actions.Profile(num_profile_steps=5, logdir=workdir)]
    train_metrics_last_t = time.time()
    logging.info('Initial compilation, this might take some minutes...')
    for step, batch in zip(tqdm(range(step_offset, num_steps)), train_iter):
        rng, *train_step_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
        train_step_rng = jnp.asarray(train_step_rng)
        state, metrics = p_train_step(train_step_rng, state, batch)
        for h in hooks:
            h(step)
        if step == step_offset:
            logging.info('Initial compilation completed.')
            logging.info(f"Number of devices: {batch['image'].shape[0]}")
            logging.info(f"Batch size per device {batch['image'].shape[1]}")
            logging.info(f"input shape: {batch['image'].shape[2:]}")

        # update state.params_ema
        if (step + 1) <= config.ema.update_after_step:
            state = p_copy_params_to_ema(state)
        elif (step + 1) % config.ema.update_every == 0:
            ema_decay = ema_decay_fn(step)
            logging.info(f'update ema parameters with decay rate {ema_decay}')
            state =  p_apply_ema(state, ema_decay)

        if config.training.get('log_every_steps'):
            train_metrics.append(metrics)
            if (step + 1) % config.training.log_every_steps == 0:
                train_metrics = common_utils.get_metrics(train_metrics)
                summary = {
                        f'train/{k}': v
                        for k, v in jax.tree.map(lambda x: x.mean(), train_metrics).items()
                    }
                summary['time/seconds_per_step'] =  (time.time() - train_metrics_last_t) /config.training.log_every_steps
                summary['num_model_parameters'] = num_model_parameters
                writer.write_scalars(step + 1, summary)
                train_metrics = []
                train_metrics_last_t = time.time()

                if config.wandb.log_train and jax.process_index() == 0:
                    wandb.log({
                        "train/step": step, ** summary
                    })
        
        # Save a checkpoint periodically and generate samples.
        if (step + 1) % config.training.save_and_sample_every == 0 or step + 1 == num_steps:
            # generate and save sampling 
            logging.info(f'generating samples....')
            samples = []
            for i in trange(0, config.training.num_sample, config.data.batch_size):
                rng, sample_rng = jax.random.split(rng)
                samples.append(sample_loop(sample_rng, state, tuple(batch['image'].shape), p_sample_step, config.ddpm.timesteps))
            samples = jnp.concatenate(samples) 
            # TODO: HF unet unecessarily moves the channel dimension
            samples = jnp.moveaxis(samples,-3,-1) # num_devices, batch, H, W, C
            
            this_sample_dir = os.path.join(sample_dir, f"iter_{step}_host_{jax.process_index()}")
            tf.io.gfile.makedirs(this_sample_dir)
            
            with tf.io.gfile.GFile(
                os.path.join(this_sample_dir, "sample.png"), "wb") as fout:
                samples_array = save_image(samples, config.training.num_sample, fout, padding=2)
                if config.wandb.log_sample and jax.process_index() == 0:
                    wandb_log_image(samples_array, step+1)
            # save the chceckpoint
            save_checkpoint(state, workdir)
            # if step + 1 == num_steps and config.wandb.log_model:
            #     wandb_log_model(workdir, step+1)

    # Wait until computations are done before exiting
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

    return state

def main(argv):
    config = FLAGS.config
    train(config)

if __name__ == '__main__':
    flags.mark_flags_as_required(['config'])
    app.run(main)