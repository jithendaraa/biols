import os, sys, pdb
sys.path.append("..")
sys.path.append("../modules")
import argparse

import pathlib
import ruamel.yaml as yaml
import wandb

from datagen import SyntheticDatagen
import numpy as onp
import jax
from jax import jit, lax, vmap, value_and_grad, config
from jax import numpy as jnp
import haiku as hk
import utils

config.update("jax_enable_x64", True)

def load_yaml(configs):
    """
        Takes in a config dict return options as Namespace

        Parameters
        ----------
        configs: dict
            Configuration of the experiment to be run

        Returns
        -------
        opt: argparse.Namespace
    """
    default_config = 'defaults'
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', default=default_config)
    args, remaining = parser.parse_known_args()
    defaults = {}
    names = args.configs

    if isinstance(names, list) is False: names = names.split(' ')
    for name in names:  defaults.update(configs[name])

    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
      arg_type = utils.args_type(value)
      parser.add_argument(f'--{key}', type=arg_type, default=arg_type(value))
    
    opt = parser.parse_args(remaining)

    try:
      opt.num_samples = int(opt.pts_per_interv * opt.n_interv_sets) + opt.obs_data
    except:
      opt.num_samples = 2 * opt.n_pairs

    return opt


# Load config yaml as options for experiment
configs = yaml.safe_load((pathlib.Path(".") / "create_dataset_config.yaml").read_text())
opt = load_yaml(configs)

# Set seeds
onp.random.seed(opt.data_seed)
rng_key = jax.random.PRNGKey(opt.data_seed)
hk_key = hk.PRNGSequence(opt.data_seed)

num_interv_data = opt.pts_per_interv * opt.n_interv_sets
opt.num_samples = 2 * opt.n_pairs

assert opt.n_pairs == num_interv_data # always has to be true; else wscrl cannot be run as a baseline
assert opt.n_pairs % opt.n_interv_sets == 0
assert opt.max_interv_value == -opt.min_interv_value

artifact_metadata = {
    'datagen_type': opt.datagen_type,
    'graph_type': opt.graph_type,
    'num_nodes': opt.num_nodes,
    'proj_dims': opt.proj_dims,
    'projection': opt.proj,
    'exp_edges': opt.exp_edges,
    'num_samples': opt.num_samples,
    'n_pairs': opt.n_pairs,
    'interv_data': num_interv_data,
    'interv_type': opt.interv_type,
    'n_interv_sets': opt.n_interv_sets,
    'pts_per_interv': opt.pts_per_interv,
    'interv_value_sampling': opt.interv_value_sampling,
    'dataset': opt.dataset,
    'noise_sigma': opt.noise_sigma,
    'decoder_sigma': opt.decoder_sigma,
    'sem_type': opt.sem_type,
    'fix_noise': opt.fix_noise,
    'no_interv_noise': opt.no_interv_noise
}

if opt.interv_value_sampling == 'uniform':
    artifact_metadata['min_interv_value'] = opt.min_interv_value
    artifact_metadata['max_interv_value'] = opt.max_interv_value

# TODO: To remove this, foldername has to be changed to image-{foldername}
assert opt.dataset == 'vector'

if opt.graph_type == 'erdos-renyi':
    zfilled_nodes = str(opt.num_nodes).zfill(3)
    zfilled_proj_dims = str(opt.proj_dims).zfill(4)
    
    fix_noise_str = 'fix_noise' if opt.fix_noise is True else 'nofix_noise'
    interv_noise_str = 'no_interv_noise' if opt.no_interv_noise is True else 'interv_noise'

    if opt.datagen_type == 'weakly_supervised':
        folder_name = f'er{int(opt.exp_edges)}-ws_datagen_{fix_noise_str}_{interv_noise_str}-{opt.proj}proj-d{zfilled_nodes}-D{zfilled_proj_dims}-{opt.interv_type}-n_pairs{opt.n_pairs}-sets{opt.n_interv_sets}-{opt.interv_value_sampling}interv'
    elif opt.datagen_type == 'default':
        folder_name = f'er{int(opt.exp_edges)}-def_datagen_{fix_noise_str}_{interv_noise_str}-{opt.proj}proj-d{zfilled_nodes}-D{zfilled_proj_dims}-{opt.interv_type}-n_pairs{opt.n_pairs}-sets{opt.n_interv_sets}-{opt.interv_value_sampling}interv'
    else:
        raise NotImplementedError

elif opt.graph_type == 'scale-free':
    raise NotImplementedError

# Instantiate random SCM with structure and parameters
scm = SyntheticDatagen(
    data_seed=opt.data_seed,
    hk_key=hk_key,
    rng_key=rng_key,
    num_nodes=opt.num_nodes,
    degree=opt.exp_edges,
    interv_type=opt.interv_type,
    proj_dims=opt.proj_dims,
    projection=opt.proj,
    decoder_sigma=opt.decoder_sigma,
    interv_value_sampling=opt.interv_value_sampling,
    datagen_type=opt.datagen_type,
    sem_type=opt.sem_type,
    graph_type=opt.graph_type,
    dataset_type='linear',
    min_interv_value=opt.min_interv_value,
)

if opt.datagen_type == 'weakly_supervised':
    x1, x2, z1, z2, labels, interv_targets, interv_values, interv_noise = scm.sample_weakly_supervised(
        rng_key, 
        opt.n_pairs, 
        opt.n_interv_sets, 
        return_interv_values=True, 
        fix_noise=opt.fix_noise, 
        no_interv_noise=opt.no_interv_noise,
        return_interv_noise=True
    )

    x_samples = jnp.concatenate([x1, x2], axis=0)
    z_samples = jnp.concatenate([z1, z2], axis=0)
    interv_nodes = jnp.concatenate((jnp.ones_like(labels) * opt.num_nodes, labels), axis=0)
    if opt.interv_type == 'single':     interv_nodes = interv_nodes[:, None]
    interv_targets = jnp.concatenate([jnp.zeros(z1.shape).astype(int), interv_targets], axis=0)
    interv_values = jnp.concatenate([jnp.zeros(z1.shape), interv_values], axis=0)
    interv_noise = jnp.concatenate([jnp.zeros(z1.shape).astype(int), interv_noise], axis=0)

elif opt.datagen_type == 'default':
    x_samples, z_samples, interv_nodes, interv_targets, interv_values = scm.sample_default(rng_key)

gt_W, gt_P, gt_L = scm.W, scm.P, scm.P.T @ scm.W.T @ scm.P
gt_sigmas = jnp.exp(scm.log_sigma_W)

folder_path = os.path.join(opt.baseroot, 'scratch/biols_datasets', folder_name, str(opt.data_seed).zfill(2))
os.makedirs(folder_path, exist_ok=True)
onp.save(f'{folder_path}/x_samples.npy', x_samples)
onp.save(f'{folder_path}/z_samples.npy', z_samples)
onp.save(f'{folder_path}/interv_nodes.npy', interv_nodes)
onp.save(f'{folder_path}/interv_targets.npy', interv_targets)
onp.save(f'{folder_path}/interv_values.npy', interv_values)
onp.save(f'{folder_path}/interv_noise.npy', interv_noise)

onp.save(f'{folder_path}/weighted_adjacency.npy', gt_W)
onp.save(f'{folder_path}/perm.npy', gt_P)
onp.save(f'{folder_path}/edge_weights.npy', gt_L)
onp.save(f'{folder_path}/gt_sigmas.npy', gt_sigmas)

onp.save(f'{folder_path}/artifact_metadata.npy', artifact_metadata)
print(f'Saved datasets at {folder_path}')
print("DONE")