import os, sys, pdb
from eval import evaluate
import wandb
import optax
sys.path.append("..")
sys.path.append("../modules")

from os.path import join
import pathlib
import ruamel.yaml as yaml
from tqdm import tqdm
from collections import namedtuple

import numpy as onp
import jax.numpy as jnp
import jax
from jax import jit, vmap, value_and_grad, config
import haiku as hk
from jax.flatten_util import ravel_pytree

from modules.vae_model_init import init_vector_vae_params, get_vae_auroc, get_cross_correlation, get_mse, get_covar, get_single_kl
import utils

config.update("jax_enable_x64", True)
Losses = namedtuple('Losses', ['nll', 'x_mse', 'z_mse', 'kl_loss', 'neg_elbo'])
PredSamples = namedtuple('PredSamples', ['z', 'x'])

# Load config yaml as options for experiment
configs = yaml.safe_load((pathlib.Path("..") / "biols_config.yaml").read_text())
opt, folder_path = utils.load_yaml(configs)
opt.obs_data = opt.n_pairs

# Set seeds
onp.random.seed(opt.data_seed)
rng_key = jax.random.PRNGKey(opt.data_seed)
key = hk.PRNGSequence(opt.data_seed)

# Set some constants
logdir = utils.set_logdir(opt)
os.makedirs(logdir, exist_ok=True)
opt.num_samples = 2 * opt.n_pairs
l_dim = opt.num_nodes * (opt.num_nodes - 1) // 2
pred_sigma = opt.pred_sigma

# Load saved data
gt_samples, interventions = utils.read_biols_dataset(folder_path)
gt_W = onp.load(f'{folder_path}/weighted_adjacency.npy')
gt_P = onp.load(f'{folder_path}/perm.npy')
gt_L = onp.load(f'{folder_path}/edge_weights.npy')
gt_sigmas = onp.load(f'{folder_path}/gt_sigmas.npy')
binary_gt_W = jnp.where(jnp.abs(gt_W) >= opt.edge_threshold, 1, 0)
print(gt_W)

L_mse = jnp.mean(gt_W ** 2)
p_z_mu = jnp.zeros((opt.num_nodes))
p_z_covar = jnp.eye(opt.num_nodes)
mcc = 50.

if opt.off_wandb is False:   
    wandb.init(project = opt.wandb_project, 
                entity = opt.wandb_entity, 
                config = vars(opt), 
                settings = wandb.Settings(start_method="fork"))
    wandb.run.name = logdir.split('/')[-1]
    wandb.run.save()
    utils.save_graph_images(gt_P, gt_L, gt_W, 'gt_P.png', 'gt_L.png', 'gt_w.png', logdir)
    wandb.log({"graph_structure(GT-pred)/Ground truth W": wandb.Image(join(logdir, 'gt_w.png'))}, step=0)


def get_neg_elbo(rng_key, params, gt_samples):
    x_pred, z_pred, q_z_mus, z_L_chols = forward.apply(params, rng_key, rng_key, opt.num_nodes, gt_samples.x, opt.corr)
    z_mse = jnp.mean(get_mse(z_pred, gt_samples.z))
    x_mse = jnp.mean(get_mse(x_pred, gt_samples.x))
    pred_samples = PredSamples(z=z_pred, x=x_pred)

    nll = utils.nll_gaussian(gt_samples.x, x_pred, pred_sigma)
    q_z_covars = vmap(get_covar, 0, 0)(z_L_chols)
    kl_loss = jnp.mean(vmap(get_single_kl, (None, None, 0, 0, None), 0)(p_z_covar, p_z_mu, q_z_covars, q_z_mus, opt))
    neg_elbo = nll + kl_loss
    losses = Losses(
        z_mse=z_mse,
        x_mse=x_mse,
        nll=nll,
        kl_loss=kl_loss,
        neg_elbo=neg_elbo,
    )
    return neg_elbo, (losses, pred_samples)


@jit
def get_gradients(params, gt_samples):
    (loss, aux_res), grads = value_and_grad(get_neg_elbo, argnums=(1), has_aux=True)(rng_key, params, gt_samples)
    losses, pred_samples = aux_res
    return losses, grads, pred_samples


@jit
def update_params(params, grads, opt_state):
    model_updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, model_updates)
    return params, opt_state

forward, params, optimizer, opt_state = init_vector_vae_params(opt, key, rng_key, gt_samples.x)


with tqdm(range(opt.num_steps)) as pbar:
    for i in pbar:
        losses, grads, pred_samples = get_gradients(params, gt_samples)
        if jnp.any(jnp.isnan(ravel_pytree(grads)[0])):  
            raise Exception("Got NaNs in model gradients")

        params, opt_state = update_params(params, grads, opt_state)

        # Get AUROC and MCC
        auroc = get_vae_auroc(opt.num_nodes, gt_W)
        try:
            mcc = get_cross_correlation(onp.array(pred_samples.z), onp.array(gt_samples.z))
        except:
            pass

        wandb_dict = {
            "ELBO": losses.neg_elbo,
            "Z_MSE": losses.z_mse,
            "X_MSE": losses.x_mse,
            "NLL": losses.nll,
            "KL Loss": losses.kl_loss,
            "L_MSE": L_mse,
            "Evaluations/SHD": jnp.sum(binary_gt_W),
            "Evaluations/AUROC": auroc,
            'Evaluations/MCC': mcc,
        }
        
        if (i+1) % 50 == 0 or i == 0:       
            if opt.off_wandb is False:  
                wandb.log(wandb_dict, step=i)

        pbar.set_postfix(
            ELBO=f"{losses.neg_elbo:.4f}",
            Z_MSE=f"{losses.z_mse:.4f}",
            X_MSE=f"{losses.x_mse:.4f}",
            SHD=jnp.sum(binary_gt_W),
            MCC=mcc, 
            L_mse=f"{L_mse:.3f}",
            AUROC=f"{auroc:.2f}"
        )