import os, sys, pdb
from eval import evaluate
sys.path.append("..")
sys.path.append("../modules")

from os.path import join
import pathlib
import ruamel.yaml as yaml
from tqdm import tqdm

import wandb
import optax

import numpy as onp
import jax.numpy as jnp
import jax
from jax import jit, lax, vmap, value_and_grad, config
import haiku as hk
from jax.tree_util import tree_map
from jax.flatten_util import ravel_pytree

from modules.vae_model_init import init_vector_vae_params, get_vae_auroc, get_cross_correlation, get_mse, get_covar, get_single_kl
import utils

config.update("jax_enable_x64", True)

# Load config yaml as options for experiment
configs = yaml.safe_load((pathlib.Path("..") / "biols_config.yaml").read_text())
opt, folder_path = utils.load_yaml(configs)
opt.obs_data = opt.n_pairs

# Set seeds
onp.random.seed(opt.data_seed)
rng_key = jax.random.PRNGKey(opt.data_seed)
key = hk.PRNGSequence(opt.data_seed)

# Set some constants
opt.num_samples = 2 * opt.n_pairs
n = opt.num_samples
d = opt.num_nodes
l_dim = d * (d - 1) // 2

logdir = utils.set_logdir(opt)
os.makedirs(logdir, exist_ok=True)

# Load saved data
x_samples, z_samples, interv_labels, interv_targets, interv_values, interv_noise = utils.read_biols_dataset(folder_path)
gt_W = onp.load(f'{folder_path}/weighted_adjacency.npy')
gt_P = onp.load(f'{folder_path}/perm.npy')
gt_L = onp.load(f'{folder_path}/edge_weights.npy')
gt_sigmas = onp.load(f'{folder_path}/gt_sigmas.npy')
binary_gt_W = jnp.where(jnp.abs(gt_W) >= 0.3, 1, 0)
print(gt_W)

L_mse = jnp.mean(gt_W ** 2)

if opt.off_wandb is False:   
    wandb.init(project = opt.wandb_project, 
                entity = opt.wandb_entity, 
                config = vars(opt), 
                settings = wandb.Settings(start_method="fork"))
    wandb.run.name = logdir.split('/')[-1]
    wandb.run.save()
    utils.save_graph_images(gt_P, gt_L, gt_W, 'gt_P.png', 'gt_L.png', 'gt_w.png', logdir)
    wandb.log({"graph_structure(GT-pred)/Ground truth W": wandb.Image(join(logdir, 'gt_w.png'))}, step=0)

p_z_mu = jnp.zeros((d))
p_z_covar = jnp.eye(d)

forward, model_params, model_opt_params, opt_model = init_vector_vae_params(opt, opt.proj_dims, key, rng_key, x_samples)
pred_sigma = 1.

def get_neg_elbo(model_params, rng_key, x_data):
    X_recons, z_pred, q_z_mus, z_L_chols = forward.apply(model_params, rng_key, d, opt.proj_dims, rng_key, x_data, opt.corr)
    nll = utils.nll_gaussian(z_samples, z_pred, pred_sigma)
    q_z_covars = vmap(get_covar, 0, 0)(z_L_chols)
    kl_loss = jnp.mean(vmap(get_single_kl, (None, None, 0, 0, None), 0)(p_z_covar, p_z_mu, q_z_covars, q_z_mus, opt))
    return (1e-4 * (nll + kl_loss))

@jit
def gradient_step(model_params, x_data):
    loss, grads = value_and_grad(get_neg_elbo, argnums=(0))(model_params, rng_key, x_data)
    X_recons, z_pred, q_z_mus, z_L_chols = forward.apply(model_params, rng_key, d, opt.proj_dims, rng_key, x_data, opt.corr)
    q_z_covars = vmap(get_covar, 0, 0)(z_L_chols)
    log_dict = {
        "z_mse": jnp.mean(get_mse(z_pred, z_samples)),
        "x_mse": jnp.mean(get_mse(x_data[:, :opt.proj_dims], X_recons)),
        "L_MSE": L_mse
    }
    return X_recons, loss, grads, z_pred, log_dict

@jit
def update_params(grads, model_opt_params, model_params):
    model_updates, model_opt_params = opt_model.update(grads, model_opt_params, model_params)
    model_params = optax.apply_updates(model_params, model_updates)
    return model_params, model_opt_params

mcc = 50.
with tqdm(range(opt.num_steps)) as pbar:
    for i in pbar:
        X_recons, loss, grads, z_pred, log_dict = gradient_step(model_params, x_samples)
        try:
            mcc = get_cross_correlation(onp.array(z_pred), onp.array(z_samples))
        except:
            pass
        auroc = get_vae_auroc(d, gt_W)

        wandb_dict = {
            "ELBO": onp.array(loss),
            "Z_MSE": onp.array(log_dict["z_mse"]),
            "X_MSE": onp.array(log_dict["x_mse"]),
            "L_MSE": L_mse,
            "Evaluations/SHD": jnp.sum(binary_gt_W),
            "Evaluations/AUROC": auroc,
            'Evaluations/MCC': mcc,
        }
        
        if (i+1) % 50 == 0 or i == 0:       
            if opt.off_wandb is False:  
                wandb.log(wandb_dict, step=i)

        pbar.set_postfix(
            ELBO=f"{loss:.4f}",
            SHD=jnp.sum(binary_gt_W),
            MCC=mcc, 
            L_mse=f"{log_dict['L_MSE']:.3f}",
            AUROC=f"{auroc:.2f}"
        )
        # pdb.set_trace()
        model_params, model_opt_params = update_params(grads, model_opt_params, model_params)
        if jnp.any(jnp.isnan(ravel_pytree(model_params)[0])):  
            raise Exception("Got NaNs in model params")