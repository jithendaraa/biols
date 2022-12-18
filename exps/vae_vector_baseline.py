import os
import sys, pdb
from typing import OrderedDict
sys.path.append("..")
sys.path.append("../modules")

from os.path import join
import pathlib
import ruamel.yaml as yaml
from tqdm import tqdm
from collections import OrderedDict

import matplotlib.pyplot as plt
import wandb
import optax

import numpy as onp
import jax.numpy as jnp
import jax
from jax import jit, lax, vmap, value_and_grad, config
import haiku as hk
from tensorflow_probability.substrates.jax.distributions import Horseshoe
from jax.tree_util import tree_map
from jax.flatten_util import ravel_pytree

from modules.vae_model_init import init_vector_vae_params, get_vae_auroc, get_cross_correlation, get_mse, get_covar, get_single_kl
from modules.SyntheticSCM import SyntheticSCM

import utils
import datagen
from eval import evaluate

config.update("jax_enable_x64", True)

# Load config yaml as options for experiment
configs = yaml.safe_load((pathlib.Path("..") / "configs.yaml").read_text())
opt = utils.load_yaml(configs)

# Set seeds
onp.random.seed(opt.data_seed)
rng_key = jax.random.PRNGKey(opt.data_seed)
key = hk.PRNGSequence(opt.data_seed)

# Set some constants
hard = True
opt.num_samples = int(opt.pts_per_interv * opt.n_interv_sets) + opt.obs_data
n = opt.num_samples
d = opt.num_nodes
l_dim = d * (d - 1) // 2

if opt.eq_noise_var:    
    noise_dim = 1
    log_sigma_W = jnp.zeros(d)
else:   
    noise_dim = d
    log_sigma_W = onp.random.uniform(low=0, high=jnp.log(2), size=(d,))

degree = opt.exp_edges
num_bethe_iters = opt.bethe_iters
horseshoe_tau = utils.set_horseshoe_tau(n, d, degree)
logdir = utils.set_logdir(opt)
os.makedirs(logdir, exist_ok=True)

# Instantiate random ER-DAG, with SCM structure and parameters
sd = SyntheticSCM(
    n=opt.num_samples,
    d=opt.num_nodes,
    graph_type=opt.graph_type,
    degree=2 * degree,
    sem_type=opt.sem_type,
    sigmas=jnp.exp(log_sigma_W),
    dataset_type='linear',
    data_seed=opt.data_seed,
)

gt_W, gt_P, gt_L = sd.W, sd.P, sd.P.T @ sd.W.T @ sd.P
gt_sigmas = jnp.exp(log_sigma_W)
binary_gt_W = jnp.where(jnp.abs(gt_W) >= 0.3, 1, 0)
print(gt_W)
L_mse = jnp.mean(gt_W ** 2)

plt.imshow(gt_W)
plt.savefig(join(logdir, 'gt_w.png'))
if opt.off_wandb is False:
    if opt.offline_wandb is True: os.system('wandb offline')
    else:   os.system('wandb online')    
    wandb.init(project = opt.wandb_project, 
                entity = opt.wandb_entity, 
                config = vars(opt), 
                settings = wandb.Settings(start_method="fork"))
    wandb.run.name = logdir.split('/')[-1]
    wandb.run.save()
    wandb.log({"graph_structure(GT-pred)/Ground truth W": wandb.Image(join(logdir, 'gt_w.png'))}, step=0)

p_z_obs_joint_mu, p_z_obs_joint_covar = utils.get_joint_dist_params(jnp.exp(log_sigma_W), sd.W)

# Generate data from SCM
z_gt, interv_nodes, x_gt, P, interv_values = datagen.get_data(
                                                        rng_key,
                                                        opt, 
                                                        opt.n_interv_sets, 
                                                        sd, 
                                                        jnp.exp(log_sigma_W),
                                                        min_interv_value=opt.min_interv_value,
                                                        max_interv_value=opt.max_interv_value
                                                    )

p_z_mu = jnp.zeros((d))
p_z_covar = jnp.eye(d)

forward, model_params, model_opt_params, opt_model = init_vector_vae_params(opt, opt.proj_dims, key, rng_key, x_gt)

def get_elbo(model_params, rng_key, x_data):
    X_recons, z_pred, q_z_mus, z_L_chols = forward.apply(model_params, rng_key, d, opt.proj_dims, rng_key, x_data, opt.corr)
    mse_loss = jnp.mean(get_mse(x_data[:, :opt.proj_dims], X_recons))
    q_z_covars = vmap(get_covar, 0, 0)(z_L_chols)
    kl_loss = jnp.mean(vmap(get_single_kl, (None, None, 0, 0, None), 0)(p_z_covar, p_z_mu, q_z_covars, q_z_mus, opt))
    return (1e-5 * jnp.mean(mse_loss + kl_loss))

@jit
def gradient_step(model_params, x_data):
    loss, grads = value_and_grad(get_elbo, argnums=(0))(model_params, rng_key, x_data)
    X_recons, z_pred, q_z_mus, z_L_chols = forward.apply(model_params, rng_key, d, opt.proj_dims, rng_key, x_data, opt.corr)
    q_z_covars = vmap(get_covar, 0, 0)(z_L_chols)
    true_obs_KL_term_Z = vmap(get_single_kl, (None, None, 0, 0, None), 0)(p_z_obs_joint_covar, p_z_obs_joint_mu, q_z_covars, q_z_mus, opt)

    log_dict = {
        "z_mse": jnp.mean(get_mse(z_pred, z_gt)),
        "x_mse": jnp.mean(get_mse(x_data[:, :opt.proj_dims], X_recons)),
        "L_MSE": L_mse,
        "true_obs_KL_term_Z": jnp.mean(true_obs_KL_term_Z)
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
        X_recons, loss, grads, z_pred, log_dict = gradient_step(model_params, x_gt)
        try:
            mcc = get_cross_correlation(onp.array(z_pred), onp.array(z_gt))
        except:
            pass
        auroc = get_vae_auroc(d, gt_W)

        wandb_dict = {
            "ELBO": onp.array(loss),
            "Z_MSE": onp.array(log_dict["z_mse"]),
            "X_MSE": onp.array(log_dict["x_mse"]),
            "L_MSE": L_mse,
            "true_obs_KL_term_Z": onp.array(log_dict["true_obs_KL_term_Z"]),
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
            AUROC=f"{auroc:.2f}",
            KL_Z=f"{onp.array(log_dict['true_obs_KL_term_Z']):.4f}"
        )

        model_params, model_opt_params = update_params(grads, model_opt_params, model_params)

        
