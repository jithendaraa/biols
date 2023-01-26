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

from modules.biols_model_init import init_model
from modules.SyntheticSCM import SyntheticSCM
from modules.GumbelSinkhorn import GumbelSinkhorn

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

LΣ_prior_dist = Horseshoe(scale=jnp.ones(l_dim + noise_dim) * horseshoe_tau)
gumbel_sinkhorn = GumbelSinkhorn(d, noise_type="gumbel", tol=opt.max_deviation)

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
print(gt_W)

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
z_gt, interv_nodes, x_gt, P, interv_targets, interv_values = datagen.get_data(
                                                        rng_key,
                                                        opt, 
                                                        opt.n_interv_sets, 
                                                        sd, 
                                                        jnp.exp(log_sigma_W),
                                                        min_interv_value=opt.min_interv_value,
                                                        max_interv_value=opt.max_interv_value
                                                    )

def calc_neg_elbo(model_params, LΣ_params, rng_key, interv_nodes, gt_x_data, interv_values):
    """
        Compute the neg. ELBO that has to be minimized:
            - exp_{P, L, Σ} [ exp_{Z | P, L, Σ} log p(X | Z) ] 
            + exp_{P | L, Σ} KL(q(P | L, Σ) || P(P))
            + exp_{L, Σ} KL(q(L, Σ) || P(L)p(Σ))
    """
    rng_keys = jax.random.split(rng_key, 1)
    
    def get_outer_expectation(rng_key):
        
        KL_term_LΣ, KL_term_P = 0., 0.
        (   
            pred_X,
            batch_P,
            batch_P_logits,
            batched_qz_samples,
            full_l_batch, 
            full_log_prob_l,
            L_samples, 
            W_samples, 
            log_noise_std_samples
                                    ) = forward.apply(model_params, 
                                                        rng_key, 
                                                        hard, 
                                                        rng_key, 
                                                        opt, 
                                                        interv_nodes, 
                                                        interv_values, 
                                                        LΣ_params, 
                                                        P=gt_P)

        # - exp_{P, L, Σ} exp_{Z | P, L, Σ} log p(X | Z)
        nll = jit(vmap(utils.get_mse, (None, 0), 0))(gt_x_data, pred_X)

        if opt.L_KL and opt.learn_L:    # ! KL over edge weights L and Σ
            LΣ_prior_probs = jnp.sum(LΣ_prior_dist.log_prob(full_l_batch + 1e-7)[:, :l_dim], axis=1)
            KL_term_LΣ = full_log_prob_l - LΣ_prior_probs
        
        if opt.P_KL and opt.learn_P:    # ! KL over permutation P
            logprob_P = vmap(gumbel_sinkhorn.logprob, in_axes=(0, 0, None))(batch_P, batch_P_logits, num_bethe_iters)
            log_P_prior = -jnp.sum(jnp.log(onp.arange(d) + 1))
            KL_term_P = logprob_P - log_P_prior

        neg_elbo = jnp.mean(nll + KL_term_LΣ + KL_term_P)
        return neg_elbo

    _, neg_elbos = lax.scan(lambda _, rng_key: (None, get_outer_expectation(rng_key)), None, rng_keys)
    return jnp.mean(neg_elbos)

@jit
def gradient_step(model_params, LΣ_params, rng_key, interv_nodes, gt_z_data, gt_x_data, interv_values):
    """
        1. Compute negative ELBO to be minimized
        2. Obtain gradients with respect to LΣ and the model 
            (P and decoder parameters)
        3. Based on output from forward pass, compute metrics in a dict
    """
    get_loss_and_grad = value_and_grad(calc_neg_elbo, argnums=(0, 1))
    loss, grads = get_loss_and_grad(model_params, 
                                    LΣ_params,
                                    rng_key, 
                                    interv_nodes,  
                                    gt_x_data, 
                                    interv_values)
                    
    model_grads, LΣ_grads = tree_map(lambda x_: x_, grads)
    rng_key_ = jax.random.split(rng_key, 1)[0]
    
    (   
        pred_X,
        _,
        _,
        z_samples,
        full_l_batch, 
        full_log_prob_l,
        L_samples, 
        W_samples, 
        log_noise_std_samples
                                ) = forward.apply(model_params, 
                                                    rng_key_, 
                                                    hard, 
                                                    rng_key_, 
                                                    opt, 
                                                    interv_nodes, 
                                                    interv_values, 
                                                    LΣ_params, 
                                                    P=gt_P)

    # Obtain KL between predicted and GT observational joint distribution
    batch_get_obs_joint_dist_params = vmap(utils.get_joint_dist_params, (0, 0), (0, 0))
    if opt.eq_noise_var:
        log_noise_std_samples = jnp.tile(log_noise_std_samples, (1, d))
    batch_q_z_obs_joint_mus, batch_q_z_obs_joint_covars = batch_get_obs_joint_dist_params(jnp.exp(log_noise_std_samples), W_samples)
    vmapped_kl = vmap(utils.get_single_kl, (None, None, 0, 0, None), (0))
    true_obs_KL_term_Z = vmapped_kl(p_z_obs_joint_covar, 
                                    p_z_obs_joint_mu, 
                                    batch_q_z_obs_joint_covars, 
                                    batch_q_z_obs_joint_mus, 
                                    opt)

    pred_W_means = jnp.mean(W_samples, axis=0)
    LΣ_prior_probs = jnp.sum(LΣ_prior_dist.log_prob(full_l_batch + 1e-7)[:, :l_dim], axis=1)
    KL_term_LΣ = full_log_prob_l - LΣ_prior_probs

    G_samples = jnp.where(jnp.abs(W_samples) >= 0.3, 1, 0) 

    log_dict = {
        "L_mse": jnp.mean(vmap(utils.get_mse, (None, 0), 0)(gt_W, jnp.multiply(W_samples, G_samples))),
        "z_mse": jnp.mean(vmap(utils.get_mse, (None, 0), 0)(gt_z_data, z_samples)),
        "x_mse": jnp.mean(vmap(utils.get_mse, (None, 0), 0)(gt_x_data, pred_X)),
        "true_obs_KL_term_Z": jnp.mean(true_obs_KL_term_Z),
        "KL(L)": jnp.mean(KL_term_LΣ)
    }

    return (loss, rng_key, model_grads, LΣ_grads, log_dict, W_samples, z_samples, pred_X, pred_W_means)

@jit
def update_params(model_grads, model_opt_params, model_params, LΣ_grads, LΣ_opt_params, LΣ_params):
    """
        Given all parameters and gradients, update all parameters and optimizer states.
    """
    model_updates, model_opt_params = opt_model.update(model_grads, model_opt_params, model_params)
    model_params = optax.apply_updates(model_params, model_updates)
    LΣ_updates, LΣ_opt_params = opt_LΣ.update(LΣ_grads, LΣ_opt_params, LΣ_params)
    LΣ_params = optax.apply_updates(LΣ_params, LΣ_updates)
    return model_params, model_opt_params, LΣ_params, LΣ_opt_params

(   
    forward, 
    model_params, 
    LΣ_params, 
    model_opt_params, 
    LΣ_opt_params, 
    opt_model, 
    opt_LΣ  
    ) = init_model(key, 
                    False, 
                    rng_key, 
                    opt, 
                    interv_nodes, 
                    interv_values, 
                    l_dim, 
                    noise_dim, 
                    P=gt_P)

# Training loop
with tqdm(range(opt.num_steps)) as pbar:
    for i in pbar:
        
        (   loss, 
            rng_key, 
            model_grads, 
            LΣ_grads, 
            log_dict, 
            batch_W, 
            z_samples, 
            X_recons, 
            pred_W_means    ) = gradient_step(model_params, LΣ_params, rng_key, interv_nodes, z_gt, x_gt, interv_values)
        
        if (i+1) % 500 == 0 or i == 0:  
            wandb_dict, eval_dict = evaluate(
                                            rng_key, 
                                            model_params, 
                                            LΣ_params, 
                                            forward, 
                                            interv_nodes, 
                                            interv_values, 
                                            z_samples, 
                                            z_gt, 
                                            gt_W,
                                            gt_sigmas,
                                            gt_P,
                                            gt_L,
                                            loss,
                                            log_dict,
                                            opt
                                        )

            if opt.off_wandb is False:  
                plt.imshow(pred_W_means)
                plt.savefig(join(logdir, 'pred_w.png'))
                wandb_dict["graph_structure(GT-pred)/Predicted W"] = wandb.Image(join(logdir, 'pred_w.png'))
                wandb.log(wandb_dict, step=i)

            shd = eval_dict["shd"]
            tqdm.write(f"Step {i} | {loss}")
            tqdm.write(f"Z_MSE: {log_dict['z_mse']} | X_MSE: {log_dict['x_mse']}")
            tqdm.write(f"L MSE: {log_dict['L_mse']}")
            tqdm.write(f"SHD: {eval_dict['shd']} | CPDAG SHD: {eval_dict['shd_c']} | AUROC: {eval_dict['auroc']}")
            tqdm.write(f"KL(learned || true): {onp.array(log_dict['true_obs_KL_term_Z'])}")
            tqdm.write(f" ")

        postfix_dict = OrderedDict(
            KL=f"{log_dict['true_obs_KL_term_Z']:.4f}", 
            L_mse=f"{log_dict['L_mse']:.3f}",
            SHD=shd,
            AUROC=f"{eval_dict['auroc']:.2f}",
            loss=f"{loss:.4f}",
        )
        pbar.set_postfix(postfix_dict)

        model_params, model_opt_params, LΣ_params, LΣ_opt_params = update_params(model_grads, 
                                                                                model_opt_params, 
                                                                                model_params, 
                                                                                LΣ_grads, 
                                                                                LΣ_opt_params, 
                                                                                LΣ_params)

        if jnp.any(jnp.isnan(ravel_pytree(LΣ_params)[0])):   raise Exception("Got NaNs in L params")