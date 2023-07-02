import os, sys, pdb
from eval import evaluate
sys.path.append("..")
sys.path.append("../modules")

from os.path import join
import pathlib
import ruamel.yaml as yaml
from tqdm import tqdm
from collections import OrderedDict

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
from modules.GumbelSinkhorn import GumbelSinkhorn

import utils

config.update("jax_enable_x64", True)
# Load config yaml as options for experiment
configs = yaml.safe_load((pathlib.Path("..") / "biols_config.yaml").read_text())
opt, folder_path = utils.load_yaml(configs)
opt.obs_data = opt.n_pairs

# Set seeds
onp.random.seed(opt.data_seed)
rng_key = jax.random.PRNGKey(opt.data_seed)
hk_key = hk.PRNGSequence(opt.data_seed)
assert opt.max_interv_value == -opt.min_interv_value

# Set some constants
hard = True
s_prior_std = 3.0
l_dim = opt.num_nodes * (opt.num_nodes - 1) // 2
horseshoe_tau = utils.set_horseshoe_tau(opt.num_samples, opt.num_nodes, opt.exp_edges)
logdir = utils.set_logdir(opt)
noise_dim = opt.num_nodes
os.makedirs(logdir, exist_ok=True)
print(opt.learn_P)

# Instantiate random SCM with structure and parameters
# scm = SyntheticDatagen(
#     data_seed=opt.data_seed,
#     hk_key=hk_key,
#     rng_key=rng_key,
#     num_nodes=opt.num_nodes,
#     degree=opt.exp_edges,
#     interv_type=opt.interv_type,
#     proj_dims=opt.proj_dims,
#     projection=opt.proj,
#     decoder_sigma=opt.decoder_sigma,
#     interv_value_sampling=opt.interv_value_sampling,
#     datagen_type=opt.datagen_type,
#     sem_type=opt.sem_type,
#     graph_type=opt.graph_type,
#     dataset_type='linear',
#     min_interv_value=opt.min_interv_value,
# )

# x1, x2, z1, z2, labels, interv_targets, interv_values, interv_noise = scm.sample_weakly_supervised(
#     rng_key, 
#     opt.n_pairs, 
#     opt.n_interv_sets, 
#     return_interv_values=True, 
#     fix_noise=opt.fix_noise, 
#     no_interv_noise=opt.no_interv_noise,
#     return_interv_noise=True
# )

# x_samples = jnp.concatenate([x1, x2], axis=0)
# z_samples = jnp.concatenate([z1, z2], axis=0)
# interv_labels = jnp.concatenate((jnp.ones_like(labels) * opt.num_nodes, labels), axis=0)
# interv_targets = jnp.concatenate([jnp.zeros(z1.shape).astype(int), interv_targets], axis=0)
# interv_values = jnp.concatenate([jnp.zeros(z1.shape), interv_values], axis=0)
# interv_noise = jnp.concatenate([jnp.zeros(z1.shape).astype(int), interv_noise], axis=0)

# gt_W, gt_P, gt_L = scm.W, scm.P, scm.P.T @ scm.W.T @ scm.P
# gt_sigmas = jnp.exp(scm.log_sigma_W)

# Load saved data
x_samples, z_samples, interv_labels, interv_targets, interv_values, interv_noise = utils.read_biols_dataset(folder_path)
gt_W = onp.load(f'{folder_path}/weighted_adjacency.npy')
gt_P = onp.load(f'{folder_path}/perm.npy')
gt_L = onp.load(f'{folder_path}/edge_weights.npy')
gt_sigmas = onp.load(f'{folder_path}/gt_sigmas.npy')
print(gt_W)

# assert opt.learn_Z is True
LΣ_prior_dist = Horseshoe(scale=jnp.ones(l_dim + noise_dim) * 2)
gumbel_sinkhorn = GumbelSinkhorn(opt.num_nodes, noise_type="gumbel", tol=opt.max_deviation)

if opt.off_wandb is False:  
    wandb.init(project = opt.wandb_project, 
                entity = opt.wandb_entity, 
                config = vars(opt), 
                settings = wandb.Settings(start_method="fork"))
    wandb.run.name = f'{opt.exp_name}_{opt.data_seed}'
    wandb.run.save()
    utils.save_graph_images(gt_P, gt_L, gt_W, 'gt_P.png', 'gt_L.png', 'gt_w.png', logdir)
    wandb.log({   
                "graph_structure(GT-pred)/Ground truth W": wandb.Image(join(logdir, 'gt_w.png')),
                "graph_structure(GT-pred)/Ground truth P": wandb.Image(join(logdir, 'gt_P.png')), 
                "graph_structure(GT-pred)/Ground truth L": wandb.Image(join(logdir, 'gt_L.png')), 
            }, step=0)


pred_sigma = 1.0

def calc_neg_elbo(model_params, LΣ_params, rng_key, interv_labels, gt_x_data, interv_values):
    """
        Compute the neg. ELBO that has to be minimized:
            - exp_{P, L, Σ} [ exp_{Z | P, L, Σ} log p(X | Z) ] 
            + exp_{P | L, Σ} KL(q(P | L, Σ) || P(P))
            + exp_{L, Σ} KL(q(L, Σ) || P(L)p(Σ))
    """
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
                                                    interv_labels, 
                                                    interv_values, 
                                                    LΣ_params, 
                                                    P=gt_P)
    
    # - exp_{P, L, Σ} exp_{Z | P, L, Σ} log p(X | Z)
    vmapped_nll_gaussian = vmap(utils.nll_gaussian, (None, 0, None), 0)
    nll = vmapped_nll_gaussian(z_samples, batched_qz_samples, pred_sigma)
    
    L_prior_probs = jnp.sum(LΣ_prior_dist.log_prob(full_l_batch)[:, :l_dim], axis=1)
    Σ_prior_probs = jnp.sum(full_l_batch[:, l_dim:] ** 2 / (2 * s_prior_std ** 2), axis=-1)
    LΣ_prior_probs = L_prior_probs + Σ_prior_probs
    KL_term_LΣ = full_log_prob_l - LΣ_prior_probs
    
    if opt.learn_P:
        logprob_P = vmap(gumbel_sinkhorn.logprob, in_axes=(0, 0, None))(batch_P, batch_P_logits.astype(jnp.float64), opt.bethe_iters)
        log_P_prior = -jnp.sum(jnp.log(onp.arange(opt.num_nodes) + 1))
        KL_term_P = logprob_P - log_P_prior

    neg_elbo = jnp.mean(nll + KL_term_LΣ + KL_term_P)
    aux_res = (pred_X, batch_P, batched_qz_samples, full_l_batch, full_log_prob_l, L_samples, W_samples, log_noise_std_samples)
    return neg_elbo, aux_res


@jit
def gradient_step(model_params, LΣ_params, rng_key, interv_labels, gt_z_data, gt_x_data, interv_values):
    """
        1. Compute negative ELBO to be minimized
        2. Obtain gradients with respect to LΣ and the model 
            (P and decoder parameters)
        3. Based on output from forward pass, compute metrics in a dict
    """
    get_loss_and_grad = value_and_grad(calc_neg_elbo, argnums=(0, 1), has_aux=True)
    (loss, aux_res), grads = get_loss_and_grad(model_params, LΣ_params, rng_key, interv_labels, gt_x_data, interv_values)
    pred_X, P_samples, pred_z_samples, full_l_batch, full_log_prob_l, L_samples, W_samples, log_noise_std_samples = aux_res
                    
    model_grads, LΣ_grads = tree_map(lambda x_: x_, grads)
    rng_key_ = jax.random.split(rng_key, 1)[0]

    pred_W_means = jnp.mean(W_samples, axis=0)
    L_prior_probs = jnp.sum(LΣ_prior_dist.log_prob(full_l_batch)[:, :l_dim], axis=1)
    Σ_prior_probs = jnp.sum(full_l_batch[:, l_dim:] ** 2 / (2 * s_prior_std ** 2), axis=-1)
    LΣ_prior_probs = L_prior_probs + Σ_prior_probs
    KL_term_LΣ = full_log_prob_l - LΣ_prior_probs

    G_samples = jnp.where(jnp.abs(W_samples) >= opt.edge_threshold, 1, 0) 

    log_dict = {
        "L_mse": jnp.mean(vmap(utils.get_mse, (None, 0), 0)(gt_W, jnp.multiply(W_samples, G_samples))),
        "z_mse": jnp.mean(vmap(utils.get_mse, (None, 0), 0)(gt_z_data, pred_z_samples)),
        "x_mse": jnp.mean(vmap(utils.get_mse, (None, 0), 0)(gt_x_data, pred_X)),
        "KL(L)": jnp.mean(KL_term_LΣ)
    }

    return (loss, rng_key_, P_samples, L_samples, model_grads, LΣ_grads, log_dict, W_samples, pred_z_samples, pred_X, pred_W_means)


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


forward, model_params, LΣ_params, model_opt_params, LΣ_opt_params, opt_model, opt_LΣ  = init_model(
    hk_key, 
    False, 
    rng_key, 
    opt, 
    interv_labels, 
    interv_values, 
    l_dim, 
    noise_dim, 
    P=gt_P
)

# Training loop
with tqdm(range(opt.num_steps)) as pbar:
    for i in pbar:
        loss, rng_key, P_samples, L_samples, model_grads, LΣ_grads, log_dict, batch_W, pred_z_samples, _, pred_W_means = gradient_step(
            model_params, 
            LΣ_params, 
            rng_key, 
            interv_labels, 
            z_samples, 
            x_samples, 
            interv_values
        )

        if (i+1) % 500 == 0 or i == 0:  
            wandb_dict, eval_dict = evaluate(rng_key, model_params, LΣ_params, forward, interv_labels, interv_values, 
                pred_z_samples, z_samples, gt_W, gt_sigmas, gt_P, gt_L, loss, log_dict, opt)

            if opt.off_wandb is False:  
                utils.save_graph_images(P_samples[0], L_samples[0], pred_W_means, 'sample_P.png', 'sample_L.png', 'pred_w.png', logdir)

                wandb_dict["graph_structure(GT-pred)/Predicted W"] = wandb.Image(join(logdir, 'pred_w.png'))
                wandb_dict["graph_structure(GT-pred)/Sample P"] = wandb.Image(join(logdir, 'sample_P.png'))
                wandb_dict["graph_structure(GT-pred)/Sample L"] = wandb.Image(join(logdir, 'sample_L.png'))
                wandb.log(wandb_dict, step=i)

            shd = eval_dict["shd"]
            tqdm.write(f"Step {i} | {loss}")
            tqdm.write(f"Z_MSE: {log_dict['z_mse']} | X_MSE: {log_dict['x_mse']}")
            tqdm.write(f"L MSE: {log_dict['L_mse']}")
            tqdm.write(f"SHD: {eval_dict['shd']} | CPDAG SHD: {eval_dict['shd_c']} | AUROC: {eval_dict['auroc']}")
            tqdm.write(f" ")

        postfix_dict = OrderedDict(
            L_mse=f"{log_dict['L_mse']:.3f}",
            SHD=shd,
            AUROC=f"{eval_dict['auroc']:.2f}",
            loss=f"{loss:.4f}",
            KL_L=f"{log_dict['KL(L)']:.4f}"
        )

        pbar.set_postfix(postfix_dict)

        model_params, model_opt_params, LΣ_params, LΣ_opt_params = update_params(model_grads, model_opt_params, model_params, LΣ_grads, LΣ_opt_params, LΣ_params)
        if jnp.any(jnp.isnan(ravel_pytree(LΣ_params)[0])) or jnp.any(jnp.isnan(ravel_pytree(model_params)[0])):   
            raise Exception("Got NaNs in L params")