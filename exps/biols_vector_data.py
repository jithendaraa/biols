import os, sys, pdb
from eval import evaluate
sys.path.append("..")
sys.path.append("../modules")

from os.path import join
import pathlib
import ruamel.yaml as yaml
from tqdm import tqdm
from collections import OrderedDict, namedtuple

import wandb
import optax

import numpy as onp
import jax.numpy as jnp
import jax
from jax import jit, vmap, value_and_grad, config
import haiku as hk
from tensorflow_probability.substrates.jax.distributions import Horseshoe
from jax.tree_util import tree_map
from jax.flatten_util import ravel_pytree

from modules.biols_model_init import init_model, Parameters, OptimizerState
from modules.GumbelSinkhorn import GumbelSinkhorn

import utils

config.update("jax_enable_x64", True)
Gradients = namedtuple('Gradients', ['model', 'LΣ'])
Losses = namedtuple('Losses', ['nll', 'LΣ_posterior_logprobs', 'LΣ_prior_logprobs', 'L_prior_logprobs', 'Σ_prior_logprobs', 'KL_term_LΣ', 'KL_term_P', 'neg_elbo'])

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
l_dim = opt.num_nodes * (opt.num_nodes - 1) // 2
horseshoe_tau = utils.set_horseshoe_tau(opt.num_samples, opt.num_nodes, opt.exp_edges)
logdir = utils.set_logdir(opt)
noise_dim = opt.num_nodes
os.makedirs(logdir, exist_ok=True)
print(f"Learning permutation: {opt.learn_P}")

# Load saved data
gt_samples, interventions = utils.read_biols_dataset(folder_path)
gt_W = onp.load(f'{folder_path}/weighted_adjacency.npy')
gt_P = onp.load(f'{folder_path}/perm.npy')
gt_L = onp.load(f'{folder_path}/edge_weights.npy')
gt_sigmas = onp.load(f'{folder_path}/gt_sigmas.npy')
print(gt_W)

# assert opt.learn_Z is True
LΣ_prior_dist = Horseshoe(scale=jnp.ones(l_dim + noise_dim) * horseshoe_tau)
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

def calc_neg_elbo(rng_key, params, interventions, gt_samples):
    """
        Compute the neg. ELBO that has to be minimized:
            - exp_{P, L, Σ} [ exp_{Z | P, L, Σ} log p(X | Z) ] 
            + exp_{P | L, Σ} KL(q(P | L, Σ) || P(P))
            + exp_{L, Σ} KL(q(L, Σ) || P(L)p(Σ))
    """
    KL_term_LΣ, KL_term_P = 0., 0.

    (   
        pred_samples,
        batch_P_logits,
        LΣ_samples,
        LΣ_posterior_logprobs,
        log_noise_std_samples
                                ) = forward.apply(params.model, 
                                                    rng_key, 
                                                    hard, 
                                                    rng_key, 
                                                    opt, 
                                                    interventions, 
                                                    params.LΣ, 
                                                    P=gt_P)
    
    # - exp_{P, L, Σ} exp_{Z | P, L, Σ} log p(X | Z)
    vmapped_nll_gaussian = vmap(utils.nll_gaussian, (None, 0, None), 0)
    nll = vmapped_nll_gaussian(gt_samples.z, pred_samples.z, pred_sigma)
    
    L_prior_logprobs = jnp.sum(LΣ_prior_dist.log_prob(LΣ_samples)[:, :l_dim], axis=1)
    Σ_prior_logprobs = jnp.sum(LΣ_samples[:, l_dim:] ** 2 / (2 * opt.s_prior_std ** 2), axis=-1)
    LΣ_prior_logprobs = L_prior_logprobs + Σ_prior_logprobs
    KL_term_LΣ = LΣ_posterior_logprobs - LΣ_prior_logprobs
    
    if opt.learn_P:
        logprob_P = vmap(gumbel_sinkhorn.logprob, in_axes=(0, 0, None))(pred_samples.P, batch_P_logits.astype(jnp.float64), opt.bethe_iters)
        log_P_prior = -jnp.sum(jnp.log(onp.arange(opt.num_nodes) + 1))
        KL_term_P = logprob_P - log_P_prior

    neg_elbo = jnp.mean(nll + KL_term_LΣ + KL_term_P)
    losses = Losses(
        nll=nll, 
        LΣ_posterior_logprobs=LΣ_posterior_logprobs,
        LΣ_prior_logprobs=LΣ_prior_logprobs,
        L_prior_logprobs=L_prior_logprobs,
        Σ_prior_logprobs=Σ_prior_logprobs,
        KL_term_LΣ=KL_term_LΣ, 
        KL_term_P=KL_term_P, 
        neg_elbo=neg_elbo
    )

    aux_res = (losses, pred_samples, LΣ_samples, log_noise_std_samples)
    return neg_elbo, aux_res


@jit
def gradient_step(rng_key, params, interventions, gt_samples):
    """
        1. Compute negative ELBO to be minimized
        2. Obtain gradients with respect to LΣ and the model 
            (P and decoder parameters)
        3. Based on output from forward pass, compute metrics in a dict
    """
    get_loss_and_grad = value_and_grad(calc_neg_elbo, argnums=(1), has_aux=True)
    (loss, aux_res), grads = get_loss_and_grad(rng_key, params, interventions, gt_samples)
    gradients = Gradients(
        model=grads.model, 
        LΣ=grads.LΣ
    )
    (losses, pred_samples, LΣ_samples, log_noise_std_samples) = aux_res
                    
    pred_W_means = jnp.mean(pred_samples.W, axis=0)
    G_samples = jnp.where(jnp.abs(pred_samples.W) >= opt.edge_threshold, 1, 0) 

    log_dict = {
        "L_mse": jnp.mean(vmap(utils.get_mse, (None, 0), 0)(gt_W, jnp.multiply(pred_samples.W, G_samples))),
        "z_mse": jnp.mean(vmap(utils.get_mse, (None, 0), 0)(gt_samples.z, pred_samples.z)),
        "x_mse": jnp.mean(vmap(utils.get_mse, (None, 0), 0)(gt_samples.x, pred_samples.x)),
        "KL(LΣ)": jnp.mean(losses.KL_term_LΣ),
        "KL(P)": jnp.mean(losses.KL_term_P),
    }

    return (loss, losses, gradients, pred_samples, LΣ_samples, log_noise_std_samples, pred_W_means, log_dict)


@jit
def update_params(params, gradients, opt_state):
    """
        Given current parameters, gradients and optimizer states, return updated parameters and optimizer states.
    """
    LΣ_updates, LΣ_opt_state = optimizers.LΣ.update(gradients.LΣ, opt_state.LΣ, params.LΣ)
    new_LΣ_params = optax.apply_updates(params.LΣ, LΣ_updates)

    model_updates, new_model_opt_state = optimizers.model.update(gradients.model, opt_state.model, params.model)
    new_model_params = optax.apply_updates(params.model, model_updates)

    params = Parameters(new_LΣ_params, new_model_params)
    opt_state = OptimizerState(LΣ_opt_state, new_model_opt_state)
    return params, opt_state


forward, params, optimizers, opt_state  = init_model(
    hk_key, 
    rng_key, 
    opt, 
    interventions, 
    l_dim, 
    noise_dim, 
    P=gt_P
)

# Training loop
with tqdm(range(opt.num_steps)) as pbar:
    for i in pbar:
        
        loss, losses, gradients, pred_samples, LΣ_samples, log_noise_std_samples, pred_W_means, log_dict = gradient_step(
            rng_key, params, interventions, gt_samples
        )

        params, opt_state = update_params(params, gradients, opt_state)

        if jnp.any(jnp.isnan(ravel_pytree(gradients.model)[0])):
            pdb.set_trace()
            raise Exception("Got NaNs in model_grads")
        
        if jnp.any(jnp.isnan(ravel_pytree(gradients.LΣ)[0])):   
            neg_elbo, aux_res = calc_neg_elbo(rng_key, params, interventions, gt_samples)
            losses_, _, LΣ_samples_, _ = aux_res
            pdb.set_trace()
            raise Exception("Got NaNs in LΣ_grads")

        if (i+1) % 500 == 0 or i == 0:
            wandb_dict, eval_dict = evaluate(
                rng_key, 
                params, 
                forward, 
                interventions, 
                pred_samples, 
                gt_samples, 
                gt_W, 
                gt_sigmas, 
                gt_P, 
                gt_L, 
                loss, 
                log_dict, 
                opt
            )

            if opt.off_wandb is False:  
                utils.save_graph_images(pred_samples.P[0], pred_samples.L[0], pred_W_means, 'sample_P.png', 'sample_L.png', 'pred_w.png', logdir)
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
            KL_LΣ=f"{log_dict['KL(LΣ)']:.3f}",
            KL_P=f"{log_dict['KL(P)']:.3f}",
        )

        pbar.set_postfix(postfix_dict)