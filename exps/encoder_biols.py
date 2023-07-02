from eval_encoder_biols import evaluate
import os, sys, pdb
import pathlib
import ruamel.yaml as yaml
from tqdm import tqdm
from collections import namedtuple, OrderedDict
import wandb

# Add to PATH
sys.path.append("..")
sys.path.append("../modules")

import numpy as onp
import jax
from jax import numpy as jnp
from jax import jit, lax, vmap, value_and_grad, config
from jax.flatten_util import ravel_pytree
import haiku as hk
import optax

import utils
from tensorflow_probability.substrates.jax.distributions import Horseshoe
from modules.GumbelSinkhorn import GumbelSinkhorn
from modules.encoder_biols_model_init import init_model
from datagen import SyntheticDatagen

config.update("jax_enable_x64", True)

# Load config yaml as options for experiment
configs = yaml.safe_load((pathlib.Path("..") / "configs.yaml").read_text())
opt = utils.load_yaml(configs)

# Set seeds
onp.random.seed(opt.data_seed)
rng_key = jax.random.PRNGKey(opt.data_seed)
hk_key = hk.PRNGSequence(opt.data_seed)

horseshoe_tau = utils.set_horseshoe_tau(opt.num_samples, opt.num_nodes, opt.exp_edges)
logdir = utils.set_logdir(opt)
os.makedirs(logdir, exist_ok=True)

Params = namedtuple('Params', ['model', 'LΣ'])
OptimizerParams = namedtuple('OptimizerParams', ['model', 'LΣ'])
Optimizers = namedtuple('Optimizers', ['model', 'LΣ'])

opt.num_samples = int(opt.pts_per_interv * opt.n_interv_sets) + opt.obs_data
l_dim = (opt.num_nodes * (opt.num_nodes - 1)) // 2

scm = SyntheticDatagen(
        data_seed=opt.data_seed,
        hk_key=hk_key,
        rng_key=rng_key,
        num_samples=opt.num_samples,
        num_obs_samples=opt.obs_data,
        num_nodes=opt.num_nodes,
        degree=opt.exp_edges,
        interv_sets=opt.n_interv_sets,
        interv_type=opt.interv_type,
        proj_dims=opt.proj_dims,
        projection=opt.proj,
        decoder_sigma=opt.decoder_sigma,
        interv_value_sampling=opt.interv_value_sampling,
        datagen_type=opt.datagen_type,
        eq_noise_var=opt.eq_noise_var,
        sem_type=opt.sem_type,
        graph_type=opt.graph_type,
        dataset_type='linear',
        min_interv_value=opt.min_interv_value
    )

x1, x2, z1, z2, labels, targets, interv_values = scm.sample_weakly_supervised(
    rng_key, 
    opt.obs_data, 
    opt.n_interv_sets, 
    return_interv_values=True
)
assert jnp.allclose(z2[targets == 1], interv_values[targets == 1])

x_samples = jnp.concatenate([x1, x2], axis=0)
z_samples = jnp.concatenate([z1, z2], axis=0)

interv_targets = jnp.concatenate([jnp.zeros(z1.shape).astype(int), targets], axis=0)
interv_labels = jnp.concatenate((jnp.ones_like(labels) * opt.num_nodes, labels), axis=0)
interv_values = jnp.concatenate([jnp.zeros(z1.shape), interv_values], axis=0)

gt_W, gt_P, gt_L = scm.W, scm.P, scm.P.T @ scm.W.T @ scm.P
gt_sigmas = jnp.exp(scm.log_sigma_W)
print(gt_W)

p_z_obs_joint_mu, p_z_obs_joint_covar = utils.get_joint_dist_params(jnp.exp(scm.log_sigma_W), scm.W)
LΣ_prior_dist = Horseshoe(scale=jnp.ones(l_dim + scm.noise_dim) * horseshoe_tau)
gumbel_sinkhorn = GumbelSinkhorn(opt.num_nodes, noise_type="gumbel", tol=opt.max_deviation)

if opt.off_wandb is False:
    if opt.offline_wandb is True: os.system('wandb offline')
    else:   os.system('wandb online')    
    wandb.init(project = opt.wandb_project, 
                entity = opt.wandb_entity, 
                config = vars(opt), 
                settings = wandb.Settings(start_method="fork"))
    wandb.run.name = f'BIOLS_s{opt.data_seed}_d{opt.num_nodes}_steps{opt.num_steps}_obs{opt.obs_data}_KL{opt.L_KL}'
    wandb.run.save()
    utils.save_graph_images(gt_P, gt_L, gt_W, 'gt_P.png', 'gt_L.png', 'gt_w.png', logdir)
    wandb.log({   
                "graph_structure(GT-pred)/Ground truth W": wandb.Image(join(logdir, 'gt_w.png')),
                "graph_structure(GT-pred)/Ground truth P": wandb.Image(join(logdir, 'gt_P.png')), 
                "graph_structure(GT-pred)/Ground truth L": wandb.Image(join(logdir, 'gt_L.png')), 
            }, step=0)

forward, params, optimizers, optimizer_params  = init_model(
    hk_key,  
    rng_key, 
    x_samples,
    opt, 
    interv_labels, 
    interv_values, 
    l_dim, 
    scm.noise_dim, 
    P=gt_P
)

batch_get_obs_joint_dist_params = vmap(utils.get_joint_dist_params, (0, 0), (0, 0))
vmapped_kl = vmap(utils.get_single_kl, (None, None, 0, 0, None), (0))
print()

@jit
def calc_neg_elbo(params, rng_key, interv_labels, interv_values, gt_z_data, gt_x_data):
    """
        Compute the neg. ELBO that has to be minimized:
            - exp_{P, L, Σ} [ exp_{Z | P, L, Σ} log p(X | Z) ] 
            + exp_{P | L, Σ} KL(q(P | L, Σ) || P(P))
            + exp_{L, Σ} KL(q(L, Σ) || P(L)p(Σ))
    """
    
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
                                    ) = forward.apply(params.model, 
                                                        rng_key, 
                                                        rng_key, 
                                                        True, 
                                                        gt_x_data,
                                                        opt, 
                                                        interv_labels, 
                                                        interv_values, 
                                                        params.LΣ, 
                                                        P=gt_P)

        if opt.learn_Z:
            # ! - exp_{P, L, Σ} exp_{Z | P, L, Σ} log p(X | Z)
            nll = jit(vmap(utils.get_mse, (None, 0), 0))(gt_x_data, pred_X)
        
        else: # ! not learning Z
            nll = jit(vmap(utils.get_mse, (None, 0), 0))(gt_z_data, batched_qz_samples)

        if opt.L_KL and opt.learn_L:    # ! KL over edge weights L and Σ
            LΣ_prior_probs = jnp.sum(LΣ_prior_dist.log_prob(full_l_batch + 1e-7)[:, :l_dim], axis=1)
            KL_term_LΣ = full_log_prob_l - LΣ_prior_probs
        
        if opt.P_KL and opt.learn_P:    # ! KL over permutation P
            logprob_P = vmap(gumbel_sinkhorn.logprob, in_axes=(0, 0, None))(batch_P, batch_P_logits, num_bethe_iters)
            log_P_prior = -jnp.sum(jnp.log(onp.arange(d) + 1))
            KL_term_P = logprob_P - log_P_prior

        neg_elbo = jnp.mean(nll + KL_term_LΣ + KL_term_P)
        return neg_elbo
    
    rng_keys = jax.random.split(rng_key, 1)
    _, neg_elbos = lax.scan(lambda _, rng_key: (None, get_outer_expectation(rng_key)), None, rng_keys)
    return jnp.mean(neg_elbos)


@jit
def gradient_step(rng_key, params, interv_labels, interv_values, gt_z_data, gt_x_data):
    get_loss_and_grad = value_and_grad(calc_neg_elbo, argnums=(0))
    loss, grads = get_loss_and_grad(
        params,
        rng_key, 
        interv_labels,  
        interv_values,
        gt_z_data,
        gt_x_data, 
    )

    (
        pred_X,
        P_samples,
        _,
        pred_z_samples,
        full_l_batch,
        full_log_prob_l,
        L_samples,
        W_samples,
        log_noise_std_samples ) = forward.apply(
                                    params.model, 
                                    rng_key, 
                                    rng_key, 
                                    True, 
                                    gt_x_data,
                                    opt, 
                                    interv_labels, 
                                    interv_values, 
                                    params.LΣ, 
                                    P=gt_P
                                )
    
    if opt.eq_noise_var:
        log_noise_std_samples = jnp.tile(log_noise_std_samples, (1, opt.num_nodes))
    
    batch_q_z_obs_joint_mus, batch_q_z_obs_joint_covars = batch_get_obs_joint_dist_params(jnp.exp(log_noise_std_samples), W_samples)
    true_obs_KL_term_Z = vmapped_kl(
        p_z_obs_joint_covar, 
        p_z_obs_joint_mu, 
        batch_q_z_obs_joint_covars, 
        batch_q_z_obs_joint_mus, 
        opt
    )

    pred_W_means = jnp.mean(W_samples, axis=0)
    LΣ_prior_probs = jnp.sum(LΣ_prior_dist.log_prob(full_l_batch + 1e-7)[:, :l_dim], axis=1)
    KL_term_LΣ = full_log_prob_l - LΣ_prior_probs

    G_samples = jnp.where(jnp.abs(W_samples) >= opt.edge_threshold, 1, 0) 

    log_dict = {
        "L_mse": jnp.mean(vmap(utils.get_mse, (None, 0), 0)(gt_W, jnp.multiply(W_samples, G_samples))),
        "z_mse": jnp.mean(vmap(utils.get_mse, (None, 0), 0)(gt_z_data, pred_z_samples)),
        "x_mse": jnp.mean(vmap(utils.get_mse, (None, 0), 0)(gt_x_data, pred_X)),
        "true_obs_KL_term_Z": jnp.mean(true_obs_KL_term_Z),
        "KL(L)": jnp.mean(KL_term_LΣ)
    }

    model_outputs = (P_samples, L_samples, W_samples, pred_z_samples, pred_X, pred_W_means)
    return loss, grads, log_dict, model_outputs


@jit
def update_params(grads, params, optimizer_params):
    """
        Given all parameters and gradients, update all parameters and optimizer states.
    """
    model_updates, model_opt_params = optimizers.model.update(grads.model, optimizer_params.model, params.model)   
    new_model_params = optax.apply_updates(params.model, model_updates)

    LΣ_updates, LΣ_opt_params = optimizers.LΣ.update(grads.LΣ, optimizer_params.LΣ, params.LΣ)
    new_LΣ_params = optax.apply_updates(params.LΣ, LΣ_updates)

    params = Params(
        model=new_model_params, 
        LΣ=new_LΣ_params
    )
    optimizer_params = OptimizerParams(
        model=model_opt_params,
        LΣ=LΣ_opt_params
    )
    return params, optimizer_params


with tqdm(range(opt.num_steps)) as pbar:
    for i in pbar:
        rng_key = jax.random.split(rng_key, 1)[0]
        loss, grads, log_dict, model_outputs = gradient_step(rng_key, params, interv_labels, interv_values, z_samples, x_samples)
        params, optimizer_params = update_params(grads, params, optimizer_params)

        _, _, _, pred_z_samples, _, _ = model_outputs
        
        if (i+1) % 500 == 0 or i == 0:  
            wandb_dict, eval_dict = evaluate(
                rng_key, 
                forward,
                x_samples,
                loss,
                log_dict,
                opt, 
                params, 
                interv_labels, 
                interv_values, 
                pred_z_samples,
                z_samples,
                gt_W,
                gt_sigmas,
                gt_P
            )

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

        if jnp.any(jnp.isnan(ravel_pytree(params.LΣ)[0])):   
            raise Exception("Got NaNs in L params")



            