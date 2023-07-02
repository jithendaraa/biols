import os, sys, pdb
sys.path.append("..")
sys.path.append("../modules")

from os.path import join
import pathlib
import ruamel.yaml as yaml
from tqdm import tqdm
from matplotlib import pyplot as plt
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

from modules.biols_model_init import init_interv_model
from modules.GumbelSinkhorn import GumbelSinkhorn

import utils
from datagen import SyntheticDatagen
from eval import evaluate

config.update("jax_enable_x64", True)

# Load config yaml as options for experiment
configs = yaml.safe_load((pathlib.Path("..") / "configs.yaml").read_text())
opt = utils.load_yaml(configs)

# Set seeds
onp.random.seed(opt.data_seed)
rng_key = jax.random.PRNGKey(opt.data_seed)
hk_key = hk.PRNGSequence(opt.data_seed)

assert opt.max_interv_value == -opt.min_interv_value

# Set some constants
hard = True
opt.num_samples = int(opt.pts_per_interv * opt.n_interv_sets) + opt.obs_data
l_dim = opt.num_nodes * (opt.num_nodes - 1) // 2
num_bethe_iters = opt.bethe_iters

horseshoe_tau = utils.set_horseshoe_tau(opt.num_samples, opt.num_nodes, opt.exp_edges)
logdir = utils.set_logdir(opt)
os.makedirs(logdir, exist_ok=True)

# Instantiate random linear-gaussian SCM with structure and parameters
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
        min_interv_value=opt.min_interv_value,
        identity_perm=True
    )
    
p_z_obs_joint_mu, p_z_obs_joint_covar = utils.get_joint_dist_params(jnp.exp(scm.log_sigma_W), scm.W)
LΣ_prior_dist = Horseshoe(scale=jnp.ones(l_dim + scm.noise_dim) * horseshoe_tau)
gumbel_sinkhorn = GumbelSinkhorn(opt.num_nodes, noise_type="gumbel", tol=opt.max_deviation)

gt_W, gt_P, gt_L = scm.W, scm.P, scm.P.T @ scm.W.T @ scm.P
gt_sigmas = jnp.exp(scm.log_sigma_W)
print(gt_W)

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

x1, x2, z1, z2, labels, targets, interv_values = scm.sample_weakly_supervised(
    rng_key, 
    opt.obs_data, 
    opt.n_interv_sets, 
    return_interv_values=True
)

x_samples = jnp.concatenate([x1, x2], axis=0)
z_samples = jnp.concatenate([z1, z2], axis=0)
interv_targets = jnp.concatenate([jnp.zeros(x1.shape).astype(int), targets], axis=0)
interv_values = jnp.concatenate([jnp.zeros(x1.shape), interv_values], axis=0)

interv_nodes = scm.get_interv_nodes(
    interv_targets.shape[0],
    opt.num_nodes,
    interv_targets.astype(bool)
)

forward, model_params, LΣ_params, model_opt_params, LΣ_opt_params, opt_model, opt_LΣ = init_interv_model(
    hk_key, 
    False, 
    rng_key, 
    opt, 
    x_samples, 
    interv_targets,
    interv_values, 
    l_dim, 
    scm.noise_dim, 
    P=gt_P,
    L=gt_L,
    learn_intervs=opt.learn_intervs
)

vmapped_W_intervened_shd = vmap(utils.W_intervened_shd, (None, 0, None, None))
# batch_get_obs_joint_dist_params = vmap(utils.get_joint_dist_params, (0, 0), (0, 0))
vmapped_kl = vmap(utils.get_single_kl, (None, None, 0, 0, None), (0))

# TODO: Has to be changed for multi target case
gt_obs_targets = 1 - interv_targets.sum(1)
concat_gt_interv_targets = jnp.concatenate((interv_targets, gt_obs_targets[:, None]), axis=1)


def calc_neg_elbo(model_params, LΣ_params, rng_key, interv_nodes, gt_x_data, interv_values):
    """
        Compute the neg. ELBO that has to be minimized:
            - exp_{P, L, Σ} [ exp_{Z | P, L, Σ} log p(X | Z) ] 
            + exp_{P | L, Σ} KL(q(P | L, Σ) || P(P))
            + exp_{L, Σ} KL(q(L, Σ) || P(L)p(Σ))
    """
    nll = 0.
    KL_term_LΣ, KL_term_P = 0., 0.
    (   
        pred_X,
        batch_P,
        batch_P_logits,
        batched_qz_samples,
        full_l_batch, 
        full_log_prob_l, 
        _, _, _, 
        extended_interv_targets, 
        pred_is_observ_logits, 
        pred_is_observ
                ) = forward.apply(
                        model_params, 
                        rng_key, 
                        hard, 
                        rng_key, 
                        opt, 
                        x_samples,
                        interv_nodes,
                        interv_values, 
                        LΣ_params, 
                        P=gt_P,
                        L=gt_L,
                        learn_intervs=opt.learn_intervs
                    )

    if opt.learn_Z is False:
        nll = jit(vmap(utils.get_mse, (None, 0), 0))(z_samples, batched_qz_samples)
    
    elif opt.learn_Z:
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
    obs_classification_loss = utils.bce_loss(gt_obs_targets, pred_is_observ_logits)

    interv_classification_loss = utils.batched_cross_entropy_loss(concat_gt_interv_targets, extended_interv_targets)
    return interv_classification_loss + obs_classification_loss


@jit
def gradient_step(model_params, LΣ_params, rng_key, gt_z_data, gt_x_data, interv_nodes, interv_values):
    """
        1. Compute negative ELBO to be minimized
        2. Obtain gradients with respect to LΣ and the model 
            (P and decoder parameters)
        3. Based on output from forward pass, compute metrics in a dict
    """

    get_loss_and_grad = value_and_grad(calc_neg_elbo, argnums=(0, 1))
    loss, grads = get_loss_and_grad(
        model_params, 
        LΣ_params,
        rng_key,  
        interv_nodes,
        gt_x_data, 
        interv_values
    )

    model_grads, LΣ_grads = tree_map(lambda x_: x_, grads)
    rng_key_ = jax.random.split(rng_key, 1)[0]

    (   pred_X, _, _, 
        z_samples, 
        full_l_batch, 
        full_log_prob_l, _, 
        W_samples, 
        log_noise_std_samples, 
        extended_interv_target,
        pred_is_observ_logits,
        pred_is_observ ) = forward.apply(
                                        model_params, 
                                        rng_key_, 
                                        hard, 
                                        rng_key_, 
                                        opt, 
                                        gt_x_data,
                                        interv_nodes,
                                        interv_values, 
                                        LΣ_params, 
                                        P=gt_P,
                                        L=gt_L,
                                        learn_intervs=opt.learn_intervs
                                    )

    # Obtain KL between predicted and GT observational joint distribution
    batch_get_obs_joint_dist_params = vmap(utils.get_joint_dist_params, (0, 0), (0, 0))
    if opt.eq_noise_var:
        log_noise_std_samples = jnp.tile(log_noise_std_samples, (1, d))
    batch_q_z_obs_joint_mus, batch_q_z_obs_joint_covars = batch_get_obs_joint_dist_params(jnp.exp(log_noise_std_samples), W_samples)
    true_obs_KL_term_Z = vmapped_kl(p_z_obs_joint_covar, 
                                    p_z_obs_joint_mu, 
                                    batch_q_z_obs_joint_covars, 
                                    batch_q_z_obs_joint_mus, 
                                    opt)

    pred_W_means = jnp.mean(W_samples, axis=0)
    LΣ_prior_probs = jnp.sum(LΣ_prior_dist.log_prob(full_l_batch + 1e-7)[:, :l_dim], axis=1)
    KL_term_LΣ = full_log_prob_l - LΣ_prior_probs

    G_samples = jnp.where(jnp.abs(W_samples) >= opt.edge_threshold, 1, 0) 

    log_dict = {
        "L_mse": jnp.mean(vmap(utils.get_mse, (None, 0), 0)(gt_W, jnp.multiply(W_samples, G_samples))),
        "z_mse": jnp.mean(vmap(utils.get_mse, (None, 0), 0)(gt_z_data, z_samples)),
        "x_mse": jnp.mean(vmap(utils.get_mse, (None, 0), 0)(gt_x_data, pred_X)),
        "true_obs_KL_term_Z": jnp.mean(true_obs_KL_term_Z),
        "KL(L)": jnp.mean(KL_term_LΣ)
    }

    return (loss, rng_key_, model_grads, LΣ_grads, log_dict, W_samples, z_samples, pred_X, pred_W_means, 
            extended_interv_target, pred_is_observ_logits, pred_is_observ)


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


tqdm_update_frequency = 200
wandb_log_frequency = 500

(   
    pred_X,
    batch_P,
    batch_P_logits,
    batched_qz_samples,
    full_l_batch, 
    full_log_prob_l, 
    _, _, _, 
    extended_interv_targets, 
    pred_is_observ_logits, 
    pred_is_observ
            ) = forward.apply(
                    model_params, 
                    rng_key, 
                    hard, 
                    rng_key, 
                    opt, 
                    x_samples,
                    interv_nodes,
                    interv_values, 
                    LΣ_params, 
                    P=gt_P,
                    L=gt_L,
                    learn_intervs=opt.learn_intervs
                )

obs_classification_loss = utils.bce_loss(gt_obs_targets, pred_is_observ_logits)

# Training loop
# with tqdm(range(opt.num_steps), miniters=tqdm_update_frequency) as pbar:
#     for i in pbar:
        
#         (   loss, 
#             rng_key, 
#             model_grads, 
#             LΣ_grads, 
#             log_dict, 
#             batch_W, 
#             pred_z_samples, 
#             X_recons, 
#             pred_W_means,
#             extended_interv_target,
#             pred_is_observ_logits,
#             pred_is_observ  ) = gradient_step(
#                                     model_params, 
#                                     LΣ_params,
#                                     rng_key,
#                                     z_samples,
#                                     x_samples,
#                                     interv_nodes,
#                                     interv_values
#                                 )

#         if jnp.any(jnp.isnan(LΣ_grads)):
#             raise Exception("Got NaNs in L grads")

#         if i == 0 or (i+1) % tqdm_update_frequency == 0:
#             pred_interv_targets = extended_interv_target[:, :-1]
#             obs_classification_loss, obs_classification_accuracy = utils.get_obs_classification_accuracy(pred_is_observ_logits, pred_is_observ, gt_obs_targets)
#             interv_classification_loss = utils.batched_cross_entropy_loss(concat_gt_interv_targets, extended_interv_target)
#             W_intervened_shds = vmapped_W_intervened_shd(gt_W, batch_W, interv_targets, pred_interv_targets)
#             pred_match = (extended_interv_target == concat_gt_interv_targets).all(1)
#             num_correct_rows = pred_match.sum()
#             total_accuracy = 100 * (num_correct_rows / interv_targets.shape[0])
#             correct_obs_rows = pred_match[:x1.shape[0]].sum()
#             total_accuracy_obs_rows = 100 * (correct_obs_rows / x1.shape[0])
#             correct_interv_rows = pred_match[x1.shape[0]:].sum()
#             total_accuracy_interv_rows = 100 * (correct_interv_rows / x2.shape[0])

#         if (i+1) % wandb_log_frequency == 0 or i == 0:  
#             wandb_dict, eval_dict = evaluate(
#                 rng_key, 
#                 model_params, 
#                 LΣ_params, 
#                 forward, 
#                 interv_nodes, 
#                 interv_values, 
#                 pred_z_samples, 
#                 z_samples, 
#                 gt_W,
#                 gt_sigmas,
#                 gt_P,
#                 gt_L,
#                 loss,
#                 log_dict,
#                 opt,
#                 interv_model=True,
#                 x_gt=x_samples,
#                 gt_interv_targets=interv_targets,
#                 generative_interv=False
#             )

#             wandb_dict['Interventions/Intervened SHD'] = jnp.mean(W_intervened_shds)
#             wandb_dict['Interventions/Obs. Classification Loss'] = obs_classification_loss
#             wandb_dict['Interventions/Obs. Classification Accuracy'] = obs_classification_accuracy
#             wandb_dict['Interventions/Interv. Classification Loss'] = interv_classification_loss

#             wandb_dict['Interventions/Accuracy (Obs rows)'] = total_accuracy_obs_rows
#             wandb_dict['Interventions/Accuracy (Interv rows)'] = total_accuracy_interv_rows
#             wandb_dict['Interventions/Accuracy (All rows)'] = total_accuracy

#             if opt.off_wandb is False:
#                 plt.savefig(join(logdir, 'pred_w.png'))
#                 wandb_dict["graph_structure(GT-pred)/Predicted W"] = wandb.Image(join(logdir, 'pred_w.png'))
#                 wandb.log(wandb_dict, step=i)

#             shd = eval_dict["shd"]
#             tqdm.write(f"Step {i} | {loss}")
#             tqdm.write(f"Z_MSE: {log_dict['z_mse']} | X_MSE: {log_dict['x_mse']}")
#             tqdm.write(f"L MSE: {log_dict['L_mse']}")
#             tqdm.write(f"SHD: {eval_dict['shd']} | CPDAG SHD: {eval_dict['shd_c']} | AUROC: {eval_dict['auroc']}")
#             tqdm.write(f" ")

#         postfix_dict = OrderedDict( 
#             obs_class_loss=f"{obs_classification_loss:.4f}",
#             L_mse=f"{log_dict['L_mse']:.3f}",
#             # SHD=shd,
#             # AUROC=f"{eval_dict['auroc']:.2f}",
#             acc_obs_rows=f"{total_accuracy_obs_rows:.2f}",
#             acc_interv_rows=f"{total_accuracy_interv_rows:.2f}",
#             acc_all_rows=f"{total_accuracy:.2f}",
#             obs_class_acc=f"{obs_classification_accuracy:.2f}",
#             loss=f"{loss:.4f}",
#         )
#         pbar.set_postfix(postfix_dict)

#         model_params, model_opt_params, LΣ_params, LΣ_opt_params = update_params(
#             model_grads, 
#             model_opt_params, 
#             model_params, 
#             LΣ_grads, 
#             LΣ_opt_params, 
#             LΣ_params
#         )

#         if jnp.any(jnp.isnan(ravel_pytree(LΣ_params)[0])):   
#             raise Exception("Got NaNs in L params")
        
#         if jnp.any(jnp.isnan(ravel_pytree(model_params)[0])):  
#             pdb.set_trace() 
#             raise Exception("Got NaNs in model params")
