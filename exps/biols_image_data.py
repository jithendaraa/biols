import os
from os.path import join
import sys
import pathlib
import ruamel.yaml as yaml
from tqdm import tqdm
import wandb
import pdb
from eval import evaluate
from collections import OrderedDict, namedtuple, defaultdict
import matplotlib.pyplot as plt

sys.path.append('..')
sys.path.append('../CausalMBRL')
sys.path.append('../CausalMBRL/envs')
sys.path.append("../modules")

import jax
from jax import numpy as jnp
from jax import jit, value_and_grad, vmap
import haiku as hk
import optax
from tensorflow_probability.substrates.jax.distributions import Horseshoe
from jax.flatten_util import ravel_pytree
import numpy as onp

import utils
from modules.GumbelSinkhorn import GumbelSinkhorn
from modules.biols_model_init import init_model, OptimizerState, Parameters
import envs
from chem_datagen import generate_chemdata, generate_test_samples

Interventions = namedtuple('Interventions', ['labels', 'values', 'targets'])
Gradients = namedtuple('Gradients', ['model', 'LΣ'])
Losses = namedtuple('Losses', ['nll', 'LΣ_posterior_logprobs', 'LΣ_prior_logprobs', 'KL_term_LΣ', 'log_P_posterior', 'log_P_prior', 'KL_term_P', 'neg_elbo'])

# Load config yaml as options for experiment
configs = yaml.safe_load((pathlib.Path("..") / "biols_config.yaml").read_text())
opt, folder_path = utils.load_yaml(configs)

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
proj_dims = (1, 50, 50)
noise_dim = d
degree = opt.exp_edges
num_bethe_iters = opt.bethe_iters
horseshoe_tau = utils.set_horseshoe_tau(n, d, degree)
logdir = utils.set_logdir(opt)
os.makedirs(logdir, exist_ok=True)

bs = opt.batch_size
num_batches = n // bs
if n % bs != 0: num_batches += 1

LΣ_prior_dist = Horseshoe(scale=jnp.ones(l_dim + noise_dim) * horseshoe_tau)
gumbel_sinkhorn = GumbelSinkhorn(d, noise_type="gumbel", tol=opt.max_deviation)

gt_samples, interventions = utils.read_biols_dataset(folder_path, opt.obs_data)
print(gt_samples.W)
plt.imshow(gt_samples.W)
plt.savefig(join(logdir, 'gt_w.png'))

if opt.off_wandb is False: 
    wandb.init(project = opt.wandb_project, 
                entity = opt.wandb_entity, 
                config = vars(opt), 
                settings = wandb.Settings(start_method="fork"))
    wandb.run.name = logdir.split('/')[-1]
    wandb.run.save()



@jit
def calc_neg_elbo(params, rng_key, batch_interventions, gt_x_data):
    """
        Compute the neg. ELBO that has to be minimized:
            - exp_{P, L, Σ} [ exp_{Z | P, L, Σ} log p(X | Z) ] 
            + exp_{P | L, Σ} KL(q(P | L, Σ) || P(P))
            + exp_{L, Σ} KL(q(L, Σ) || P(L)p(Σ))
    """

    KL_term_LΣ, KL_term_P = 0., 0.
    log_P_posterior, log_P_prior = 0., 0.

    pred_samples, batch_P_logits, LΣ_samples, LΣ_posterior_logprobs, _ = forward.apply(
        params.model, 
        rng_key, 
        hard, 
        rng_key, 
        opt.posterior_samples,
        opt, 
        batch_interventions, 
        params.LΣ, 
        P=gt_samples.P
    )

    # - exp_{P, L, Σ} exp_{Z | P, L, Σ} log p(X | Z)
    vmapped_nll_gaussian = vmap(utils.nll_gaussian, (None, 0, None), 0)
    nll = vmapped_nll_gaussian(gt_x_data/255., pred_samples.x/255., opt.pred_sigma)

    L_prior_logprobs = jnp.sum(LΣ_prior_dist.log_prob(LΣ_samples)[:, :l_dim], axis=1)
    Σ_prior_logprobs = jnp.sum(LΣ_samples[:, l_dim:] ** 2 / (2 * opt.s_prior_std ** 2), axis=-1)
    LΣ_prior_logprobs = L_prior_logprobs + Σ_prior_logprobs
    KL_term_LΣ = LΣ_posterior_logprobs - LΣ_prior_logprobs
    
    if opt.learn_P:
        log_P_posterior = vmap(gumbel_sinkhorn.logprob, in_axes=(0, 0, None))(pred_samples.P, batch_P_logits, num_bethe_iters)
        log_P_prior = -jnp.sum(jnp.log(onp.arange(d) + 1))
        KL_term_P = log_P_posterior - log_P_prior

    neg_elbo = jnp.mean(nll + KL_term_LΣ + KL_term_P)

    losses = Losses(
        nll=nll,
        LΣ_posterior_logprobs=LΣ_posterior_logprobs,
        LΣ_prior_logprobs=LΣ_prior_logprobs,
        KL_term_LΣ=KL_term_LΣ,
        log_P_posterior=log_P_posterior,
        log_P_prior=log_P_prior,
        KL_term_P=KL_term_P,
        neg_elbo=neg_elbo,

    )
    return neg_elbo, (pred_samples, losses)


@jit
def gradient_step(rng_key, params, interventions, gt_z_data, gt_x_data):
    """
        1. Compute negative ELBO to be minimized
        2. Obtain gradients with respect to LΣ and the model 
            (P and decoder parameters)
        3. Based on output from forward pass, compute metrics in a dict
    """
    get_loss_and_grad = value_and_grad(calc_neg_elbo, argnums=(0), has_aux=True)
    (loss, aux_res), grads = get_loss_and_grad(params, rng_key, interventions, gt_x_data)
    pred_samples, losses = aux_res

    gradients = Gradients(model=grads.model, LΣ=grads.LΣ)
    pred_W = jnp.where(jnp.abs(pred_samples.W) >= opt.edge_threshold, pred_samples.W, 0) 
    vmap_mse = vmap(utils.get_mse, (None, 0), 0)

    log_dict = {
        "L_mse": jnp.mean(vmap_mse(gt_samples.W, pred_W)),
        "z_mse": jnp.mean(vmap_mse(gt_z_data, pred_samples.z)),
        "x_mse": jnp.mean(vmap_mse(gt_x_data/255., pred_samples.x/255.)),
        "KL(LΣ)": losses.KL_term_LΣ,
        "KL(P)": losses.KL_term_P,
        "nll": losses.nll,
        "ELBO": losses.neg_elbo,
    }

    return (loss, losses, gradients, log_dict, pred_samples)


def update_params(params, gradients, opt_state):
    """
        Given parameters and gradients, update parameters and optimizer states.
    """
    model_updates, new_model_opt_state = optimizers.model.update(gradients.model, opt_state.model, params.model)
    new_model_params = optax.apply_updates(params.model, model_updates)

    LΣ_updates, new_LΣ_opt_state = optimizers.LΣ.update(gradients.LΣ, opt_state.LΣ, params.LΣ)
    new_LΣ_params = optax.apply_updates(params.LΣ, LΣ_updates)

    new_params = Parameters(model=new_model_params, LΣ=new_LΣ_params)
    new_opt_state = OptimizerState(model=new_model_opt_state, LΣ=new_LΣ_opt_state)
    return new_params, new_opt_state


@jit
def train_batch(rng_key, params, opt_state, batch_interventions, z_gt, images):
    """
        Train a batch of data and update model parameters.
    """
    loss, losses, gradients, log_dict, pred_samples = gradient_step(rng_key, params, batch_interventions, z_gt, images)
    new_params, new_opt_state = update_params(params, gradients, opt_state)
    return ( loss, losses, new_params, new_opt_state, log_dict, pred_samples )                     


def evaluate_random_batch(rng_key, params, gt_samples, interventions):
    random_idxs = onp.random.choice(n, size=bs, replace=False)  

    random_batch_interventions = Interventions(
        labels=interventions.labels[random_idxs],
        values=interventions.values[random_idxs],
        targets=interventions.targets[random_idxs]
    )

    pred_samples, batch_P_logits, LΣ_samples, _, _ = forward.apply(
        params.model, 
        rng_key, 
        hard,
        rng_key, 
        opt.posterior_samples,
        opt, 
        random_batch_interventions, 
        params.LΣ, 
        P=gt_samples.P
    )

    wandb_dict, eval_dict = evaluate(  
        rng_key,
        params,
        forward,
        random_batch_interventions,
        pred_samples,
        gt_samples,
        loss,
        log_dict,
        opt,
        batch_idxs=random_idxs
    )

    if opt.off_wandb is False:  
        pred_W_means = jnp.mean(pred_samples.W, axis=0)
        plt.imshow(pred_W_means)
        plt.savefig(join(logdir, 'pred_w.png'))
        wandb_dict["graph_structure(GT-pred)/Predicted W"] = wandb.Image(join(logdir, 'pred_w.png'))
        wandb.log(wandb_dict, step=i)

    return eval_dict


forward, params, optimizers, opt_state = init_model(
    key, 
    rng_key, 
    opt, 
    interventions,
    l_dim, 
    noise_dim, 
    P=gt_samples.P,
    image=True
)

with tqdm(range(opt.num_steps)) as pbar:
    for i in pbar:
        epoch_dict = defaultdict(lambda: 0.)

        for b in range(num_batches):
            start_idx = b * bs
            end_idx = min(n, (b+1) * bs)

            batch_interventions = Interventions(
                labels=interventions.labels[start_idx:end_idx],
                values=interventions.values[start_idx:end_idx],
                targets=interventions.targets[start_idx:end_idx]
            )

            loss, losses, params, opt_state, log_dict, pred_samples = train_batch(
                rng_key, 
                params, 
                opt_state,
                batch_interventions,
                gt_samples.z[start_idx:end_idx],
                gt_samples.x[start_idx:end_idx],
            )
            
            if jnp.any(jnp.isnan(ravel_pytree(params.LΣ)[0])):   raise Exception("Got NaNs in L params")
            if jnp.any(jnp.isnan(ravel_pytree(params.model)[0])):   raise Exception("Got NaNs in model params")

            for key, val in log_dict.items():
                epoch_dict[key] += (val / num_batches)
        
        if (i+1) % 100 == 0 or i == 0:  
            eval_dict = evaluate_random_batch(rng_key, params, gt_samples, interventions)
            shd = eval_dict["shd"]
            tqdm.write(f"Step {i} | Neg. ELBO: {loss}")
            tqdm.write(f"Z_MSE: {epoch_dict['z_mse']} | X_MSE: {epoch_dict['x_mse']}")
            tqdm.write(f"L MSE: {epoch_dict['L_mse']}")
            tqdm.write(f"SHD: {eval_dict['shd']} | CPDAG SHD: {eval_dict['shd_c']} | AUROC: {eval_dict['auroc']}")
            tqdm.write(f" ")
        
        postfix_dict = OrderedDict(
            Epoch=i,
            L_mse=f"{epoch_dict['L_mse']:.5f}",
            SHD=shd,
            AUROC=f"{eval_dict['auroc']:.3f}",
            loss=f"{loss:.4f}",
        )
        pbar.set_postfix(postfix_dict)


pad_len = 20


# Generate test samples
test_interv_z, test_interventions, gt_test_images = generate_test_samples(
    rng_key,
    gt_samples.W, 
    scm_noise_sigmas=gt_samples.sigmas, 
    interv_noise_dist_sigma=opt.interv_noise_dist_sigma,
    use_interv_noise=not opt.no_interv_noise,
    decoder_sigma=opt.decoder_sigma,
    clamp_low=opt.clamp_low, 
    clamp_high=opt.clamp_high,
    pad_len=pad_len
)


pred_samples, _, _, _, _, = forward.apply(
    params.model, 
    rng_key, 
    hard, 
    rng_key, 
    2000,
    opt, 
    test_interventions, 
    params.LΣ, 
    P=gt_samples.P
)


def get_padded_images(images):
    padded_images = jnp.zeros((h, pad_len, c))
    for i in range(d):
        image = images[i]
        padded_images = jnp.concatenate((padded_images, image, jnp.zeros((h, pad_len, c))), axis=1)
    return padded_images[:, :, 0]

_, h, w, c = pred_samples.x[0].shape
padded_pred_images = get_padded_images(pred_samples.x[0])

padded_posterior_samples = vmap(get_padded_images, (0), (0))(pred_samples.x) # (posterior_samples, padded_h, padded_w)
sample_difference = onp.mean(onp.abs(gt_test_images[None, :, :] - padded_posterior_samples), axis=0)
test_gt_image_path = f'{folder_path}/gt_test_images.png'
test_pred_image_path = f'{folder_path}/pred_test_images.png'
difference_image_path = f'{folder_path}/difference_images.png'

plt.imsave(test_gt_image_path, gt_test_images, cmap='Purples')
plt.imsave(test_pred_image_path, padded_pred_images, cmap='Purples')
plt.imsave(difference_image_path, sample_difference, cmap='Purples')

if opt.off_wandb is False:
    wandb.log({
        "graph_structure(GT-pred)/Ground truth W": wandb.Image(join(logdir, 'gt_w.png')),
        "OOD Generalization/GT interv. images": wandb.Image(test_gt_image_path),
        "OOD Generalization/Pred interv. images": wandb.Image(test_pred_image_path),
    }, step=opt.num_steps+1)