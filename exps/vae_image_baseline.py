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
from collections import defaultdict

from modules.vae_model_init import init_image_vae_params, get_vae_auroc, get_cross_correlation, get_mse, get_covar, get_single_kl
import utils


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
gt_samples, interventions = utils.read_biols_dataset(folder_path, opt.obs_data)
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


n, h, w, c = gt_samples.x.shape
proj_dims = h*w*c
flat_images = gt_samples.x.reshape(-1, h*w*c)


def get_neg_elbo(rng_key, params, gt_samples):
    x_pred, z_pred, q_z_mus, z_L_chols = forward.apply(params, rng_key, rng_key, proj_dims, opt.num_nodes, gt_samples.x, opt.corr)
    z_mse = jnp.mean(get_mse(z_pred, gt_samples.z))
    x_mse = jnp.mean(get_mse(x_pred/255., gt_samples.x/255.))
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
def get_gradients(params, batch_gt_samples):
    (loss, aux_res), grads = value_and_grad(get_neg_elbo, argnums=(1), has_aux=True)(rng_key, params, batch_gt_samples)
    losses, pred_samples = aux_res
    return losses, grads, pred_samples

@jit
def train_batch(opt_state, params, batch_gt_samples):
    losses, grads, pred_samples = get_gradients(params, batch_gt_samples)
    model_updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, model_updates)
    return losses, grads, pred_samples, opt_state, params

forward, params, optimizer, opt_state = init_image_vae_params(
    opt, 
    proj_dims,
    key, 
    rng_key, 
    flat_images[:opt.batch_size]
)

GTSamples = namedtuple('GTSamples', ['x', 'z'])
batch_gt_samples = GTSamples(
    x=flat_images[:opt.batch_size],
    z=gt_samples.z[:opt.batch_size]
)

losses, grads, pred_samples, opt_state, params = train_batch(opt_state, params, batch_gt_samples)

num_batches = n // opt.batch_size
if n % opt.batch_size != 0: num_batches += 1

with tqdm(range(opt.num_steps)) as pbar:
    for i in pbar:
        epoch_dict = defaultdict(lambda: 0.)

        for b in range(num_batches):
            start_idx = b * opt.batch_size
            end_idx = min(n, (b+1) * opt.batch_size)

            batch_gt_samples = GTSamples(
                x=flat_images[start_idx:end_idx],
                z=gt_samples.z[start_idx:end_idx]
            )

            losses, grads, pred_samples, opt_state, params = train_batch(opt_state, params, batch_gt_samples)
            epoch_dict['ELBO'] += losses.neg_elbo / num_batches
            epoch_dict['z_mse'] += losses.z_mse / num_batches
            epoch_dict['x_mse'] += losses.x_mse / num_batches
            epoch_dict['nll'] += losses.nll / num_batches
            epoch_dict['kl_loss'] += losses.kl_loss / num_batches

        auroc = get_vae_auroc(opt.num_nodes, gt_W)
        wandb_dict = {
            "ELBO": epoch_dict['ELBO'],
            "Z_MSE": epoch_dict['z_mse'],
            "X_MSE": epoch_dict['x_mse'],
            "L_MSE": L_mse,
            "Evaluations/SHD": jnp.sum(binary_gt_W),
            "Evaluations/AUROC": auroc,
        }

        if (i+1) % 50 == 0 or i == 0:       
            if opt.off_wandb is False:  
                wandb.log(wandb_dict, step=i)

        pbar.set_postfix(
            ELBO=f"{losses.neg_elbo:.4f}",
            Z_MSE=f"{losses.z_mse:.4f}",
            X_MSE=f"{losses.x_mse:.4f}",
            SHD=jnp.sum(binary_gt_W),
            L_mse=f"{L_mse:.3f}",
            AUROC=f"{auroc:.2f}"
        )