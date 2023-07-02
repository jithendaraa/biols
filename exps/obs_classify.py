import os, sys, pdb
sys.path.append("..")
sys.path.append("../modules")

import pathlib
import ruamel.yaml as yaml
from tqdm import tqdm

import optax

import numpy as onp
import jax.numpy as jnp
import jax
from jax import config, value_and_grad, jit
import haiku as hk
from collections import OrderedDict


from models.Classifier import init_classifier_model
import utils
from datagen import SyntheticDatagen

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

gt_W, gt_P, gt_L = scm.W, scm.P, scm.P.T @ scm.W.T @ scm.P
gt_sigmas = jnp.exp(scm.log_sigma_W)
print(gt_W)

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

encoder_out = 1
gt_obs_targets = 1 - interv_targets.sum(1)
forward, model_params, opt_model, model_opt_params = init_classifier_model(
    hk_key,
    opt.lr,
    opt.proj_dims,
    encoder_out,
    x_samples
)


@jit
def loss_fn(model_params, rng_key, x_samples):
    pred_logits = forward.apply(
        model_params,
        rng_key,  
        opt.proj_dims, 
        encoder_out,
        x_samples
    )
    loss = utils.bce_loss(gt_obs_targets, pred_logits)
    return loss


@jit
def gradient_step(rng_key, model_params, x_gt):
    get_loss_and_grad = value_and_grad(loss_fn, argnums=(0))
    loss, gradients = get_loss_and_grad(
        model_params,
        rng_key,
        x_gt
    )
    return loss, gradients


@jit
def update_params(model_params, gradients, opt_params):
    updates, opt_params = opt_model.update(gradients, opt_params, model_params)
    model_params = optax.apply_updates(model_params, updates)
    return model_params, opt_params


@jit
def forward_pass(rng_key, model_params, x_samples):
    pred_logits = forward.apply(
        model_params,
        rng_key,  
        opt.proj_dims, 
        encoder_out,
        x_samples
    )
    return pred_logits


with tqdm(range(opt.num_steps)) as pbar:
    for i in pbar:
        loss, gradients = gradient_step(
            rng_key,
            model_params,
            x_samples
        )

        pred_logits = forward_pass(rng_key, model_params, x_samples)
        obs_classification_loss, obs_classification_accuracy = utils.get_obs_classification_accuracy(
            rng_key, 
            pred_logits, 
            gt_obs_targets
        )

        postfix_dict = OrderedDict( 
            loss=f"{loss:.4f}",
            obs_class_loss=f"{obs_classification_loss:.2f}",
            obs_class_accuracy=f"{obs_classification_accuracy:.2f}"
        )
        pbar.set_postfix(postfix_dict)

        model_params, model_opt_params = update_params(model_params, gradients, model_opt_params)

        