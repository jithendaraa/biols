import os
import sys, pdb
sys.path.append("..")
sys.path.append("../modules")

from os.path import join
import pathlib
import ruamel.yaml as yaml

import matplotlib.pyplot as plt
import wandb
import numpy as onp
import jax.numpy as jnp
import jax
from jax import config, vmap, random, jit
import haiku as hk
import utils
import datagen
import time
from sklearn.metrics import roc_curve, auc
from scipy.optimize import linear_sum_assignment

from modules.SyntheticSCM import SyntheticSCM
from causallearn.search.HiddenCausal.GIN.GIN import GIN

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
gt_G = jnp.where(jnp.abs(gt_W) != 0, 1, 0)
print(gt_W)
print(gt_G)

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

data = onp.array(x_gt[:opt.obs_data])
start_time = time.time()
G, K = GIN(data)
end_time = time.time()
print(f"Time for d={opt.num_nodes}, D={opt.proj_dims}: {end_time - start_time}s")

res = onp.zeros_like(gt_W)
latent_idxs = []
num_Xs = 0
len_Ks = [len(K[i]) for i in range(len(K))]
for i in range(len(len_Ks)):
    num_Xs += len_Ks[i]
i = 0
len_K_idx = 0
while i < len(G.graph) and len_K_idx < len(K):
    latent_idxs.append(i)
    i += len_Ks[len_K_idx] + 1
    len_K_idx += 1

idx_list = onp.argwhere(G.graph.T == 1)
for idx in idx_list:
    latent_idxs.append(idx[0])

latent_idxs = onp.sort(onp.unique(onp.array(latent_idxs))).tolist()
res = G.graph[latent_idxs][:, latent_idxs].T

if K == []:
    res = onp.zeros_like(gt_W)

elif num_Xs != opt.proj_dims and num_Xs > 0:
    print("No match!")
    latent_idxs = []
    pred_graph = G.graph[(opt.proj_dims - num_Xs):, (opt.proj_dims - num_Xs):]

    i = 0
    len_K_idx = 0
    while i < len(pred_graph) and len_K_idx < len(K):
        latent_idxs.append(i)
        i += len_Ks[len_K_idx] + 1
        len_K_idx += 1

    idx_list = onp.argwhere(pred_graph.T == 1)
    for idx in idx_list:
        latent_idxs.append(idx[0])

    latent_idxs = onp.sort(onp.unique(onp.array(latent_idxs))).tolist()
    res = pred_graph[latent_idxs][:, latent_idxs].T

print(res)
res = onp.where(res > 0, 1, 0)

with open('gin.npy', 'wb') as f:
    onp.save(f, onp.array(res))

gt_len, pred_len = len(gt_W), len(res)

if gt_len > pred_len:
    res = onp.concatenate((res, onp.zeros((gt_len - pred_len, pred_len)) ), axis=0)
    res = onp.concatenate((res, onp.zeros((gt_len, gt_len - pred_len)) ), axis=1)

elif gt_len < pred_len:
    gt_G = onp.concatenate((gt_G, onp.zeros((pred_len - gt_len, gt_len)) ), axis=0)
    gt_G = onp.concatenate((gt_G, onp.zeros((pred_len, pred_len - gt_len)) ), axis=1)

    gt_W = onp.concatenate((gt_W, onp.zeros((pred_len - gt_len, gt_len)) ), axis=0)
    gt_W = onp.concatenate((gt_W, onp.zeros((pred_len, pred_len - gt_len)) ), axis=1)

res = res.astype(int)
print(res.shape, gt_G.shape)

def from_W(W: jnp.ndarray, dim: int) -> jnp.ndarray:
    """Turns a d x d matrix into a (d x (d-1)) vector with zero diagonal."""
    out_1 = W[onp.triu_indices(dim, 1)]
    out_2 = W[onp.tril_indices(dim, -1)]
    return onp.concatenate([out_1, out_2])

def auroc(Ws, W_true, threshold=0.3):
    """
        Given a sample of adjacency graphs of shape (n, d, d)
        compute the AUROC for detecting edges. For each edge, we compute
        a probability that there is an edge there which is the frequency with 
        which the sample has edges over threshold.
    """
    _, dim, dim = Ws.shape
    edge_present = jnp.abs(Ws) > threshold
    prob_edge_present = jnp.mean(edge_present, axis=0)
    true_edges = from_W(jnp.abs(W_true) > threshold, dim).astype(int)
    predicted_probs = from_W(prob_edge_present, dim)
    fprs, tprs, _ = roc_curve(y_true=true_edges, y_score=predicted_probs, pos_label=1)
    auroc_score = auc(fprs, tprs)
    return auroc_score


shd = onp.sum(onp.abs(res - gt_G))
auroc_score = auroc(jnp.array(res[None, :, :]), gt_G)
L_mse = onp.mean((gt_W - res)**2)

print(f"SHD: {shd}")
print(f"AUROC: {auroc_score}")
print(f"L_MSE: {L_mse}")


