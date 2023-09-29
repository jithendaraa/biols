import sys
sys.path.append('../models')

import optax, pdb
import haiku as hk
import numpy as onp
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_curve, auc
from jax import jit
from jax import numpy as jnp
from VAE import VAE

get_mse = lambda recon, x: jnp.mean(jnp.square(recon - x)) 

def numerical_vae_forward_fn(rng_key, d, X, corr):
    if len(X.shape) == 4:
        proj_dims = X.shape[-2] * X.shape[-3]
    else:
        proj_dims = X.shape[-1]
    model = VAE(d, proj_dims, corr, sigmoid=False)
    return model(rng_key, X, corr)

def init_vector_vae_params(opt, key, rng_key, x_data):
    model_layers = [optax.scale_by_belief(eps=1e-8), optax.scale(-opt.lr)]
    forward = hk.transform(numerical_vae_forward_fn)
    params = forward.init(next(key), rng_key, opt.num_nodes, x_data, opt.corr)
    optimizer = optax.chain(*model_layers)
    opt_state = optimizer.init(params)
    return forward, params, optimizer, opt_state

def image_vae_forward_fn(rng_key, proj_dims, d, X, corr):
    model = VAE(d, proj_dims, corr, sigmoid=False)
    return model(rng_key, X, corr)

def init_image_vae_params(opt, proj_dims, key, rng_key, x_data):
    model_layers = [optax.scale_by_belief(eps=1e-8), optax.scale(-opt.lr)]
    optimizer = optax.chain(*model_layers)
    forward = hk.transform(image_vae_forward_fn)
    params = forward.init(next(key), rng_key, proj_dims, opt.num_nodes, x_data, opt.corr)
    opt_state = optimizer.init(params)
    return forward, params, optimizer, opt_state


def from_W(W: jnp.ndarray, dim: int) -> jnp.ndarray:
    """Turns a d x d matrix into a (d x (d-1)) vector with zero diagonal."""
    out_1 = W[onp.triu_indices(dim, 1)]
    out_2 = W[onp.tril_indices(dim, -1)]
    return onp.concatenate([out_1, out_2])


def get_vae_auroc(d, gt_W, threshold=0.3):
    """Given a sample of adjacency graphs of shape n x d x d, 
    compute the AUROC for detecting edges. For each edge, we compute
    a probability that there is an edge there which is the frequency with 
    which the sample has edges over threshold."""
    
    pred_Ws = jnp.zeros((1, d, d))
    _, dim, dim = pred_Ws.shape
    edge_present = jnp.abs(pred_Ws) > threshold
    prob_edge_present = jnp.mean(edge_present, axis=0)
    true_edges = from_W(jnp.abs(gt_W) > threshold, dim).astype(int)
    predicted_probs = from_W(prob_edge_present, dim)
    fprs, tprs, _ = roc_curve(y_true=true_edges, y_score=predicted_probs, pos_label=1)
    auroc = auc(fprs, tprs)
    return auroc


def get_cross_correlation(pred_latent, true_latent):
    dim= pred_latent.shape[1]
    cross_corr= onp.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            cross_corr[i,j]= (onp.cov( pred_latent[:,i], true_latent[:,j] )[0,1]) / ( onp.std(pred_latent[:,i])*onp.std(true_latent[:,j]) )
    
    cost= -1*onp.abs(cross_corr)
    row_ind, col_ind= linear_sum_assignment(cost)
    
    score= 100*onp.sum( -1*cost[row_ind, col_ind].sum() )/(dim)
    return score


def get_single_kl(p_z_covar, p_z_mu, q_z_covar, q_z_mu, opt):
    mu_diff = p_z_mu - q_z_mu
    kl_loss = 0.5 * (   jnp.log(jnp.linalg.det(p_z_covar)) - \
                        jnp.log(jnp.linalg.det(q_z_covar)) - \
                        opt.num_nodes + \
                        jnp.trace( jnp.matmul(jnp.linalg.inv(p_z_covar), q_z_covar) ) + \
                        jnp.matmul(jnp.matmul(jnp.transpose(mu_diff), jnp.linalg.inv(p_z_covar)), mu_diff)
                    )

    return kl_loss

@jit
def get_covar(L):
    return L @ L.T