import sys
sys.path.append('../models')

import optax
import haiku as hk
import numpy as onp
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_curve, auc
from jax import jit
from jax import numpy as jnp

from VAE import VAE

get_mse = lambda recon, x: jnp.mean(jnp.square(recon - x)) 

def numerical_vae_forward_fn(d, proj_dims, rng_key, X, corr):
    model = VAE(d, proj_dims, corr, sigmoid=False)
    return model(rng_key, X, corr)

def init_vector_vae_params(opt, proj_dims, key, rng_key, x_data):
    forward = hk.transform(numerical_vae_forward_fn)
    model_layers = [optax.scale_by_belief(eps=1e-8), optax.scale(-opt.lr)]
    opt_model = optax.chain(*model_layers)
    model_params = forward.init(next(key), opt.num_nodes, 
                                proj_dims, rng_key, 
                                x_data, 
                                opt.corr)
    model_opt_params = opt_model.init(model_params)
    return forward, model_params, model_opt_params, opt_model

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