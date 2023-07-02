import sys, pdb
from collections import namedtuple
sys.path.append('../models')

from models.EncoderBIOLS import EncoderBIOLS

from jax import jit
import jax.numpy as jnp
import numpy as onp
import haiku as hk
import optax, pdb
from jax.flatten_util import ravel_pytree


Params = namedtuple('Params', ['model', 'LΣ'])
OptimizerParams = namedtuple('OptimizerParams', ['model', 'LΣ'])
Optimizers = namedtuple('Optimizers', ['model', 'LΣ'])


def encoder_biols_forward_fn(rng_key, hard, X, opt, interv_targets, interv_values, LΣ_params, P=None, L=None, learn_intervs=True):
    model = EncoderBIOLS(
        opt.num_nodes, 
        opt.posterior_samples, 
        opt.eq_noise_var, 
        opt.fixed_tau,
        opt.hidden_size,
        opt.max_deviation,
        opt.learn_P,
        opt.proj_dims, 
        opt.interv_value_sampling,
        log_stds_max=opt.log_stds_max,
        logit_constraint=opt.logit_constraint,
        P=P
    )

    return model(rng_key, X, interv_targets, interv_values, LΣ_params, hard)


def init_model(key, rng_key, X, opt, interv_targets, interv_values, l_dim, noise_dim, P=None):
    """
        Initialize model, parameters, and optimizer parameters (or states)
    """
    next_key = next(key)
    forward = hk.transform(encoder_biols_forward_fn)    
    assert opt.learn_noise is True # Not implemented for False
   
    means = jnp.zeros(l_dim + noise_dim)
    log_stds = jnp.zeros(l_dim + noise_dim) - 1
    LΣ_params = jnp.concatenate((means, log_stds)).astype(jnp.float32)

    model_params = forward.init(
        next_key, 
        rng_key, 
        False, 
        X,
        opt, 
        interv_targets, 
        interv_values, 
        LΣ_params, 
        P
    )

    model_layers = [optax.scale_by_belief(eps=1e-8), optax.scale(-opt.lr)]
    LΣ_layers = [optax.scale_by_belief(eps=1e-8), optax.scale(-opt.lr)]
    
    params = Params(
        model=model_params, 
        LΣ=LΣ_params
    )

    optimizers = Optimizers(
        model=optax.chain(*model_layers),
        LΣ=optax.chain(*LΣ_layers)
    )

    optimizer_params = OptimizerParams(
        model=optimizers.model.init(params.model),
        LΣ=optimizers.LΣ.init(params.LΣ)
    )

    print(f"Model has {ff2(num_params(params.model))} parameters")
    print(f"L and Σ has {ff2(num_params(params.LΣ))} parameters")
    return forward, params, optimizers, optimizer_params


def ff2(x):
    if type(x) is str: return x
    if onp.abs(x) > 1000 or onp.abs(x) < 0.1: return onp.format_float_scientific(x, 3)
    else: return f"{x:.2f}"


def num_params(params: hk.Params) -> int:
    return len(ravel_pytree(params)[0])


