import sys
sys.path.append('../models')

from jax import jit
import jax.numpy as jnp
import numpy as onp
import haiku as hk
import optax, pdb
from jax.flatten_util import ravel_pytree

from BIOLS import BIOLS
from BIOLS_Interv import BIOLS_Interv
from BIOLS_Image import BIOLS_Image
from Generative_BIOLS_Interv import GenerativeBIOLS_Interv

def biols_interv_forward_fn(hard, rng_key, opt, X, interv_targets, 
        interv_values, LΣ_params, P=None, L=None, learn_intervs=True):

    model = BIOLS_Interv(
        opt.num_nodes, 
        opt.posterior_samples, 
        opt.eq_noise_var, 
        opt.fixed_tau,
        opt.hidden_size,
        opt.max_deviation,
        opt.learn_P,
        opt.learn_L,
        opt.proj_dims,
        interv_type=opt.interv_type,
        log_stds_max=opt.log_stds_max,
        logit_constraint=opt.logit_constraint,
        P=P,
        L=L,
        interv_targets=interv_targets,
        learn_intervs=learn_intervs
    )

    return model(rng_key, X, interv_values, LΣ_params, hard)


def init_interv_model(hk_key, hard, rng_key, opt, X, interv_targets, interv_values, 
        l_dim, noise_dim, P=None, L=None, learn_intervs=True):
    """
        Initialize model, parameters, and optimizer states
    """
    forward = hk.transform(biols_interv_forward_fn)
    model_layers = [optax.scale_by_belief(eps=1e-8), optax.scale(-opt.lr)]
    LΣ_layers = [optax.scale_by_belief(eps=1e-8), optax.scale(-opt.lr)]
    opt_model = optax.chain(*model_layers)
    opt_LΣ = optax.chain(*LΣ_layers)

    if opt.learn_noise:
        LΣ_params = jnp.concatenate((jnp.zeros(l_dim), jnp.zeros(noise_dim), jnp.zeros(l_dim + noise_dim) - 1)).astype(jnp.float32)
    else:
        raise NotImplementedError("Need to implement learn_noise False")

    next_key = next(hk_key)
    model_params = forward.init(
        next_key, 
        hard, 
        rng_key, 
        opt, 
        X, 
        interv_targets, 
        interv_values, 
        LΣ_params, 
        P=P, 
        L=L, 
        learn_intervs=learn_intervs
    )
    model_opt_params = opt_model.init(model_params)
    LΣ_opt_params = opt_LΣ.init(LΣ_params)

    print(f"L and Σ has {ff2(num_params(LΣ_params))} parameters")
    print(f"Model has {ff2(num_params(model_params))} parameters")
    return forward, model_params, LΣ_params, model_opt_params, LΣ_opt_params, opt_model, opt_LΣ


def generative_biols_interv_forward_fn(hard, rng_key, opt, interv_targets, interv_values, LΣ_params, interv_logit_params, P=None, L=None, learn_intervs=True):
    model = GenerativeBIOLS_Interv(
        opt.num_nodes, 
        opt.posterior_samples, 
        opt.eq_noise_var, 
        opt.fixed_tau,
        opt.hidden_size,
        opt.max_deviation,
        opt.learn_P,
        opt.learn_L,
        opt.proj_dims,
        interv_type=opt.interv_type,
        log_stds_max=opt.log_stds_max,
        logit_constraint=opt.logit_constraint,
        P=P,
        L=L,
        interv_targets=interv_targets,
        learn_intervs=learn_intervs
    )

    return model(rng_key, interv_values, LΣ_params, interv_logit_params, hard)


def init_generative_interv_model(hk_key, hard, rng_key, opt, interv_targets, interv_values, 
        l_dim, noise_dim, P=None, L=None, learn_intervs=True):
    """
        Initialize model, parameters, and optimizer states
    """
    forward = hk.transform(generative_biols_interv_forward_fn)
    model_layers = [optax.scale_by_belief(eps=1e-8), optax.scale(-opt.lr)]
    LΣ_layers = [optax.scale_by_belief(eps=1e-8), optax.scale(-opt.lr)]
    interv_logit_layers = [optax.scale_by_belief(eps=1e-8), optax.scale(-opt.lr)]

    opt_model = optax.chain(*model_layers)
    opt_LΣ = optax.chain(*LΣ_layers)
    opt_interv_logits = optax.chain(*interv_logit_layers)

    if opt.learn_noise: LΣ_params = jnp.concatenate((jnp.zeros(l_dim), jnp.zeros(noise_dim), jnp.zeros(l_dim + noise_dim) - 1)).astype(jnp.float32)
    else:               raise NotImplementedError("Need to implement learn_noise False")
    
    n, d = interv_targets.shape
    interv_logit_params = 10 * jnp.ones((n, d+1)).astype(jnp.float32)

    next_key = next(hk_key)
    model_params = forward.init(
        next_key, 
        hard, 
        rng_key, 
        opt, 
        interv_targets, 
        interv_values, 
        LΣ_params, 
        interv_logit_params,
        P=P, 
        L=L, 
        learn_intervs=learn_intervs
    )
    model_opt_params = opt_model.init(model_params)
    LΣ_opt_params = opt_LΣ.init(LΣ_params)
    interv_logit_opt_params = opt_interv_logits.init(interv_logit_params)

    print(f"Intervention logits has {ff2(num_params(interv_logit_params))} parameters")
    print(f"L and Σ has {ff2(num_params(LΣ_params))} parameters")
    print(f"Model has {ff2(num_params(model_params))} parameters")

    params = (model_params, LΣ_params, interv_logit_params)
    opt_params = (model_opt_params, LΣ_opt_params, interv_logit_opt_params)
    optimizers = (opt_model, opt_LΣ, opt_interv_logits)
    return forward, params, opt_params, optimizers


def forward_fn(hard, rng_key, opt, interv_targets, interv_values, LΣ_params, P=None):
    model = BIOLS(
        opt.num_nodes, 
        opt.posterior_samples, 
        opt.fixed_tau,
        opt.hidden_size,
        opt.max_deviation,
        opt.learn_P,
        opt.proj_dims, 
        opt.interv_value_sampling,
        opt.no_interv_noise,
        log_stds_max=opt.log_stds_max,
        logit_constraint=opt.logit_constraint,
        P=P
    )

    return model(rng_key, interv_targets, interv_values, LΣ_params, hard)


def init_model(key, hard, rng_key, opt, interv_targets, interv_values, l_dim, noise_dim, P=None):
    """
        Initialize model, parameters, and optimizer states
    """
    forward = hk.transform(forward_fn)
    model_layers = [optax.scale_by_belief(eps=1e-8), optax.scale(-opt.lr)]
    LΣ_layers = [optax.scale_by_belief(eps=1e-8), optax.scale(-opt.lr)]
    
    opt_model = optax.chain(*model_layers)
    opt_LΣ = optax.chain(*LΣ_layers)
    LΣ_params = jnp.concatenate((jnp.zeros(l_dim), jnp.zeros(noise_dim), jnp.zeros(l_dim + noise_dim) - 1)).astype(jnp.float32)
    
    next_key = next(key)
    model_params = forward.init(next_key, hard, rng_key, opt, interv_targets, interv_values, LΣ_params, P)
    model_opt_params = opt_model.init(model_params)
    LΣ_opt_params = opt_LΣ.init(LΣ_params)

    print(f"L and Σ has {ff2(num_params(LΣ_params))} parameters")
    print(f"Model has {ff2(num_params(model_params))} parameters")
    return forward, model_params, LΣ_params, model_opt_params, LΣ_opt_params, opt_model, opt_LΣ


def image_forward_fn(hard, proj_dims, rng_key, opt, interv_targets, interv_values, LΣ_params, P=None):

    model = BIOLS_Image(opt.num_nodes, 
                    opt.posterior_samples, 
                    proj_dims,
                    opt.eq_noise_var, 
                    opt.fixed_tau,
                    opt.hidden_size,
                    opt.max_deviation,
                    opt.learn_P,
                    opt.log_stds_max,
                    opt.logit_constraint,
                    P=P)

    return model(rng_key, interv_targets, interv_values, LΣ_params, hard)


def init_image_model(hard, rng_key, opt, proj_dims, interv_targets, interv_values, 
                    l_dim, noise_dim, P=None):
    """
        Initialize model, parameters, and optimizer states
    """                    
    forward = hk.transform(image_forward_fn)
    model_layers = [optax.scale_by_belief(eps=1e-8), optax.scale(-opt.lr)]
    LΣ_layers = [optax.scale_by_belief(eps=1e-8), optax.scale(-opt.lr)]
    
    opt_model = optax.chain(*model_layers)
    opt_LΣ = optax.chain(*LΣ_layers)

    if opt.learn_noise:
        LΣ_params = jnp.concatenate((jnp.zeros(l_dim), jnp.zeros(noise_dim), jnp.zeros(l_dim + noise_dim) - 1)).astype(jnp.float32)
    else:
        raise NotImplementedError("Need to implement learn_noise False")

    model_params = forward.init(rng_key,
                                hard, 
                                proj_dims,
                                rng_key, 
                                opt,
                                interv_targets[:opt.batches], 
                                interv_values[:opt.batches], 
                                LΣ_params, 
                                P)
    model_opt_params = opt_model.init(model_params)
    LΣ_opt_params = opt_LΣ.init(LΣ_params)

    print(f"L and Σ has {ff2(num_params(LΣ_params))} parameters")
    print(f"Model has {ff2(num_params(model_params))} parameters")
    return forward, model_params, LΣ_params, model_opt_params, LΣ_opt_params, opt_model, opt_LΣ


def ff2(x):
    if type(x) is str: return x
    if onp.abs(x) > 1000 or onp.abs(x) < 0.1: return onp.format_float_scientific(x, 3)
    else: return f"{x:.2f}"


def num_params(params: hk.Params) -> int:
    return len(ravel_pytree(params)[0])
