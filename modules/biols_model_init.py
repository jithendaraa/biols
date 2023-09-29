import sys
sys.path.append('../models')

from jax import jit
import jax.numpy as jnp
import numpy as onp
import haiku as hk
import optax, pdb
from jax.flatten_util import ravel_pytree
from collections import namedtuple

from BIOLS import BIOLS
from BIOLS_Interv import BIOLS_Interv
from BIOLS_Image import BIOLS_Image
from Generative_BIOLS_Interv import GenerativeBIOLS_Interv

Parameters = namedtuple('Parameters', ['LΣ', 'model'])
Optimizers = namedtuple('Optimizers', ['LΣ', 'model'])
OptimizerState = namedtuple('OptimizerState', ['LΣ', 'model'])


def forward_fn(hard, rng_key, num_posterior_samples, opt, interventions, LΣ_params, P=None):
    model = BIOLS(
        d=opt.num_nodes, 
        tau=opt.fixed_tau,
        hidden_size=opt.hidden_size,
        max_deviation=opt.max_deviation,
        learn_P=opt.learn_P,
        proj_dims=opt.proj_dims, 
        interv_value_sampling=opt.interv_value_sampling,
        no_interv_noise=opt.no_interv_noise,
        log_stds_max=opt.log_stds_max,
        logit_constraint=opt.logit_constraint,
        P=P,
        pred_sigma=opt.pred_sigma,
        interv_noise_dist_sigma=opt.interv_noise_dist_sigma
    )

    return model(rng_key, num_posterior_samples, interventions.labels, interventions.values, LΣ_params, hard)


def image_forward_fn(hard, rng_key, num_posterior_samples, opt, interventions, LΣ_params, P=None):
    proj_dims = (1, int(opt.proj_dims ** 0.5), int(opt.proj_dims ** 0.5))
    model = BIOLS_Image(
        opt.num_nodes,
        proj_dims,
        opt.fixed_tau,
        opt.hidden_size,
        opt.learn_P,
        interv_value_sampling=opt.interv_value_sampling,
        no_interv_noise=opt.no_interv_noise,
        log_stds_max=opt.log_stds_max,
        logit_constraint=opt.logit_constraint,
        P=P,
        pred_sigma=opt.pred_sigma,
        interv_noise_dist_sigma=opt.interv_noise_dist_sigma
    )

    return model(rng_key, num_posterior_samples, interventions.labels, interventions.values, LΣ_params, hard)


def init_model(key, rng_key, opt, interventions, l_dim, noise_dim, P=None, image=False):
    """
        Initialize model, parameters, and optimizer states
    """
    if image:
        forward = hk.transform(image_forward_fn)
    else:
        forward = hk.transform(forward_fn)

    model_layers = [optax.scale_by_belief(eps=1e-8), optax.scale(-opt.lr)]
    LΣ_layers = [optax.scale_by_belief(eps=1e-8), optax.scale(-opt.lr)]
    
    opt_model = optax.chain(*model_layers)
    opt_LΣ = optax.chain(*LΣ_layers)
    
    LΣ_params = jnp.concatenate((jnp.zeros(l_dim), jnp.zeros(noise_dim), jnp.zeros(l_dim + noise_dim) - 1)).astype(jnp.float32)
    model_params = forward.init(next(key), False, rng_key, opt.posterior_samples, opt, interventions, LΣ_params, P)

    model_opt_state = opt_model.init(model_params)
    LΣ_opt_state = opt_LΣ.init(LΣ_params)
    
    params = Parameters(LΣ=LΣ_params, model=model_params)
    optimizers = Optimizers(LΣ=opt_LΣ, model=opt_model)
    opt_state = OptimizerState(LΣ=LΣ_opt_state, model=model_opt_state)

    print(f"L and Σ has {ff2(num_params(params.LΣ))} parameters")
    print(f"Model has {ff2(num_params(params.model))} parameters")
    return forward, params, optimizers, opt_state





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




def ff2(x):
    if type(x) is str: return x
    if onp.abs(x) > 1000 or onp.abs(x) < 0.1: return onp.format_float_scientific(x, 3)
    else: return f"{x:.2f}"


def num_params(params: hk.Params) -> int:
    return len(ravel_pytree(params)[0])