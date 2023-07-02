import haiku as hk
import jax
import jax.numpy as jnp
import optax, pdb

class ObsClassifier(hk.Module):
    def __init__(self, proj_dims, encoder_out):
        super().__init__()

        encoder_in = max(32, proj_dims // 2)

        self.encoder = hk.Sequential([
            hk.Linear(encoder_in, name='interv_encoder_0', with_bias=True), jax.nn.gelu,
            hk.Linear(32, name='interv_encoder_1', with_bias=True), jax.nn.gelu,
            hk.Linear(32, name='interv_encoder_2', with_bias=True), jax.nn.gelu,
            hk.Linear(32, name='interv_encoder_3', with_bias=True), jax.nn.gelu,
            hk.Linear(32, name='interv_encoder_5', with_bias=True), jax.nn.gelu,
            hk.Linear(32, name='interv_encoder_6', with_bias=True), jax.nn.gelu,
            hk.Linear(encoder_out, name='interv_encoder_4', with_bias=True)
        ])

    def __call__(self, x):
        logits = self.encoder(x)
        return logits[:, -1]


def classifier_fwd_fn(proj_dims, encoder_out, X):
    model = ObsClassifier(proj_dims, encoder_out)
    return model(X)


def init_classifier_model(hk_key, lr, proj_dims, encoder_out, X):
    forward = hk.transform(classifier_fwd_fn)
    next_key = next(hk_key)
    model_params = forward.init(next_key, proj_dims, encoder_out, X)
    model_layers = [optax.scale_by_belief(eps=1e-8), optax.scale(-lr)]
    opt_model = optax.chain(*model_layers)
    model_opt_params = opt_model.init(model_params)
    return forward, model_params, opt_model, model_opt_params