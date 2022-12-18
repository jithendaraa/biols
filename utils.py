import argparse

import numpy as onp
from jax import numpy as jnp
from jax import jit

def load_yaml(configs):
    """
        Takes in a config dict return options as Namespace

        Parameters
        ----------
        configs: dict
            Configuration of the experiment to be run

        Returns
        -------
        opt: argparse.Namespace
    """
    default_config = 'defaults'
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', default=default_config)
    args, remaining = parser.parse_known_args()
    defaults = {}
    names = args.configs

    if isinstance(names, list) is False: names = names.split(' ')
    for name in names:  defaults.update(configs[name])

    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
      arg_type = args_type(value)
      parser.add_argument(f'--{key}', type=arg_type, default=arg_type(value))
    
    opt = parser.parse_args(remaining)
    opt.num_samples = int(opt.pts_per_interv * opt.n_interv_sets) + opt.obs_data
    return opt


def args_type(default):
  def parse_string(x):
    if default is None:
      return x
    if isinstance(default, bool):
      return bool(['False', 'True'].index(x))
    if isinstance(default, int):
      return float(x) if ('e' in x or '.' in x) else int(x)
    if isinstance(default, (list, tuple)):
      return tuple(args_type(default[0])(y) for y in x.split(','))
    return type(default)(x)
  def parse_object(x):
    if isinstance(default, (list, tuple)):
      return tuple(x)
    return x
  return lambda x: parse_string(x) if isinstance(x, str) else parse_object(x)


def set_horseshoe_tau(n, d, degree):
  """
    Sets the parameter for the Horseshoe prior distribution.

    Parameters
    ----------
    n: int
    d: int
    degree: float

    Returns
    -------
    horseshoe_tau: float
      Distribution parameter for Horseshoe
  """
  if ((d - 1) - 2 * degree) == 0:
    p_n_over_n = 2 * degree / (d - 1)
    if p_n_over_n > 1:  p_n_over_n = 1
    horseshoe_tau = p_n_over_n * jnp.sqrt(jnp.log(1.0 / p_n_over_n))
  else:
      horseshoe_tau = (1 / onp.sqrt(n)) * (2 * degree / ((d - 1) - 2 * degree) )
  if horseshoe_tau < 0:   horseshoe_tau = 1 / (2 * d)
  return horseshoe_tau


def set_logdir(opt): 
  logdir = 'logs/' 
  if opt.exp_name[:5] == 'BIOLS':
    logdir += f'BIOLS_({opt.num_nodes})_seed{opt.data_seed}_L_KL{opt.L_KL}'
  return logdir

@jit
def get_joint_dist_params(sigma, W):
    """
      Obtains the parameters for the joint Normal distribution --
      p(z_1, z_2, ... z_d) -- induced by a linear Gaussian SCM.
      Works only for independent, equal variance noise variables.
      z = W.T @ z + eps where eps ~ Normal(0, sigma**2*I)

      Parameters
      ----------
      sigma: list or jnp array
        Corresponds to covariance matrix sigma**2 . I
      
      W: jnp.ndarray (d, d)
        Weighted adjacency matrix of the linear Gaussian SCM

      Returns
      -------
        mu_joint: jnp.ndarray (d,)
          Mean vector for the joint Normal distribution
          p(z_1, z_2, ... z_d)

        Sigma_joint: jnp.ndarray (d, d)
          Covariance matrix for the joint Normal distribution
          p(z_1, z_2, ... z_d)
    """
    dim, _ = W.shape
    Sigma = jnp.diag(sigma)
    inv_matrix = jnp.linalg.inv((jnp.eye(dim) - W))
    
    mu_joint = jnp.array([0.] * dim)
    Sigma_joint = inv_matrix.T @ Sigma @ inv_matrix
    
    return mu_joint, Sigma_joint


def get_lower_elems(L, dim, k=-1):
  return L[jnp.tril_indices(dim, k=k)]


def get_single_kl(p_z_covar, p_z_mu, q_z_covar, q_z_mu, opt):
  """
    Monte carlo based estimate of KL(q || p)
  """
  mu_diff = p_z_mu - q_z_mu
  kl_loss = 0.5 * (   jnp.log(jnp.linalg.det(p_z_covar)) - \
                      jnp.log(jnp.linalg.det(q_z_covar)) - \
                      opt.num_nodes + \
                      jnp.trace( jnp.matmul(jnp.linalg.inv(p_z_covar), q_z_covar) ) + \
                      jnp.matmul(jnp.matmul(jnp.transpose(mu_diff), jnp.linalg.inv(p_z_covar)), mu_diff)
                  )
  return kl_loss

get_mse = lambda recon, x: jnp.mean(jnp.square(recon - x)) 

