import sys
sys.path.append('../modules')

import jax, pdb
import jax.numpy as jnp
import numpy as onp
import haiku as hk
from typing import cast

from modules.NonLinearProjection import init_projection_params 
from modules.SyntheticSCM import LinearGaussianColorSCM

def get_data(rng_key, opt, interv_sets, sd, sigmas, interv_value=None, min_interv_value=None, max_interv_value=None):
    """
        Generates GT observational and interventional samples of causal variables and 
        high dim. vector x, according to interventions in interv_sets.
        Intervention values are sampled in U(min_interv_value, max_interv_value)

        Parameters
        ----------
        rng_key: DeviceArray

        opt: argparse.Namespace
            Experiment configuration.
        
        interv_sets: int
            Number of random intervention sets for generating the interventional data.
            An intervention set refers to a set of nodes which are intervened. 
            E.g., Nodes 3 and 5 are intervened on as in [0, 0, 1, 0, 1]
        
        sd: SyntheticSCM class
            Contained the instantiated SCM which we use to sample causal variables
        
        sigmas: jnp.array (d, )

        interv_value: float (optional, None by default)
            If given, interventions are performed with this value. 
            Else, if None, intervention values are sampled from Uniform distribution

        min_interv_value: float (optional, None by default)
            Should be given if interv_value is None. 
            Refers to min_value in Interv_value ~ Uniform(min_val, max_val)

        max_interv_value: float (optional, None by default)
            Should be given if interv_value is None. 
            Refers to max_value in Interv_value ~ Uniform(min_val, max_val)

        Returns
        -------
        z: jnp.ndarray (opt.num_samples, opt.num_nodes)
            GT samples of causal variables generated from the SCM in sd
        
        interv_nodes: jnp.ndarray int (n, max_cols) 
            where max_cols is the maximum number of nodes intervened.
            E.g., say for datapoint i, nodes 0 and 2 were intervened and d=5, max_cols=4.
            interv_nodes[i] will look like - [0, 2, 5, 5]
            For d=5, the nodes are 0, 1, 2, 3, 4. 5 is a dummy value. 
            interv_nodes will be important later for jittable ancestral sampling.

        x: jnp.ndarray (opt.num_samples, opt.proj_dims)
            High dimensinal GT vector data after projecting causal variables (z) linearly or nonlinearly:
             - Linear projection: z @ P
             - Nonlinear projection: MLP(z)

        P: jnp.ndarray (opt.num_nodes, opt.proj_dims)
            The random projection matrix, if projection (opt.proj) is 'linear'

        interv_values: jnp.ndarray (opt.num_samples, opt.num_nodes)
            2d jax numpy array. Contains intervention values. 
            interv_values[i][j] corresponds to the intervention value for node j, sample i where interv_targets[i][j] == True.
            For all other (i', j') where interv_targets[i'][j'] is False, the value in interv_values[i'][j'] does not matter.
        
    """
    num_interv_data = opt.num_samples - opt.obs_data
    obs_data = sd.simulate_sem(
                                sd.W, 
                                opt.obs_data, 
                                sd.sem_type, 
                                dataset_type="linear",
                                sigmas=sigmas
                            )
    print(sigmas)
    obs_data = cast(jnp.ndarray, obs_data)
    z = obs_data
    P, interv_data = None, None

    num_interv_data = opt.num_samples - opt.obs_data
    interv_targets = onp.zeros((opt.num_samples, opt.num_nodes)).astype(bool) # observational
    num_interv_data = opt.num_samples - opt.obs_data
    interv_values = jax.random.uniform(rng_key, shape=(opt.num_samples, opt.num_nodes), minval=min_interv_value, maxval=max_interv_value)

    if num_interv_data > 0:
        print(f'\nGenerating {opt.interv_type} node interventions...\n')   
        interv_data, interv_targets = get_interv_data(  
                                                        opt, 
                                                        sigmas,
                                                        interv_sets, 
                                                        interv_targets, 
                                                        sd, 
                                                        interv_values 
                                                    )
        z = jnp.concatenate((obs_data, interv_data), axis=0)
    else:
        interv_targets = jnp.zeros((opt.num_samples, opt.num_nodes)).astype(bool)
        interv_values = jnp.zeros((opt.num_samples, opt.num_nodes))

    if opt.proj == 'linear': 
        # Get a random (d x D) projection matrix and project z_{GT} to get X_{GT}
        P = jnp.array(10 * onp.random.rand(opt.num_nodes, opt.proj_dims)) 
        x = z @ P + onp.random.normal(size=(opt.num_samples, opt.proj_dims))
        print(f'z linearly projected from {opt.num_nodes} dims to {opt.proj_dims} dims: {x.shape}\n')          

    elif opt.proj == '3_layer_mlp':  
        key = hk.PRNGSequence(opt.data_seed)
        rng_key = jax.random.PRNGKey(opt.data_seed)
        forward, projection_model_params = init_projection_params(key, opt.num_samples, opt.num_nodes, opt.proj_dims)
        x_mu = forward.apply(projection_model_params, rng_key, opt.proj_dims, z)
        x = x_mu + onp.random.normal(size=(opt.num_samples, opt.proj_dims))
        print(f'X after nonlinear projection of z from {opt.num_nodes} dims to {opt.proj_dims} dims: {x.shape}')

    x = jnp.array(x)
    interv_nodes = get_interv_nodes(opt.num_samples, opt.num_nodes, interv_targets)
    return z, interv_nodes, x, P, interv_targets, interv_values


def get_interv_data(opt, sigmas, interv_sets, interv_targets, sd, interv_values):
    """
        Generates `interv_sets` sets of interventional data, totally having (opt.num_samples - opt.obs_data) 
        interventional data points.
        Each set has `opt.pts_per_interv` data points with random uniform intervention values
        Each intervention set is decided by first randomly sampling number of nodes to intervene on, say k,
        and then picking k nodes at random from [0, d-1], where d is opt.num_nodes. 
        If opt.interv_type == 'single', k is set to 1.
        
        Parameters
        ----------
        opt: argparse.Namespace
            Experiment configuration.

        interv_sets: int
            Number of random intervention sets for generating the interventional data.
            An intervention set refers to a set of nodes which are intervened. 
            E.g., Nodes 3 and 5 are intervened on as in [0, 0, 1, 0, 1]
        
        sigmas: np.array
            Array of sigmas denoting the noise variables' std deviation.

        interv_targets: jnp.ndarray (opt.num_samples, opt.num_nodes)
            2d boolean jax numpy array. 
            interv_targets[i][j] == 1 if node j was intervened on in sample i.
        
        sd: SyntheticSCM class
            Contained the instantiated SCM which we use to sample causal variables
        
        interv_values: jnp.ndarray
            If given, interventions are performed with this value. 
            Else, if None, intervention values are sampled from Uniform distribution
        
        Returns
        -------
        interv_data: jnp.ndarray (opt.num_samples - opt.obs_data, opt.num_nodes)

        interv_targets: jnp.ndarray (opt.num_samples, opt.num_nodes)
            2d boolean jax numpy array. 
            interv_targets[i][j] == 1 if node j was intervened on in sample i
    """
    data_per_interv_set = []
    num_interv_data = opt.num_samples - opt.obs_data
    interv_data_pts_per_set = int(num_interv_data / interv_sets)

    for i in range(interv_sets):
        if opt.interv_type == 'single':     interv_k_nodes = 1
        elif opt.interv_type == 'multi':    interv_k_nodes = onp.random.randint(1, opt.num_nodes)

        intervened_node_idxs = onp.random.choice(opt.num_nodes, interv_k_nodes, replace=False)
        interv_value = interv_values[opt.obs_data + i * interv_data_pts_per_set : opt.obs_data + (i+1) * interv_data_pts_per_set]
        interv_data = sd.intervene_sem( 
                                        sd.W, 
                                        interv_data_pts_per_set, 
                                        opt.sem_type,
                                        sigmas=sigmas, 
                                        idx_to_fix=intervened_node_idxs, 
                                        values_to_fix=interv_value
                                    )

        data_per_interv_set.append(interv_data)
        interv_targets[opt.obs_data + i * interv_data_pts_per_set : opt.obs_data + (i+1) * interv_data_pts_per_set, intervened_node_idxs] = True

    interv_data = jnp.array(data_per_interv_set).reshape(num_interv_data, opt.num_nodes)
    return interv_data, jnp.array(interv_targets)


def get_interv_nodes(n, d, interv_targets):
  """
  Given intervention targets, returns intervened nodes as a 
  jnp.ndarray of shape (n, max_cols) where max_cols is the 
  maximum number of nodes intervened.
  
  E.g., say for datapoint i, nodes 0 and 2 were intervened and d=5, max_cols=4.
  interv_nodes[i] will look like - [0, 2, 5, 5]
  For d=5, the nodes are 0, 1, 2, 3, 4. 5 is a dummy value. 
  interv_nodes will be important later for jittable ancestral sampling.

  Parameters
  ----------
  n: int
    Number of data samples
  
  d: int
    Number of nodes
  
  interv_targets: jnp.ndarray of type bool (n, d)
    interv_targets[i, j] is True when node j in sample i has been intervened on.

  Returns
  -------
  interv_nodes: jnp.ndarray int (n, max_cols) 
  """
  max_cols = jnp.max(interv_targets.sum(1))
  data_idx_array = jnp.array([jnp.arange(d + 1)] * n)
  dummy_interv_targets = jnp.concatenate((interv_targets, jnp.array([[False]] * n)), axis=1)
  interv_nodes = onp.split(data_idx_array[dummy_interv_targets], interv_targets.sum(1).cumsum()[:-1])
  interv_nodes = jnp.array([jnp.concatenate(( interv_nodes[i], jnp.array( [d] * int(max_cols - len(interv_nodes[i])) ))) for i in range(n)]).astype(int)
  return interv_nodes

