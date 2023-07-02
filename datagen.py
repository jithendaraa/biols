import sys
sys.path.append('../modules')

import torch
import jax, pdb
import jax.numpy as jnp
from jax.scipy.linalg import expm

import numpy as onp
from typing import cast
from tqdm import tqdm

from modules.NonLinearProjection import init_projection_params 
from modules.SyntheticSCM import SyntheticSCM
from tensorflow_probability.substrates.jax.distributions import Normal

class SyntheticDatagen(SyntheticSCM):
    def __init__(self, 
                 data_seed, 
                 hk_key,
                 rng_key, 
                 num_nodes,
                 degree,
                 interv_type,
                 proj_dims, 
                 projection,
                 decoder_sigma,
                 interv_value_sampling='gaussian', # gaussian N(0, 1) or uniform
                 datagen_type='default', # default or weakly_supervised 
                 sem_type='linear-gauss', 
                 graph_type='erdos-renyi',
                 dataset_type='linear', 
                 min_interv_value=-5.,
                 identity_perm=False,
                 edge_threshold=0.3
                ):

        assert edge_threshold == 0.3

        log_sigma_W = jax.random.uniform(rng_key, shape=(num_nodes,), minval=0, maxval=jnp.log(2))
        self.num_nodes = num_nodes
        self.proj_dims = proj_dims
        self.projection = projection
        self.set_projection_matrix(rng_key, hk_key)

        super(SyntheticDatagen, self).__init__(
            d = num_nodes,
            graph_type = graph_type,
            degree = 2 * degree,
            sem_type = sem_type,
            sigmas=jnp.exp(log_sigma_W),
            dataset_type=dataset_type,
            data_seed=data_seed,
            identity_perm=identity_perm
        )

        assert graph_type in ['erdos-renyi']
        assert sem_type in ['linear-gauss']
        assert interv_type in ['single', 'multi']
        assert projection in ['linear', '3_layer_mlp', 'SON']

        self.graph = jnp.where(jnp.abs(self.W) < edge_threshold, 0, 1)
        
        self.data_seed = data_seed
        self.interv_type = interv_type
        self.decoder_sigma = decoder_sigma
        self.min_interv_value = min_interv_value

        self.sigmas = jnp.exp(log_sigma_W) # std dev of the SCM noise variables
        means = jnp.zeros(num_nodes)
        self.noise_dist = Normal(loc=means, scale=self.sigmas)
        
        self.interv_noise_dist = Normal(loc=means, scale=jnp.ones_like(means) * 0.5)

        # scale of noise variables eps_i
        self.log_sigma_W = log_sigma_W
        self.datagen_type = datagen_type
        self.interv_value_sampling = interv_value_sampling

    def set_projection_matrix(self, rng_key, hk_key):
        if self.projection == 'SON': 
            assert self.proj_dims == self.num_nodes
            std = 0.05 # Following SO(n) projection in ILCM paper
            entries = self.num_nodes * (self.num_nodes - 1) // 2
            self.coeff = std * jax.random.normal(rng_key, shape=(entries,))

            o = jnp.zeros((self.num_nodes, self.proj_dims))
            i, j = jnp.triu_indices(self.num_nodes, k=1)
            o = o.at[i, j].set(-self.coeff)
            o = o.at[j, i].set(self.coeff)

            proj_matrix = torch.matrix_exp(torch.from_numpy(onp.array(o))).numpy()
            self.proj_matrix = jnp.array(proj_matrix)
        
        elif self.projection == 'linear':
            self.proj_matrix = jax.random.uniform(
                rng_key, 
                shape=(self.num_nodes, self.proj_dims), 
                minval=-2., 
                maxval=2.
            )
        
        elif self.projection == '3_layer_mlp':
            self.hk_key = hk_key
            self.forward_fn, self.projection_model_params = init_projection_params(
                hk_key, 
                self.num_nodes, 
                self.proj_dims
            )

    def get_interv_nodes(self, num_samples, num_nodes, interv_targets):
        """
        Returns a list of intervention nodes for each sample

        Parameters
        ----------
        num_samples: int

        num_nodes: int

        interv_targets: (num_samples, num_nodes) array of booleans

        Returns
        -------
        interv_nodes: (num_samples, max_cols) array of ints
        """
        max_cols = jnp.max(interv_targets.sum(1))
        data_idx_array = jnp.arange(num_nodes + 1)[None, :].repeat(num_samples, axis=0)
        dummy_interv_targets = jnp.concatenate((interv_targets, jnp.array([[False]] * num_samples)), axis=1)
        interv_nodes = onp.split(data_idx_array[dummy_interv_targets], interv_targets.sum(1).cumsum()[:-1])
        interv_nodes = jnp.array([jnp.concatenate(( interv_nodes[i], jnp.array( [num_nodes] * int(max_cols - len(interv_nodes[i])) ))) for i in range(num_samples)]).astype(int)
        return interv_nodes

    def generate_observational_z(self):
        z_observational = self.simulate_sem(
                    self.W,
                    self.num_obs_samples,
                    self.sem_type,
                    dataset_type=self.dataset_type,
                    sigmas=self.sigmas
                )
        z_observational = cast(jnp.ndarray, z_observational)
        return z_observational

    def generate_interventional_z(self, rng_key, num_interv_sets):
        # print(f'\nGenerating {self.interv_type}-target interventions...\n')   
        data_per_interv_set = []
        interv_targets = jnp.zeros((self.num_samples, self.num_nodes)).astype(bool) # Initialise everything as observational; will be modified when generating interventional z
        if self.interv_type == 'single':    
            interv_k_nodes = 1

        if self.interv_value_sampling == 'gaussian':
            # print("Interventional values ~ N(0, 1)")
            interv_values = jax.random.normal(rng_key, shape=(self.num_samples, self.num_nodes))
            
        elif self.interv_value_sampling == 'uniform':
            # print(f"Interventional values ~ U({self.min_interv_value}, {-self.min_interv_value})")
            interv_values = jax.random.uniform(rng_key, shape=(self.num_samples, self.num_nodes), minval=self.min_interv_value, maxval=-self.min_interv_value)

        for i in range(num_interv_sets):
            if self.interv_type == 'multi':
                # How many nodes to intervene on for multi-target intervention?
                interv_k_nodes = onp.random.randint(1, self.num_nodes) 

            start_intervened_samples = self.num_obs_samples + i * self.num_interv_samples_per_set 
            end_intervened_samples = self.num_obs_samples + (i+1) * self.num_interv_samples_per_set
            intervened_node_idxs = jax.random.choice(rng_key, jnp.arange(self.num_nodes), (interv_k_nodes,), replace=False)
            rng_key, _ = jax.random.split(rng_key, 2)
            
            interv_targets = interv_targets.at[start_intervened_samples : end_intervened_samples, intervened_node_idxs].set(True)
            interv_value = interv_values[ start_intervened_samples : end_intervened_samples ]
            
            interv_data = self.intervene_sem( 
                self.W, 
                self.num_interv_samples_per_set, 
                self.sem_type,
                sigmas=self.sigmas, 
                idx_to_fix=intervened_node_idxs, 
                values_to_fix=interv_value
            )

            data_per_interv_set.append(interv_data)
        
        z_interventional = jnp.array(data_per_interv_set).reshape(self.num_interv_samples, self.num_nodes)
        return z_interventional, interv_targets, interv_values
    
    def project(self, rng_key, z_samples):
        if self.projection in ['linear', 'SON']: 
            x_mu = z_samples @ self.proj_matrix
        
        elif self.projection == '3_layer_mlp':
            # MLP-based nonlinear projection of z_samples \in \mathbb{R}^d to get x_mu \in \mathbb{R}^D 
            x_mu = self.forward_fn.apply(
                self.projection_model_params, 
                rng_key, 
                self.proj_dims, 
                z_samples
            )

        x_samples = x_mu + jax.random.normal(rng_key, shape=x_mu.shape) * self.decoder_sigma
        return x_samples

    def sample_default(self, rng_key):
        rng_key, _ = jax.random.split(rng_key)
        z_samples = jnp.zeros((self.num_samples, self.num_nodes))
        print()
        print(f'Default sampling: {self.interv_type}-target interventions...')
        
        # Generate z samples 
        obs_z_samples = self.generate_observational_z()
        interv_z_samples, interv_targets, interv_values = self.generate_interventional_z(rng_key)
        
        z_samples = z_samples.at[:self.num_obs_samples, :].set(obs_z_samples)
        z_samples = z_samples.at[self.num_obs_samples:, :].set(interv_z_samples)

        x_samples = self.project(rng_key, z_samples)
        interv_nodes = self.get_interv_nodes(
            self.num_samples, 
            self.num_nodes, 
            interv_targets
        )

        return x_samples, z_samples, interv_nodes, interv_targets, interv_values
    
    def sample_weakly_supervised(self, rng_key, n_pairs, num_interv_sets,
        return_interv_values=False, fix_noise=True, no_interv_noise=False, return_interv_noise=False):
        """
            TODO
        """
        assert n_pairs % num_interv_sets == 0

        if self.interv_value_sampling == 'gaussian':
            # print("Interventional values ~ N(0, 1)")
            interv_values = jax.random.normal(rng_key, shape=(n_pairs, self.num_nodes))
            
        elif self.interv_value_sampling == 'uniform':
            # print(f"Interventional values ~ U({self.min_interv_value}, {-self.min_interv_value})")
            interv_values = jax.random.uniform(rng_key, shape=(n_pairs, self.num_nodes), minval=self.min_interv_value, maxval=-self.min_interv_value)
        
        elif self.interv_value_sampling == 'zeros':
            interv_values = jnp.zeros((n_pairs, self.num_nodes))

        n_samples_per_interv_set = n_pairs // num_interv_sets
        interv_targets_z2 = jnp.zeros((n_pairs, self.num_nodes)).astype(bool) 
        print()
        print(f'Weakly-supervised sampling: {self.interv_type}-target interventions...')

        interv_k_nodes = 1
        intervention_labels = []
        for i in tqdm(range(num_interv_sets)):
            start_intervened_samples = i * n_samples_per_interv_set
            end_intervened_samples = (i+1) * n_samples_per_interv_set

            if self.interv_type == 'multi': # How many nodes to intervene on for multi-target intervention?
                interv_k_nodes = onp.random.randint(1, self.num_nodes)
            
            intervened_node_idxs = jax.random.choice(rng_key, jnp.arange(self.num_nodes), shape=(interv_k_nodes,), replace=False)
            interv_targets_z2 = interv_targets_z2.at[start_intervened_samples : end_intervened_samples, intervened_node_idxs].set(True)
            rng_key, _ = jax.random.split(rng_key)

            if self.interv_type == 'single':
                intervention_labels += [int(intervened_node_idxs[0])] * n_samples_per_interv_set
            elif self.interv_type == 'multi':
                intervention_labels = None

        if no_interv_noise:
            intervention_noise = jnp.zeros((n_pairs, self.num_nodes))
        else:
            intervention_noise = self.interv_noise_dist.sample(seed=rng_key, sample_shape=(n_pairs,)) # noise used for the intervened-upon variables
            rng_key, _ = jax.random.split(rng_key)

        if fix_noise:
            # Sample noise
            noise = self.noise_dist.sample(seed=rng_key, sample_shape=(n_pairs,)) # noise variables used for the data pre intervention
            rng_key, _ = jax.random.split(rng_key)
            z1_noise, z2_noise = noise, noise

        else:
            # Sample noise
            z1_noise = self.noise_dist.sample(seed=rng_key, sample_shape=(n_pairs,)) # noise variables used for the data pre intervention
            rng_key, _ = jax.random.split(rng_key)
            z2_noise = self.noise_dist.sample(seed=rng_key, sample_shape=(n_pairs,)) # noise variables used for the data pre intervention
            rng_key, _ = jax.random.split(rng_key)
        
        z1 = self.sample_z_given_noise(
            z1_noise,
            self.W,
            self.sem_type, 
        )

        z2 = self.sample_z_given_noise(
            z2_noise,
            self.W,
            self.sem_type,
            interv_targets=interv_targets_z2, 
            interv_noise=intervention_noise,
            interv_values=interv_values
        )

        z1 = jnp.array(z1).reshape(n_pairs, self.num_nodes)
        z2 = jnp.array(z2).reshape(n_pairs, self.num_nodes)
        x1 = self.project(rng_key, z1)
        x2 = self.project(rng_key, z2)
        print(f'{self.projection} projection from {self.num_nodes} dims to {self.proj_dims} dims')  

        if self.interv_type == 'multi':
            intervention_labels = self.get_interv_nodes(len(interv_targets_z2), self.num_nodes, interv_targets_z2)
        
        assert (z2[interv_targets_z2 == True] == interv_values[interv_targets_z2 == True] + intervention_noise[interv_targets_z2 == True]).all()

        return_items = [
            x1.astype(jnp.float32),
            x2.astype(jnp.float32),
            z1.astype(jnp.float32),
            z2.astype(jnp.float32),
            jnp.array(intervention_labels).astype(jnp.int32),
            interv_targets_z2.astype(jnp.int32)
        ]

        if return_interv_values:
            return_items += [interv_values]

        if return_interv_noise:
            return_items += [intervention_noise]
        
        return tuple(return_items)


