import sys, pdb
sys.path.append('..')

from typing import cast
import haiku as hk

import jax
import jax.numpy as jnp
import jax.random as rnd
from jax import vmap, random, lax

from tensorflow_probability.substrates.jax.distributions import Normal, RelaxedBernoulli, Gumbel, Exponential
from scipy.special import softmax
from modules.GumbelSinkhorn import GumbelSinkhorn


class BIOLS_Interv(hk.Module):
    def __init__(self, d, posterior_samples, eq_noise_var, tau, hidden_size, 
                max_deviation, learn_P, learn_L, proj_dims, proj, log_stds_max=10.0, 
                logit_constraint=10, P=None, L=None):
        """
            The BIOLS model to estimate SCM from vector data
        """
        super().__init__()
        self._set_vars(d, posterior_samples, eq_noise_var, tau,
                        hidden_size, max_deviation, learn_P, learn_L,
                        log_stds_max, logit_constraint, proj_dims, P, L)
        
        self.ds = GumbelSinkhorn(self.d, noise_type="gumbel", tol=max_deviation)

        self.encoder = hk.Sequential([
            hk.Linear(proj_dims, with_bias=False), jax.nn.gelu,
            hk.Linear(proj_dims // 2, with_bias=False), jax.nn.gelu,
            hk.Linear(32, with_bias=False), jax.nn.gelu,
            hk.Linear(32, with_bias=False), jax.nn.gelu,
            hk.Linear(d, with_bias=False), jax.nn.sigmoid
        ])

        if self.learn_P:
            self.p_logits_model = hk.Sequential([
                                hk.Flatten(), 
                                hk.Linear(hidden_size), jax.nn.gelu,
                                hk.Linear(hidden_size), jax.nn.gelu,
                                hk.Linear(d * d)
                            ])

        if proj == 'linear':
            self.decoder = hk.Sequential([
                hk.Flatten(), hk.Linear(proj_dims, with_bias=False)
            ])
        else:
            self.decoder = hk.Sequential([
                    hk.Flatten(), 
                    hk.Linear(16, with_bias=False), jax.nn.gelu,
                    hk.Linear(64, with_bias=False), jax.nn.gelu,
                    hk.Linear(64, with_bias=False), jax.nn.gelu,
                    hk.Linear(64, with_bias=False), jax.nn.gelu,
                    hk.Linear(proj_dims, with_bias=False)
                ])

    def _set_vars(self, d, posterior_samples, eq_noise_var, tau,
                    hidden_size, max_deviation, learn_P, learn_L, 
                    log_stds_max, logit_constraint, proj_dims, P=None, 
                    L=None):
        """
            Sets important variables/attributes for this class.
        """
        self.d = d
        self.l_dim = d * (d - 1) // 2
        self.posterior_samples = posterior_samples
        self.eq_noise_var = eq_noise_var
        self.tau = tau
        self.hidden_size = hidden_size
        self.max_deviation = max_deviation
        self.learn_P = learn_P
        self.learn_L = learn_L
        self.log_stds_max = log_stds_max
        self.logit_constraint = logit_constraint
        self.proj_dims = proj_dims
        self.P = P
        self.L = L
        if eq_noise_var:    self.noise_dim = 1
        else:               self.noise_dim = d

    def lower(self, theta):
        """
            Given d(d-1)/2 parameters for theta, forms a strictly lower-triangular matrix

            Parameters
            ----------
            theta: jnp.ndarray (d(d-1)/2) elements

            Returns
            -------
            out: jnp.ndarray (d, d)
        """
        out = jnp.zeros((self.d, self.d))
        out = out.at[jnp.tril_indices(self.d, -1)].set(theta)
        return out

    def sample_W(self, L, P):
        """
            Compute the weighted adjacency matrix given edge weights L 
            and permutation P.

            Parameters
            ----------
                L: jnp.ndarray(d, d)

                P: jnp.ndarray(d, d)
                    The permuatation matrix that defines the node ordering

            Returns
            -------
                W: jnp.ndarray(d, d)
        """
        W = (P @ L @ P.T).T
        return W

    def sample_L_and_Σ(self, rng_key, LΣ_params):
        """
            Performs sampling (L, Σ) ~ q_ϕ(L, Σ) 
                where q_ϕ is a Normal if self.eq_noise_var is True
                else q_ϕ is a Normalizing Flow (not implemented)
            L has d * (d - 1) / 2 terms
            Σ is a single term referring to noise variance on each node 
            
            Parameters
            ----------
                rng_key: DeviceArray
                    Random number generator key

                LΣ_params:  jnp.ndarray (d*(d-1) + 2, )
                    Contains mean and std for Σ and edges in L.  
                    Shape (d*(d-1) + 2d, ) if we are inferring for non-equal noise variance (not implemented)
                
            Results
            -------
                full_l_batch: jnp.ndarray (batch_size, 1 + d(d-1)/2)
                    Contains `batch_size` samples of concatenated(L_i, Σ_i) from the 
                    posterior q(L, Σ).

                full_log_prob_l: jnp.ndarray (batch_size, )
                    Log prob of each (L_i, Σ_i) sample under the posterior q(L, Σ)
                    across samples.
        """
        LΣ_params = cast(jnp.ndarray, LΣ_params)
        # Obtain mean and covariance of the Normal distribution to sample L and Σ from
        means, log_stds = LΣ_params[:self.l_dim + self.noise_dim], LΣ_params[self.l_dim + self.noise_dim :]
        if self.log_stds_max is not None:    
            # cap log_stds to (-self.log_stds_max, log_stds_max), so that std = exp(log_stds) doesn't explode.
            log_stds = jnp.tanh(log_stds / self.log_stds_max) * self.log_stds_max 
        
        # Sample (L, Σ) from the Normal
        l_distribution = Normal(loc=means, scale=jnp.exp(log_stds))
        full_l_batch = l_distribution.sample(seed=rng_key, sample_shape=(self.posterior_samples,))
        full_l_batch = cast(jnp.ndarray, full_l_batch)

        # log likelihood for q_ϕ(L, Σ)
        full_log_prob_l = jnp.sum(l_distribution.log_prob(full_l_batch), axis=1)  
        full_log_prob_l = cast(jnp.ndarray, full_log_prob_l)
        return full_l_batch, full_log_prob_l

    def get_P_logits(self, LΣ_samples):
        """
            Computes the P_logits = T = MLP_ϕ(L, Σ); MLP_ϕ is self.p_logits_model

            Parameters
            ----------
                LΣ_samples: jnp.ndarray (batch_size, d(d-1)/2 + 1)
                    `batch_size` samples of (L, Σ)

            Returns
            -------
                p_logits: jnp.ndarray (batch_size, d, d)
        """
        p_logits = self.p_logits_model(LΣ_samples)
        if self.logit_constraint is not None:   # Want to map -inf to -logit_constraint, inf to +logit_constraint
            p_logits = jnp.tanh(p_logits / self.logit_constraint) * self.logit_constraint
        return p_logits.reshape((-1, self.d, self.d))

    def get_P(self, rng_key, full_l_batch, hard):
        """
            Given batches of \hat{L} and \hat{Σ}, first calculate the logits 
            to sample permutation, then (soft or hard) sample permutation \hat{P} 
            from q{P | L, Σ} using Gumbel-Sinkhorn or Hungarian algorithm.

            1.  T <- MLP_{\phi(T)}(\hat{L}, \hat{Σ})
            2.  soft P̃ = Sinkhorn( (T+γ)/𝜏 ) 
            3.  If computing hard P, compute soft P̃ and get hard P = Hungarian(P̃) 

            Parameters
            ----------
                rng_key: DeviceArray
                    Random number generator key
                
                full_l_batch: jnp.ndarray (batch_size, 1 + d(d-1)/2)
                    Contains `batch_size` samples of concatenated(L_i, Σ_i) from the 
                    posterior q(L, Σ).
                
                hard: bool
                    If false, soft P̃ is sampled.

            Returns
            -------
                batched_P: jnp.ndarray (batch_size, d, d)
                    `batch_size` samples of P_i from q(P | L, Σ)
                
                batched_P_logits: jnp.ndarray (batch_size, d, d)
                    Logits for sampling each permutation P_i
        """

        if self.learn_P:
            # Compute logits T = MLP_{ϕ(T)}(L, Σ) for sampling from q_{ϕ}(P | L, Σ)
            batched_P_logits = self.get_P_logits(full_l_batch)

            if hard is False:
                # Compute soft P̃ = Sinkhorn( (T+γ)/𝜏 ) 
                batched_P = self.ds.sample_soft_batched_logits(batched_P_logits, self.tau, rng_key)
            elif hard is True:
                # Compute soft P̃ and get hard P = Hungarian(P̃) 
                batched_P = self.ds.sample_hard_batched_logits(batched_P_logits, self.tau, rng_key)

        elif self.learn_P is False:
            # if not learning P, use GT value
            batched_P = self.P[jnp.newaxis, :].repeat(self.posterior_samples, axis=0) 
            batched_P_logits = None

        return batched_P, batched_P_logits

    def eltwise_ancestral_sample(self, W, P, eps_std, rng_key, interv_target=None, interv_values=None):
        """
            Given a weighted adjacency matrix, permutation, and std deviation 
            of noise variables (eps_std), sample from Normal(0, eps_std) d times 
            to obtain exogenous noise vector \epsilon = {\epsilon_1, ... \epsilon_d}. 
            The traverse topologically and ancestral sample to predict 
            causal variables vector {z_1,...,z_d}. 
            Ancestral samples according to intervention targets and values

            Parameters
            ----------
            W: jnp.ndarray (d, d)
                Weighted adjacency matrix of the SCM
            
            P: jnp.ndarray (d, d)
                Permutation matrix

            eps_std: float
                Standard deviation of the noise variables.
                Noise epsilon_i will be sampled from N(0, eps_std).

            interv_target: jnp.ndarray (d)

            interv_values: jnp.ndarray (d)
                For every intervened node, or every interv_targets[j], 
                interv_values[interv_targets[j]] contains the value of intervention
                for node interv_targets[j].

            Returns
            -------
            sample: jnp.ndarray (d, )
                Predicted causal variables vector {z_1,...,z_d}
        """
        sample = jnp.zeros((self.d+1))
        ordering = jnp.arange(0, self.d)
        swapped_ordering = ordering[jnp.where(P, size=self.d)[1].argsort()]
        
        # Getting exogenous noise vector \epsilon = {\epsilon_1, ... \epsilon_d}, from eps_std
        noise_terms = jnp.multiply(eps_std, random.normal(rng_key, shape=(self.d,)))

        # Traverse node topologically and ancestral sample
        for j in swapped_ordering:
            mean = sample[:-1] @ W[:, j]
            ancestral_value = mean + noise_terms[j]
            intervened_value = interv_values[j]
            node_value = (1 - interv_target[j]) * ancestral_value + (interv_target[j] * intervened_value)
            sample = sample.at[j].set(node_value)

        return sample[:-1]

    def ancestral_sample(self, rng_key, W, P, eps_std, interv_targets, interv_values):
        """
            Given a weighted adjacency matrix, permutation, and std deviation 
            of noise variables (eps_std), sample from Normal(0, eps_std) d times 
            to obtain exogenous noise vector \epsilon = {\epsilon_1, ... \epsilon_d}. 
            Traverse topologically and ancestral sample to predict 
            Z = {z^{i}_1,...,z^{i}_d} i={1...n}, for the entire dataset 
            D={X^{i}} i={1...n} according to intervention targets \mathcal{I} 
            and intervention values.

            Parameters
            ----------
            rng_key: jax.random.PRNGKey

            W: jnp.ndarray (d, d)
                Weighted adjacency matrix of the SCM

            P: jnp.ndarray (d, d)
                Permutation matrix

            eps_std: float
                Standard deviation of the noise variables.
                Noise epsilon_i will be sampled from N(0, eps_std)

            interv_targets: jnp.ndarray (n, max_intervs)
                max_intervs refers to maximum number of nodes intervened on
                in any one sample of the dataset, i.e, any X^{i}. 

            interv_values: jnp.ndarray (n, d)
                For every intervened node, or every interv_targets[i][j], 
                interv_values[i, interv_targets[i][j]] contains the value of intervention
                for node interv_targets[i][j].

            Returns
            -------
            samples: jnp.ndarray 
        """
        interv_values = jnp.concatenate( ( interv_values, jnp.zeros((len(interv_targets), 1)) ), axis=1)
        rng_keys = rnd.split(rng_key, len(interv_targets))
        samples = vmap(self.eltwise_ancestral_sample, (None, None, None, 0, 0, 0), (0))(W, P, eps_std, rng_keys, interv_targets, interv_values)
        return samples

    def __call__(self, rng_key, X, interv_values, LΣ_params, hard):
        """
            Forward pass of BIOLS:
                1. Draw (L_i, Σ_i) ~ q_ϕ(L, Σ) 
                2. Draw P_i ~ q_ϕ(P | L, Σ)
                3. W_i = (P_i L_i P_i.T).T for every posterior sample of (P, L)
                4. z_i = AncestralSample(W_i, Σ_i) 
                5. X_i = Decoder_MLP(z_i)
            
            Parameters
            ----------
                rng_key: DeviceArray
                    Random number generator key

                interv_targets: jnp.ndarray (n, max_intervs)
                    max_intervs refers to maximum number of nodes intervened on
                    in any one sample of the dataset, i.e, any X^{i}. 

                interv_values: jnp.ndarray (n, d)
                    For every intervened node, or every interv_targets[i][j], 
                    interv_values[i, interv_targets[i][j]] contains the value of intervention
                    for node interv_targets[i][j].

                LΣ_params:  jnp.ndarray (d*(d-1) + 2, )
                    Contains mean and std for edges in L and the noise_sigma we are inferring over
                    (d*(d-1) + 2d, ) if we are inferring for non-equal noise variance

                hard: bool
                    Hard or soft sampling of permutation P

            Returns
            -------
                pred_X: jnp.ndarray (batch_size, num_samples, proj_dims)
                    \hat{X}: Predicted of ground truth X.
                    Decoded `batch_size` samples of `batched_qz_samples`.
                    Can be thought of as sampling from p(X | P, L, Σ).

                P_samples: jnp.ndarray (batch_size, d, d)
                    `batch_size` samples of P_i from q(P | L, Σ)

                batched_P_logits: jnp.ndarray (batch_size, d, d)
                    Logits for sampling each permutation P_i

                batched_qz_samples: jnp.ndarray (batch_size, num_samples, d)
                    `batch_size` samples from q{Z | P, L, Σ}

                full_l_batch: jnp.ndarray (batch_size, 1 + d(d-1)/2)
                    Contains `batch_size` samples of concatenated(L_i, Σ_i) from the 
                    posterior q(L, Σ).
                
                full_log_prob_l: jnp.ndarray (batch_size, )
                    Log prob of each (L_i, Σ_i) sample under the posterior q(L, Σ)
                    across samples.

                L_samples: jnp.ndarray (batch_size, d, d)
                    `batch_size` samples of the posterior edge weights L

                W_samples: jnp.ndarray (batch_size, d, d)
                    `batch_size` weighted adjacency matrices constructed 
                    as W_i = (P_i L_i P_i.T).T from posterior samples P_i, L_i.

                log_noise_std_samples: jnp.ndarray(batch_size, 1)
                    `batch_size` samples of log sigma_i which is used to obtain
                    Σ_i as diag(exp(2 * log sigma_i)) 
        """
        target_bern_means = self.encoder(X).reshape(-1)
        interv_target_dist = RelaxedBernoulli(temperature=1e-6, probs=target_bern_means)
        interv_target_samples = interv_target_dist.sample(seed=rng_key).reshape(interv_values.shape)

        # Draw (L, Σ) ~ q_ϕ(L, Σ) 
        full_l_batch, full_log_prob_l = self.sample_L_and_Σ(rng_key, LΣ_params)
        log_noise_std_samples = full_l_batch[:, -self.noise_dim:]  # (posterior_samples, 1)
        
        if self.learn_L:
            L_samples = vmap(self.lower, in_axes=(0))(full_l_batch[:,  :self.l_dim]) # (posterior_samples, d, d)
        else: 
            L_samples = jnp.tile(self.L[None, :, :], (self.posterior_samples, 1, 1))

        # Draw P ~ q_ϕ(P | L, Σ) if we are learning P, else use GT value
        P_samples, batched_P_logits = self.get_P(rng_key, full_l_batch, hard) # (posterior_samples, d, d) 

        # W = (PLP.T).T for every posterior sample of (P, L) 
        W_samples = vmap(self.sample_W, (0, 0), (0))(L_samples, P_samples)  # (posterior_samples, d, d)
        rng_keys = rnd.split(rng_key, self.posterior_samples)
        
        # z ~ q(Z | P, L, Σ)
        vmapped_ancestral_sample = vmap(self.ancestral_sample, (0, 0, 0, 0, None, None), (0))
        batched_qz_samples = vmapped_ancestral_sample(  
                                                        rng_keys,
                                                        W_samples,
                                                        P_samples,
                                                        jnp.exp(log_noise_std_samples),
                                                        interv_target_samples.reshape(interv_values.shape),
                                                        interv_values
                                                    )

        # Stochastic decoder -> causal variables z to predict x                                  
        Mu_X = vmap(self.decoder, (0), (0))(batched_qz_samples) # (posterior_samples, n, proj_dims)
        pred_X = Mu_X 
        # + jnp.multiply(1.0, random.normal(rng_key, shape=(Mu_X.shape)))

        return (pred_X, P_samples, batched_P_logits, batched_qz_samples, full_l_batch, 
                full_log_prob_l, L_samples, W_samples, log_noise_std_samples, interv_target_samples.reshape(interv_values.shape).astype(int))