# BIOLS: Bayesian Inference over Latent SCMs
Official implementation of [Learning Latent Structural Causal Models](https://arxiv.org/abs/2210.13583) for approximate Bayesian inference over causal variables, structure and parameters of latent SCMs from low-level data.

## Installation

## Important hyperparameters

## Usage

### 1. Learning from high-dimensional vectors obtained by linear projection of causal variables (given a fixed node ordering): 
- Observational data: `python biols_vector_data.py --config defaults linear_vector_obs_learn_L`
- Single intervention targets: `python biols_vector_data.py --config defaults linear_vector_single_interv_learn_L`
- Multi intervention targets: `python biols_vector_data.py --config defaults linear_vector_multi_interv_learn_L`

### 2. Learning from high-dimensional vectors obtained by linear projection of causal variables (learning over node orderings): 
- Observational data: `python batched_biols_vector_data.py --config defaults batched_linear_vector_obs_learn_SCM`
- Single intervention targets: `python batched_biols_vector_data.py --config defaults batched_linear_vector_single_interv_learn_SCM`
- Multi intervention targets: `python batched_biols_vector_data.py --config defaults batched_linear_vector_multi_interv_learn_SCM`

### 3. Learning from high-dimensional vectors obtained by nonlinear projection of causal variables (given a fixed node ordering): 
- Observational data: `python biols_vector_data.py --config defaults nonlinear_vector_obs_learn_L`
- Single intervention targets: `python biols_vector_data.py --config defaults nonlinear_vector_single_interv_learn_L`
- Multi intervention targets: `python biols_vector_data.py --config defaults nonlinear_vector_multi_interv_learn_L`

### 4. Learning from high-dimensional vectors obtained by linear projection of causal variables (learning over node orderings): 
- Observational data: `python batched_biols_vector_data.py --config defaults batched_nonlinear_vector_obs_learn_SCM`
- Single intervention targets: `python batched_biols_vector_data.py --config defaults batched_nonlinear_vector_single_interv_learn_SCM`
- Multi intervention targets: `python batched_biols_vector_data.py --config defaults batched_nonlinear_vector_multi_interv_learn_SCM`


### 5. Learning from image pixels obtained by nonlinear projection of causal variables (from images in the chemistry dataset proposed in [Ke et al](https://arxiv.org/abs/2107.00848)): 

`python biols_image_data.py --config defaults biols_chem_data`