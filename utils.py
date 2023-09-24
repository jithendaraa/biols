import os
import argparse
import numpy as onp
import jax, pdb
from jax import numpy as jnp
from jax import jit, vmap
from math import prod
from os.path import join
import matplotlib.pyplot as plt
from collections import namedtuple

Interventions = namedtuple('Interventions', ['targets', 'labels', 'values', 'noise'])
GTSamples = namedtuple('GTSamples', ['x', 'z', 'W', 'P', 'L', 'sigmas', 'obs_z_samples'])


def read_biols_dataset(folder_path, num_obs_samples):
	"""
		Reads previously generated datasets given `folder_path`
	"""
	x_samples = onp.load(f'{folder_path}/x_samples.npy')
	z_samples = onp.load(f'{folder_path}/z_samples.npy')
	gt_W = onp.load(f'{folder_path}/weighted_adjacency.npy')
	gt_P = onp.load(f'{folder_path}/perm.npy')
	gt_L = onp.load(f'{folder_path}/edge_weights.npy')
	gt_sigmas = onp.load(f'{folder_path}/gt_sigmas.npy')

	if len(x_samples.shape)	== 4:
		x_samples = x_samples[:, :, :, 0:1]
		
	gt_samples = GTSamples(
		x=x_samples, 
		z=z_samples, 
		obs_z_samples=z_samples[:num_obs_samples],
		W=gt_W, 
		P=gt_P, 
		L=gt_L, 
		sigmas=gt_sigmas
	)

	interv_labels = onp.load(f'{folder_path}/interv_nodes.npy')
	interv_targets = onp.load(f'{folder_path}/interv_targets.npy')
	interv_noise = onp.load(f'{folder_path}/interv_noise.npy')
	interv_values = onp.load(f'{folder_path}/interv_values.npy')
	interventions = Interventions(
		targets=interv_targets, 
		labels=interv_labels, 
		values=interv_values, 
		noise=interv_noise
	)
	return gt_samples, interventions


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

	folder_path = os.path.join(
		opt.baseroot, 
		'scratch/biols_datasets', 
		opt.biols_data_folder, 
		str(opt.data_seed).zfill(2)
	)
	artifact_metadata = onp.load(f'{folder_path}/artifact_metadata.npy', allow_pickle=True).item()

	for k, v in artifact_metadata.items():
		setattr(opt, k, v)

	opt.proj = artifact_metadata['projection']
	opt.obs_data = artifact_metadata['n_pairs']

	try:
		opt.num_samples = int(opt.pts_per_interv * opt.n_interv_sets) + opt.obs_data
	except:
		opt.num_samples = 2 * opt.n_pairs
	return opt, folder_path


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
	if (d - 1) - 2 * degree > 0.:
		horseshoe_tau = (1 / onp.sqrt(n)) * (2 * degree / ((d - 1) - 2 * degree))
	else: 
		horseshoe_tau = 1 / (2 * d)

	print(f"Horseshoe tau: {horseshoe_tau}")
	return horseshoe_tau


def set_logdir(opt): 
	logdir = 'logs/' 
	if opt.exp_name[:5] == 'BIOLS':
		logdir += f'BIOLS_({opt.num_nodes})_seed{opt.data_seed}'
	elif opt.exp_name[:3] == 'VAE':
		logdir += f'VAE_({opt.num_nodes})_seed{opt.data_seed}'
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

def nll_gaussian_per_sample(x_i, pred_x_i, pred_sigma_i):
	D = len(x_i)
	const_term = 0.5 * D * jnp.log(2 * jnp.pi)
	sigma_term = D * jnp.log(pred_sigma_i)
	sum_squares = jnp.sum((x_i - pred_x_i)**2) / pred_sigma_i**2
	loss = 0.5 * sum_squares + sigma_term + const_term
	return loss

def nll_gaussian(x, pred_x, pred_sigma):
	sample_wise_nlls = vmap(nll_gaussian_per_sample, (0, 0, None))(x, pred_x, pred_sigma)
	total_nll = jnp.sum(sample_wise_nlls)
	return total_nll


def W_intervened_shd(gt_W, pred_W, gt_interv_targets, pred_interv_targets, threshold=0.3):
	"""
		GT W might not be same but (W, I) and (W', I') might induce the same adjacency matrix. 
		This function calculates SHD((W, I), (W', I'))
	"""
	n_samples = gt_interv_targets.shape[0]
	
	samplewise_gt_W = jnp.tile(gt_W, (n_samples, 1, 1))
	gt_interv_target_indices = jnp.argwhere(gt_interv_targets)
	samplewise_gt_W = samplewise_gt_W.at[gt_interv_target_indices[:, 0], :,  gt_interv_target_indices[:, 1]].set(0.0)
	samplewise_gt_G = (jnp.abs(samplewise_gt_W) > threshold).astype(int)

	samplewise_pred_W = jnp.tile(pred_W, (n_samples, 1, 1))
	pred_interv_target_indices = jnp.argwhere(pred_interv_targets)
	samplewise_pred_W = samplewise_pred_W.at[pred_interv_target_indices[:, 0], :,  pred_interv_target_indices[:, 1]].set(0.0)
	samplewise_pred_G = (jnp.abs(samplewise_pred_W) > threshold).astype(int)
	shds = jnp.sum(jnp.abs(samplewise_gt_G - samplewise_pred_G), axis=(1, 2))
	expected_shd = jnp.mean(shds)
	return expected_shd


def save_graph_images(P, L, W, P_filename, L_filename, W_filename, logdir):
	plt.imshow(W)
	plt.savefig(join(logdir, W_filename))
	plt.close('all')

	plt.imshow(P)
	plt.savefig(join(logdir, P_filename))
	plt.close('all')

	plt.imshow(L)
	plt.savefig(join(logdir, L_filename))
	plt.close('all')

@jit
def bce_loss(labels, pred_probas):
	log_p = jnp.log(pred_probas)
	log_not_p = jnp.log(1. - pred_probas)
	return -jnp.mean(labels * log_p + (1 - labels) * log_not_p)

@jit
def cross_entropy_loss(labels, pred_logits):
	class_log_probas = jax.nn.log_softmax(pred_logits)
	ce_loss = -jnp.dot(labels, class_log_probas)
	return ce_loss

@jit
def batched_cross_entropy_loss(labels, pred_logits):
	vmapped_ce_loss = vmap(cross_entropy_loss, in_axes=(0, 0), out_axes=0)
	return jnp.mean(vmapped_ce_loss(labels, pred_logits))

@jit
def get_obs_classification_accuracy(pred_is_observ_probas, pred_is_observ, gt_is_observ):
	obs_classification_loss = bce_loss(gt_is_observ, pred_is_observ_probas)
	obs_classification_accuracy = jnp.mean((pred_is_observ == gt_is_observ).astype(int)) * 100
	return obs_classification_loss, obs_classification_accuracy

@jit
def get_classification_accuracy(pred_targets, gt_targets):
	incorrect_targets = jnp.sum(jnp.abs(gt_targets - pred_targets))
	interv_target_inaccuracy = incorrect_targets * 100 / prod(pred_targets.shape)
	accuracy = 100. - interv_target_inaccuracy
	return accuracy


