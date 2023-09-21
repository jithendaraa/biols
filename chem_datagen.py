import gym
import networkx as nx
import numpy as onp
import jax.numpy as jnp

from typing import OrderedDict
from tqdm import tqdm
from modules.SyntheticSCM import LinearGaussianColorSCM

def generate_colors(opt, chem_data, low, high, interv_low, interv_high): 
    """
        [TODO]
    """
    n = opt.num_samples
    d = opt.num_nodes
    n_interv_sets = opt.n_interv_sets
    interv_data_per_set = (opt.num_samples - opt.obs_data) // n_interv_sets
    obs_data = chem_data.obs_X

    interv_data = []
    interv_values = onp.random.uniform(low=interv_low, high=interv_high, size=(n, d))
    interv_targets = onp.full((n, d), False)

    for i in range(n_interv_sets):
        interv_k_nodes = onp.random.randint(1, d)
        intervened_node_idxs = onp.random.choice(d, interv_k_nodes, replace=False)
        interv_targets[opt.obs_data + i * interv_data_per_set : opt.obs_data + (i+1) * interv_data_per_set, intervened_node_idxs] = True
        interv_value = interv_values[opt.obs_data + i * interv_data_per_set : opt.obs_data + (i+1) * interv_data_per_set]

        interv_data_ = chem_data.intervene_sem(chem_data.W, 
                                                interv_data_per_set, 
                                                opt.sem_type,
                                                sigmas=[opt.noise_sigma], 
                                                idx_to_fix=intervened_node_idxs, 
                                                values_to_fix=interv_value, 
                                                low=low, 
                                                high=high)
        if i == 0:  interv_data = interv_data_
        else: interv_data = onp.concatenate((interv_data, interv_data_), axis=0)

    z = onp.concatenate((obs_data, interv_data), axis=0)
    return z, interv_targets, interv_values

def generate_chemdata(opt, sigmas, low=-8., high=8., interv_low=-5., interv_high=5., baseroot=None):
    """
        [TODO]
    """
    n = opt.num_samples
    d = opt.num_nodes

    if opt.generate:
        chem_data = LinearGaussianColorSCM(
                        n=opt.num_samples,
                        obs_data=opt.obs_data,
                        d=opt.num_nodes,
                        graph_type="erdos-renyi",
                        degree=2 * opt.exp_edges,
                        sem_type=opt.sem_type,
                        dataset_type="linear",
                        sigmas=sigmas,
                        data_seed=opt.data_seed,
                        low=low, 
                        high=high
                    )
        gt_W = chem_data.W
        gt_P = chem_data.P
        gt_L = chem_data.P.T @ chem_data.W.T @ chem_data.P

        # ? generate linear gaussian colors
        z, interv_targets, interv_values = generate_colors(opt, chem_data, low, high, interv_low, interv_high)
        normalized_z = 255. * ((z / (2 * high)) + 0.5)

        # Use above colors (z) to generate images
        images = generate_chem_image_dataset(opt.num_samples, opt.num_nodes, interv_values, interv_targets, z)
        onp.save(f'{baseroot}/scratch/interv_values-seed{opt.data_seed}_d{d}_ee{int(opt.exp_edges)}.npy', onp.array(interv_values))
        onp.save(f'{baseroot}/scratch/interv_targets-seed{opt.data_seed}_d{d}_ee{int(opt.exp_edges)}.npy', onp.array(interv_targets))
        onp.save(f'{baseroot}/scratch/z-seed{opt.data_seed}_d{d}_ee{int(opt.exp_edges)}.npy', onp.array(z))
        onp.save(f'{baseroot}/scratch/images-seed{opt.data_seed}_d{d}_ee{int(opt.exp_edges)}.npy', onp.array(images))
        onp.save(f'{baseroot}/scratch/W-seed{opt.data_seed}_d{d}_ee{int(opt.exp_edges)}.npy', onp.array(gt_W))
        onp.save(f'{baseroot}/scratch/P-seed{opt.data_seed}_d{d}_ee{int(opt.exp_edges)}.npy', onp.array(gt_P))

    else:
        interv_targets = jnp.array(onp.load(f'{baseroot}/scratch/interv_targets-seed{opt.data_seed}_d{d}_ee{int(opt.exp_edges)}.npy'))
        interv_values = jnp.array(onp.load(f'{baseroot}/scratch/interv_values-seed{opt.data_seed}_d{d}_ee{int(opt.exp_edges)}.npy'))
        z = jnp.array(onp.load(f'{baseroot}/scratch/z-seed{opt.data_seed}_d{d}_ee{int(opt.exp_edges)}.npy'))
        images = jnp.array(onp.load(f'{baseroot}/scratch/images-seed{opt.data_seed}_d{d}_ee{int(opt.exp_edges)}.npy'))
        gt_W = jnp.array(onp.load(f'{baseroot}/scratch/W-seed{opt.data_seed}_d{d}_ee{int(opt.exp_edges)}.npy'))
        gt_P = jnp.array(onp.load(f'{baseroot}/scratch/P-seed{opt.data_seed}_d{d}_ee{int(opt.exp_edges)}.npy'))
        gt_L = jnp.array(gt_P.T @ gt_W.T @ gt_P)

    print(gt_W)
    print()

    max_cols = jnp.max(interv_targets.sum(1))
    data_idx_array = jnp.arange(d + 1)[None, :].repeat(n, axis=0)
    dummy_interv_targets = jnp.concatenate((interv_targets, jnp.array([[False]] * n)), axis=1)
    interv_nodes = onp.split(data_idx_array[dummy_interv_targets], interv_targets.sum(1).cumsum()[:-1])
    interv_nodes = jnp.array([jnp.concatenate((interv_nodes[i], jnp.array([d] * int(max_cols - len(interv_nodes[i]))))) for i in range(n)]).astype(int)
    return z, interv_nodes, interv_values, images, gt_W, gt_P, gt_L

def generate_chem_image_dataset(n, d, interv_values, interv_targets, z):
    """
        [TODO]
    """
    images = None
    env = gym.make(f'LinGaussColorCubesRL-{d}-{d}-Static-10-v0')

    for i in tqdm(range(n)):
        action = OrderedDict()
        action['nodes'] = onp.where(interv_targets[i])
        action['values'] = interv_values[i]
        ob, _, _, _ = env.step(action, z[i])
        
        if i == 0:
            images = ob[1][jnp.newaxis, :]
        else:
            images = onp.concatenate((images, ob[1][jnp.newaxis, :]), axis=0)

    return images

def generate_test_samples(d, W, sem_type, sigmas, low, high, num_test_samples, interv_low=-5., interv_high=5.):
    test_interv_data, test_interv_targets, test_interv_values = generate_samples(d,
                                                                            W, 
                                                                            sem_type,
                                                                            sigmas,
                                                                            low, high, 
                                                                            num_test_samples,
                                                                            interv_low, 
                                                                            interv_high
                                                                            )

    test_images = generate_chem_image_dataset(num_test_samples, 
                                            d, 
                                            test_interv_values, 
                                            test_interv_targets, 
                                            test_interv_data)
    _, h, w, c = test_images.shape
    padded_test_images = onp.zeros((h, 5, c))

    for i in range(num_test_samples):
        padded_test_images = onp.concatenate((padded_test_images, test_images[i]), axis=1)
        padded_test_images = onp.concatenate((padded_test_images, onp.zeros((h, 5, c))), axis=1)

    max_cols = jnp.max(test_interv_targets.sum(1))
    data_idx_array = jnp.array([jnp.arange(d + 1)] * num_test_samples)
    test_interv_nodes = onp.split(data_idx_array[test_interv_targets], test_interv_targets.sum(1).cumsum()[:-1])
    test_interv_nodes = jnp.array([jnp.concatenate((test_interv_nodes[i], jnp.array([d] * (max_cols - len(test_interv_nodes[i])))))
        for i in range(num_test_samples)]).astype(int)

    return test_interv_data, test_interv_nodes, test_interv_values, test_images[:, :, :, 0:1], padded_test_images[:, :, 0]


def intervene_sem(
        W, n, sem_type, sigmas=None, idx_to_fix=None, values_to_fix=None,
        low=-10., high=10.
    ):
        """Simulate samples from SEM with specified type of noise.
        Args:
            W: weigthed DAG
            n: number of samples
            sem_type: {linear-gauss,linear-exp,linear-gumbel}
            noise_scale: scale parameter of noise distribution in linear SEM
            idx_to_fix: intervened node or list of intervened nodes
            values_to_fix: intervened values
        Returns:
            X: [n,d] sample matrix
        """

        G = nx.DiGraph(W)
        d = W.shape[0]
        X = onp.zeros([n, d])
        if len(sigmas) == 1:
            sigmas = onp.ones(d) * sigmas

        ordered_vertices = list(nx.topological_sort(G))
        assert len(ordered_vertices) == d
        assert sem_type == "linear-gauss"

        for j in ordered_vertices:
            parents = list(G.predecessors(j))
            if isinstance(idx_to_fix, int) and j == idx_to_fix:
                X[:, j] = values_to_fix[:, j]
            elif len(onp.argwhere(idx_to_fix == j)) > 0:
                X[:, j] = values_to_fix[:, j]
            else:
                eta = X[:, parents].dot(W[parents, j])
                X[:, j] = eta + onp.random.normal(scale=sigmas[j], size=n)

        return onp.clip(X, low, high)


def generate_samples(d, W, sem_type, sigmas, low, high, num_test_samples, interv_low, interv_high):
    interv_data = []
    test_interv_values = onp.random.uniform(low=interv_low, high=interv_high, size=(num_test_samples, d))
    interv_targets = onp.full((num_test_samples, d), False)

    for i in range(num_test_samples):
        interv_k_nodes = onp.random.randint(1, d)
        intervened_node_idxs = onp.random.choice(d, interv_k_nodes, replace=False)

        interv_targets[i, intervened_node_idxs] = True
        interv_value = test_interv_values[i:i+1]

        interv_data_ = intervene_sem(W, 
                                    1, 
                                    sem_type,
                                    sigmas=sigmas, 
                                    idx_to_fix=intervened_node_idxs, 
                                    values_to_fix=interv_value, 
                                    low=low, 
                                    high=high)
        if i == 0:  interv_data = interv_data_
        else: interv_data = onp.concatenate((interv_data, interv_data_), axis=0)

    return interv_data, interv_targets, test_interv_values





