import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pathlib, random
import os, sys, pdb
import ruamel.yaml as yaml
from tqdm import tqdm
from collections import OrderedDict
import wandb

# Add to PATH
sys.path.append("..")
sys.path.append("../modules")

import numpy as onp
import jax
from jax import numpy as jnp
from jax import jit, lax, vmap, value_and_grad, config
import haiku as hk
import optax

import utils
from modules.encoder_biols_model_init import init_model
from datagen import SyntheticDatagen

config.update("jax_enable_x64", True)

# Load config yaml as options for experiment
configs = yaml.safe_load((pathlib.Path("..") / "configs.yaml").read_text())
opt = utils.load_yaml(configs)
opt.num_samples = int(opt.pts_per_interv * opt.n_interv_sets) + opt.obs_data

# Set seeds
onp.random.seed(opt.data_seed)

logdir = utils.set_logdir(opt)
os.makedirs(logdir, exist_ok=True)

x_samples = []
z_samples = []
interv_labels = []
interv_targets = []
interv_values = []

rng_key = jax.random.PRNGKey(opt.data_seed)

if opt.load_data is False:
    for i in tqdm(range(opt.num_data_seeds), desc=f'Generating {opt.interv_value_sampling} data: {opt.interv_type}-target interventions '):
        rng_key, subkey = jax.random.split(rng_key)
        hk_key = hk.PRNGSequence(i)

        scm = SyntheticDatagen(
            data_seed=i,
            hk_key=hk_key,
            rng_key=rng_key,
            num_samples=opt.num_samples,
            num_obs_samples=opt.obs_data,
            num_nodes=opt.num_nodes,
            degree=opt.exp_edges,
            interv_sets=opt.n_interv_sets,
            interv_type=opt.interv_type,
            proj_dims=opt.proj_dims,
            projection=opt.proj,
            decoder_sigma=opt.decoder_sigma,
            interv_value_sampling=opt.interv_value_sampling,
            datagen_type=opt.datagen_type,
            eq_noise_var=opt.eq_noise_var,
            sem_type=opt.sem_type,
            graph_type=opt.graph_type,
            dataset_type='linear',
            min_interv_value=opt.min_interv_value
        )

        x1, x2, z1, z2, labels, targets, values = scm.sample_weakly_supervised(
            rng_key, 
            opt.obs_data, 
            opt.n_interv_sets, 
            return_interv_values=True
        )
        assert jnp.allclose(z2[targets == 1], values[targets == 1])

        targets = onp.concatenate([onp.zeros(z1.shape).astype(int), targets], axis=0)
        is_observational = onp.logical_not(onp.any(targets, axis=-1))
        targets = onp.concatenate([targets, is_observational[:, None]], axis=-1) # append is_observational to targets

        indices = onp.arange(opt.num_samples)
        onp.random.shuffle(indices)

        x = onp.concatenate([x1, x2], axis=0)[indices]
        z = onp.concatenate([z1, z2], axis=0)[indices]
        targets = targets[indices]
        labels = onp.concatenate((onp.ones_like(labels) * opt.num_nodes, labels), axis=0)[indices]
        values = onp.concatenate([onp.zeros(z1.shape), values], axis=0)[indices]

        x_samples.append(x)
        z_samples.append(z) 
        interv_labels.append(labels)
        interv_targets.append(targets)
        interv_values.append(values)

    x_samples = onp.stack(x_samples, axis=0)
    z_samples = onp.stack(z_samples, axis=0)
    interv_labels = onp.stack(interv_labels, axis=0)
    interv_targets = onp.stack(interv_targets, axis=0)
    interv_values = onp.stack(interv_values, axis=0)

    # Save x_samples
    onp.save(os.path.join(opt.baseroot, f'scratch/x_samples_{opt.num_data_seeds}_{opt.num_samples}.npy'), x_samples)
    onp.save(os.path.join(opt.baseroot, f'scratch/z_samples_{opt.num_data_seeds}_{opt.num_samples}.npy'), z_samples)
    onp.save(os.path.join(opt.baseroot, f'scratch/interv_labels_{opt.num_data_seeds}_{opt.num_samples}.npy'), interv_labels)
    onp.save(os.path.join(opt.baseroot, f'scratch/interv_targets_{opt.num_data_seeds}_{opt.num_samples}.npy'), interv_targets)
    onp.save(os.path.join(opt.baseroot, f'scratch/interv_values_{opt.num_data_seeds}_{opt.num_samples}.npy'), interv_values)

else:
    # Load x_samples
    x_samples = onp.load(os.path.join(opt.baseroot, f'scratch/x_samples_{opt.num_data_seeds}_{opt.num_samples}.npy'))
    z_samples = onp.load(os.path.join(opt.baseroot, f'scratch/z_samples_{opt.num_data_seeds}_{opt.num_samples}.npy'))
    interv_labels = onp.load(os.path.join(opt.baseroot, f'scratch/interv_labels_{opt.num_data_seeds}_{opt.num_samples}.npy'))
    interv_targets = onp.load(os.path.join(opt.baseroot, f'scratch/interv_targets_{opt.num_data_seeds}_{opt.num_samples}.npy'))
    interv_values = onp.load(os.path.join(opt.baseroot, f'scratch/interv_values_{opt.num_data_seeds}_{opt.num_samples}.npy'))


class SCMData(Dataset):
    def __init__(self, x_samples, z_samples, interv_labels, interv_targets, interv_values):
        self.x_samples = torch.from_numpy(x_samples)
        self.z_samples = torch.from_numpy(z_samples)
        self.interv_labels = torch.from_numpy(interv_labels)
        self.interv_targets = torch.from_numpy(interv_targets)
        self.interv_values = torch.from_numpy(interv_values)
    
    def __getitem__(self, index):
        return self.x_samples[index], self.z_samples[index], self.interv_labels[index], self.interv_targets[index], self.interv_values[index]
    
    def __len__(self):
        return len(self.x_samples)


class InterventionInference(nn.Module):
    def __init__(self, input_seq_len, num_nodes, D):
        super(InterventionInference, self).__init__()

        self.posn_embedding = nn.Parameter(torch.randn(1, input_seq_len, 256))
        self.node_embedding = nn.Parameter(torch.randn(1, num_nodes+1, 256))
        self.projector = nn.Linear(D, 256)
        self.encoder_layer = nn.TransformerEncoderLayer(256, nhead=opt.nhead)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=opt.num_encoder_layers) # (N+d, 256) -> (d, 256) -> (d,1)

    def forward(self, src):
        batch_size, N, D = src.shape
        src_embed = self.projector(src) # (B, N, D) -> (B, N, 256)
        x = torch.cat([src_embed, self.node_embedding.repeat(batch_size, 1, 1)], dim=1) # (B, N+(d+1), 256)
        x_posn_embed = x + self.posn_embedding.repeat(batch_size, 1, 1) # (B, N+(d+1), 256)
        x_summary = self.encoder(x_posn_embed) # (B, N+(d+1), 256) -> (B, d, 256)
        N_summary = x_summary[:, :N, :] # (B, N, 256)
        d_summary = x_summary[:, N:, :] # (B, d+1, 256)
        interv_logits = torch.einsum("bij,bkj->bik", N_summary, d_summary) # (B, N, 256) x (B, d+1, 256) -> (B, N, d+1)
        return interv_logits


scm_dataset = SCMData(x_samples, z_samples, interv_labels, interv_targets, interv_values)
scm_dataloader = DataLoader(scm_dataset, batch_size=opt.batch_size, shuffle=True)

model = InterventionInference(
    input_seq_len=opt.num_samples + opt.num_nodes + 1, 
    num_nodes=opt.num_nodes,
    D=opt.proj_dims
)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

if opt.off_wandb is False:
    if opt.offline_wandb is True: os.system('wandb offline')
    else:   os.system('wandb online')    
    wandb.init(project = opt.wandb_project, 
                entity = opt.wandb_entity, 
                config = vars(opt), 
                settings = wandb.Settings(start_method="fork"))
    wandb.run.name = f'TransformerInterventionInference_layers({opt.num_encoder_layers})_K{opt.num_data_seeds}'
    wandb.run.save()

for epoch in tqdm(range(opt.num_epochs), desc='Epoch'):
    pbar = tqdm(enumerate(scm_dataloader), total=len(scm_dataloader))

    for i, (x, z, _, targets, values) in pbar:
        # Set all targets to same intervention
        # targets[:, :, :] = 0
        # targets[:, :, -1] = 1

        interv_logits = model(x)
        loss = criterion(interv_logits, targets.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred_interv_targets = torch.sigmoid(interv_logits) > 0.5
        row_wise_corrects = torch.sum(pred_interv_targets == targets, dim=-1)
        all_corrects = torch.full(row_wise_corrects.shape, pred_interv_targets.shape[-1])
        classification_accuracy = torch.mean((row_wise_corrects == all_corrects).float()) * 100

        postfix_dict = OrderedDict(
            loss=f"{loss.item():.4f}",
            classification_accuracy=f"{classification_accuracy.item():.2f}",
        )
        pbar.set_postfix(postfix_dict)

        wandb_dict = {
            "Loss": loss.item(),
            "Classification accuracy": classification_accuracy.item()
        }
    
    if opt.off_wandb is False:
        wandb.log(wandb_dict, step=epoch)



# z_i (N, d) 
# x_i (N, D) [D >> "d"]
# I_i (N, 10)
# d=10;
# (K, N+d, D) -> (K, d, 1); Parameters(randn init)
# Take last 10 elems of x_i, MLP -> 1