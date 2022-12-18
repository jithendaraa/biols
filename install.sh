conda create --name biols_env python=3.9
conda activate biols_env
pip install --upgrade pip
# module load cuda/11.1
conda update -n base conda
# 0.3.20
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install tensorflow-probability==0.18.0
conda install -c conda-forge ruamel.yaml
pip install dm-haiku networkx==2.6.3 tqdm
conda install -c conda-forge optax 
pip install ott-jax # preferably version 0.2.8
conda install -c conda-forge cdt
pip install graphical-models wandb