conda create --name biols_env python=3.9
conda activate biols_env
pip install "jax[cuda]==0.3.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install tqdm matplotlib
conda install -c conda-forge ruamel.yaml
pip install optax==0.1.3 dm-haiku==0.0.8 tensorflow-probability==0.18.0 ott-jax==0.2.8
pip install networkx==2.6.3
conda install -c conda-forge cdt
pip install graphical-models wandb

