#!/bin/bash

cd ..

# d5 er1 SON projection
python create_datasets.py --config create_dataset son_d5 ws_datagen_interv_noise_fix_noise gaussian_intervs er1 single_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 0 && python create_datasets.py --config create_dataset son_d5 ws_datagen_interv_noise_fix_noise gaussian_intervs er1 single_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 1 && python create_datasets.py --config create_dataset son_d5 ws_datagen_interv_noise_fix_noise gaussian_intervs er1 single_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 2 && python create_datasets.py --config create_dataset son_d5 ws_datagen_interv_noise_fix_noise gaussian_intervs er1 single_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 3 && python create_datasets.py --config create_dataset son_d5 ws_datagen_interv_noise_fix_noise gaussian_intervs er1 single_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 4 
python create_datasets.py --config create_dataset son_d5 ws_datagen_interv_noise_fix_noise gaussian_intervs er1 multi_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 0 && python create_datasets.py --config create_dataset son_d5 ws_datagen_interv_noise_fix_noise gaussian_intervs er1 multi_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 1 && python create_datasets.py --config create_dataset son_d5 ws_datagen_interv_noise_fix_noise gaussian_intervs er1 multi_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 2 && python create_datasets.py --config create_dataset son_d5 ws_datagen_interv_noise_fix_noise gaussian_intervs er1 multi_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 3 && python create_datasets.py --config create_dataset son_d5 ws_datagen_interv_noise_fix_noise gaussian_intervs er1 multi_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 4 
python create_datasets.py --config create_dataset son_d5 ws_datagen_interv_noise_fix_noise zero_intervs er1 single_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 0 && python create_datasets.py --config create_dataset son_d5 ws_datagen_interv_noise_fix_noise zero_intervs er1 single_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 1 && python create_datasets.py --config create_dataset son_d5 ws_datagen_interv_noise_fix_noise zero_intervs er1 single_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 2 && python create_datasets.py --config create_dataset son_d5 ws_datagen_interv_noise_fix_noise zero_intervs er1 single_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 3 && python create_datasets.py --config create_dataset son_d5 ws_datagen_interv_noise_fix_noise zero_intervs er1 single_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 4 
python create_datasets.py --config create_dataset son_d5 ws_datagen_interv_noise_fix_noise zero_intervs er1 multi_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 0 && python create_datasets.py --config create_dataset son_d5 ws_datagen_interv_noise_fix_noise zero_intervs er1 multi_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 1 && python create_datasets.py --config create_dataset son_d5 ws_datagen_interv_noise_fix_noise zero_intervs er1 multi_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 2 && python create_datasets.py --config create_dataset son_d5 ws_datagen_interv_noise_fix_noise zero_intervs er1 multi_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 3 && python create_datasets.py --config create_dataset son_d5 ws_datagen_interv_noise_fix_noise zero_intervs er1 multi_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 4 

# d5 er2 SON projection
python create_datasets.py --config create_dataset son_d5 ws_datagen_interv_noise_fix_noise gaussian_intervs er2 single_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 0 && python create_datasets.py --config create_dataset son_d5 ws_datagen_interv_noise_fix_noise gaussian_intervs er2 single_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 1 && python create_datasets.py --config create_dataset son_d5 ws_datagen_interv_noise_fix_noise gaussian_intervs er2 single_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 2 && python create_datasets.py --config create_dataset son_d5 ws_datagen_interv_noise_fix_noise gaussian_intervs er2 single_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 3 && python create_datasets.py --config create_dataset son_d5 ws_datagen_interv_noise_fix_noise gaussian_intervs er2 single_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 4 
python create_datasets.py --config create_dataset son_d5 ws_datagen_interv_noise_fix_noise gaussian_intervs er2 multi_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 0 && python create_datasets.py --config create_dataset son_d5 ws_datagen_interv_noise_fix_noise gaussian_intervs er2 multi_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 1 && python create_datasets.py --config create_dataset son_d5 ws_datagen_interv_noise_fix_noise gaussian_intervs er2 multi_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 2 && python create_datasets.py --config create_dataset son_d5 ws_datagen_interv_noise_fix_noise gaussian_intervs er2 multi_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 3 && python create_datasets.py --config create_dataset son_d5 ws_datagen_interv_noise_fix_noise gaussian_intervs er2 multi_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 4 
python create_datasets.py --config create_dataset son_d5 ws_datagen_interv_noise_fix_noise zero_intervs er2 single_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 0 && python create_datasets.py --config create_dataset son_d5 ws_datagen_interv_noise_fix_noise zero_intervs er2 single_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 1 && python create_datasets.py --config create_dataset son_d5 ws_datagen_interv_noise_fix_noise zero_intervs er2 single_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 2 && python create_datasets.py --config create_dataset son_d5 ws_datagen_interv_noise_fix_noise zero_intervs er2 single_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 3 && python create_datasets.py --config create_dataset son_d5 ws_datagen_interv_noise_fix_noise zero_intervs er2 single_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 4 
python create_datasets.py --config create_dataset son_d5 ws_datagen_interv_noise_fix_noise zero_intervs er2 multi_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 0 && python create_datasets.py --config create_dataset son_d5 ws_datagen_interv_noise_fix_noise zero_intervs er2 multi_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 1 && python create_datasets.py --config create_dataset son_d5 ws_datagen_interv_noise_fix_noise zero_intervs er2 multi_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 2 && python create_datasets.py --config create_dataset son_d5 ws_datagen_interv_noise_fix_noise zero_intervs er2 multi_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 3 && python create_datasets.py --config create_dataset son_d5 ws_datagen_interv_noise_fix_noise zero_intervs er2 multi_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 4 

# d10 er1 SON projection
python create_datasets.py --config create_dataset son_d10 ws_datagen_interv_noise_fix_noise gaussian_intervs er1 single_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 0 && python create_datasets.py --config create_dataset son_d10 ws_datagen_interv_noise_fix_noise gaussian_intervs er1 single_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 1 && python create_datasets.py --config create_dataset son_d10 ws_datagen_interv_noise_fix_noise gaussian_intervs er1 single_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 2 && python create_datasets.py --config create_dataset son_d10 ws_datagen_interv_noise_fix_noise gaussian_intervs er1 single_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 3 && python create_datasets.py --config create_dataset son_d10 ws_datagen_interv_noise_fix_noise gaussian_intervs er1 single_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 4 
python create_datasets.py --config create_dataset son_d10 ws_datagen_interv_noise_fix_noise gaussian_intervs er1 multi_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 0 && python create_datasets.py --config create_dataset son_d10 ws_datagen_interv_noise_fix_noise gaussian_intervs er1 multi_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 1 && python create_datasets.py --config create_dataset son_d10 ws_datagen_interv_noise_fix_noise gaussian_intervs er1 multi_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 2 && python create_datasets.py --config create_dataset son_d10 ws_datagen_interv_noise_fix_noise gaussian_intervs er1 multi_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 3 && python create_datasets.py --config create_dataset son_d10 ws_datagen_interv_noise_fix_noise gaussian_intervs er1 multi_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 4 
python create_datasets.py --config create_dataset son_d10 ws_datagen_interv_noise_fix_noise zero_intervs er1 single_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 0 && python create_datasets.py --config create_dataset son_d10 ws_datagen_interv_noise_fix_noise zero_intervs er1 single_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 1 && python create_datasets.py --config create_dataset son_d10 ws_datagen_interv_noise_fix_noise zero_intervs er1 single_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 2 && python create_datasets.py --config create_dataset son_d10 ws_datagen_interv_noise_fix_noise zero_intervs er1 single_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 3 && python create_datasets.py --config create_dataset son_d10 ws_datagen_interv_noise_fix_noise zero_intervs er1 single_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 4 
python create_datasets.py --config create_dataset son_d10 ws_datagen_interv_noise_fix_noise zero_intervs er1 multi_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 0 && python create_datasets.py --config create_dataset son_d10 ws_datagen_interv_noise_fix_noise zero_intervs er1 multi_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 1 && python create_datasets.py --config create_dataset son_d10 ws_datagen_interv_noise_fix_noise zero_intervs er1 multi_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 2 && python create_datasets.py --config create_dataset son_d10 ws_datagen_interv_noise_fix_noise zero_intervs er1 multi_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 3 && python create_datasets.py --config create_dataset son_d10 ws_datagen_interv_noise_fix_noise zero_intervs er1 multi_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 4 

# d10 er2 SON projection
python create_datasets.py --config create_dataset son_d10 ws_datagen_interv_noise_fix_noise gaussian_intervs er2 single_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 0 && python create_datasets.py --config create_dataset son_d10 ws_datagen_interv_noise_fix_noise gaussian_intervs er2 single_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 1 && python create_datasets.py --config create_dataset son_d10 ws_datagen_interv_noise_fix_noise gaussian_intervs er2 single_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 2 && python create_datasets.py --config create_dataset son_d10 ws_datagen_interv_noise_fix_noise gaussian_intervs er2 single_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 3 && python create_datasets.py --config create_dataset son_d10 ws_datagen_interv_noise_fix_noise gaussian_intervs er2 single_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 4 
python create_datasets.py --config create_dataset son_d10 ws_datagen_interv_noise_fix_noise gaussian_intervs er2 multi_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 0 && python create_datasets.py --config create_dataset son_d10 ws_datagen_interv_noise_fix_noise gaussian_intervs er2 multi_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 1 && python create_datasets.py --config create_dataset son_d10 ws_datagen_interv_noise_fix_noise gaussian_intervs er2 multi_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 2 && python create_datasets.py --config create_dataset son_d10 ws_datagen_interv_noise_fix_noise gaussian_intervs er2 multi_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 3 && python create_datasets.py --config create_dataset son_d10 ws_datagen_interv_noise_fix_noise gaussian_intervs er2 multi_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 4 
python create_datasets.py --config create_dataset son_d10 ws_datagen_interv_noise_fix_noise zero_intervs er2 single_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 0 && python create_datasets.py --config create_dataset son_d10 ws_datagen_interv_noise_fix_noise zero_intervs er2 single_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 1 && python create_datasets.py --config create_dataset son_d10 ws_datagen_interv_noise_fix_noise zero_intervs er2 single_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 2 && python create_datasets.py --config create_dataset son_d10 ws_datagen_interv_noise_fix_noise zero_intervs er2 single_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 3 && python create_datasets.py --config create_dataset son_d10 ws_datagen_interv_noise_fix_noise zero_intervs er2 single_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 4 
python create_datasets.py --config create_dataset son_d10 ws_datagen_interv_noise_fix_noise zero_intervs er2 multi_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 0 && python create_datasets.py --config create_dataset son_d10 ws_datagen_interv_noise_fix_noise zero_intervs er2 multi_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 1 && python create_datasets.py --config create_dataset son_d10 ws_datagen_interv_noise_fix_noise zero_intervs er2 multi_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 2 && python create_datasets.py --config create_dataset son_d10 ws_datagen_interv_noise_fix_noise zero_intervs er2 multi_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 3 && python create_datasets.py --config create_dataset son_d10 ws_datagen_interv_noise_fix_noise zero_intervs er2 multi_interv --n_pairs 2000 --n_interv_sets 20  --data_seed 4 