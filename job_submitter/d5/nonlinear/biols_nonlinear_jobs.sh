#!/bin/bash

cd ../../..
./jobs.sh 1 er1-ws_datagen_fix_noise_interv_noise-3_layer_mlpproj-d005-D0100-multi-n_pairs2000-sets20-uniforminterv
./jobs.sh 1 er1-ws_datagen_fix_noise_interv_noise-3_layer_mlpproj-d005-D0100-multi-n_pairs2000-sets20-zerosinterv
./jobs.sh 1 er1-ws_datagen_fix_noise_interv_noise-3_layer_mlpproj-d005-D0100-multi-n_pairs2000-sets20-gaussianinterv

./jobs.sh 1 er1-ws_datagen_fix_noise_interv_noise-3_layer_mlpproj-d005-D0100-single-n_pairs2000-sets20-uniforminterv
./jobs.sh 1 er1-ws_datagen_fix_noise_interv_noise-3_layer_mlpproj-d005-D0100-single-n_pairs2000-sets20-zerosinterv
./jobs.sh 1 er1-ws_datagen_fix_noise_interv_noise-3_layer_mlpproj-d005-D0100-single-n_pairs2000-sets20-gaussianinterv

./jobs.sh 1 er2-ws_datagen_fix_noise_interv_noise-3_layer_mlpproj-d005-D0100-multi-n_pairs2000-sets20-uniforminterv
./jobs.sh 1 er2-ws_datagen_fix_noise_interv_noise-3_layer_mlpproj-d005-D0100-multi-n_pairs2000-sets20-zerosinterv
./jobs.sh 1 er2-ws_datagen_fix_noise_interv_noise-3_layer_mlpproj-d005-D0100-multi-n_pairs2000-sets20-gaussianinterv

./jobs.sh 1 er2-ws_datagen_fix_noise_interv_noise-3_layer_mlpproj-d005-D0100-single-n_pairs2000-sets20-uniforminterv
./jobs.sh 1 er2-ws_datagen_fix_noise_interv_noise-3_layer_mlpproj-d005-D0100-single-n_pairs2000-sets20-zerosinterv
./jobs.sh 1 er2-ws_datagen_fix_noise_interv_noise-3_layer_mlpproj-d005-D0100-single-n_pairs2000-sets20-gaussianinterv

