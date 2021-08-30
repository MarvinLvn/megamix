#!/bin/bash

shopt -s expand_aliases
alias stool='/private/home/mriviere/FairInternal/stool/stool.py'

data_path=/private/home/marvinlvn/DATA/CPC_data/train
#model_types=(dpgmm gmm)
model_types=(online_gmm dpgmm gmm kmeans)
model_types=(skgmm)
covs=(full diag spherical)
covs=(full)
#sizes=(8h 16h 32h 64h 128h 256h)
#sizes=(8h)
sizes=(16h 32h 64h)
#languages=(English_LibriVox_extracted_full_random French_LibriVox_extracted_full_random)
languages=(English_LibriVox_extracted_full_random)
# !!!! CAREFUL WITH NUMBER OF COMPONENTS
n_components=(10 50 500 800 1000)
cd /private/home/marvinlvn/megamix

for model_type in ${model_types[*]}; do
  for size in ${sizes[*]}; do
    for language in ${languages[*]}; do
      for n_component in ${n_components[*]}; do
        for cov in ${covs[*]}; do
          if [ $model_type == 'online_gmm' ]; then
            PARAMS_ONLINE="--kappa 0.5 --window 1024"
          else
            PARAMS_ONLINE=""
          fi;
          feat=${data_path}/${language}/mfccs_${size}_nb_0.pt
          SUFFIX=$(echo "_$PARAMS_ONLINE" | sed 's/ /_/g' | sed 's/--//g' | sed 's/\./_/g')
          if [ $model_type == 'skgmm' ]; then
            SUFFIX="_${cov}_cov"
            PARAMS_ONLINE="--cov_type ${cov}"
          fi;
          out_dir=/checkpoint/marvinlvn/megamix/mfccs/${model_type}$SUFFIX/${n_component}/${language}/${size}_nb_0

          if [ ! -f $out_dir/checkpoint.h5 ] && [ ! -f $out_dir/checkpoint.pkl ]; then
            stool run speech/model.py --args="--model_type ${model_type} --feat ${feat} --n_jobs 5 --n_components $n_component \
              --out_dir ${out_dir} $PARAMS_ONLINE" \
            --ncpu=20 --mem=512000 --name=megamix/mfccs/${model_type}$SUFFIX/${language}/${size}/${n_component} --partition=learnfair --anaconda=/private/home/marvinlvn/.conda/envs/megamix
          fi;
        done;
      done;
    done;
  done;
done;