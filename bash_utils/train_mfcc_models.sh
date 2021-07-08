#!/bin/bash

shopt -s expand_aliases
alias stool='/private/home/mriviere/FairInternal/stool/stool.py'

data_path=/private/home/marvinlvn/DATA/CPC_data/train
#model_types=(dpgmm gmm)
model_types=(dpgmm)
sizes=(8h 16h 32h 64h 128h 256h)
languages=(English_LibriVox_extracted_full_random French_LibriVox_extracted_full_random)
# !!!! CAREFUL WITH NUMBER OF COMPONENTS
N_COMPONENT=1000
cd /private/home/marvinlvn/megamix

for model_type in ${model_types[*]}; do
  for size in ${sizes[*]}; do
    for language in ${languages[*]}; do
      feat=${data_path}/${language}/mfccs_${size}_nb_0.pt
      out_dir=/checkpoint/marvinlvn/megamix/mfccs/${model_type}/${N_COMPONENT}/${language}/${size}_nb_0
      stool run speech/model.py --args="--model_type ${model_type} --feat ${feat} --n_jobs 20 --n_components $N_COMPONENT \
        --out_dir ${out_dir}" \
      --ncpu=20 --mem=512000 --name=megamix/mfccs/${model_type}/${language}/${size} --partition=learnfair --anaconda=/private/home/marvinlvn/.conda/envs/megamix
    done;
  done;
done;