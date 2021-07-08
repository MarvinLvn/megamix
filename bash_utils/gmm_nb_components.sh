#!/bin/bash

shopt -s expand_aliases
alias stool='/private/home/mriviere/FairInternal/stool/stool.py'

data_path=/private/home/marvinlvn/DATA/CPC_data/train
model_types=(gmm)
sizes=(64h)
languages=(English_LibriVox_extracted_full_random)
n_components=(10 100 200 500 1000)
cd /private/home/marvinlvn/megamix

for model_type in ${model_types[*]}; do
  for size in ${sizes[*]}; do
    for language in ${languages[*]}; do
      for n_component in ${n_components[*]}; do
          feat=${data_path}/${language}/mfccs_${size}_nb_0.pt
          out_dir=/checkpoint/marvinlvn/megamix/mfccs/${model_type}/${n_component}/${language}/${size}_nb_0
          stool run speech/model.py --args="--model_type ${model_type} --feat ${feat} --n_jobs 20 --n_components $n_component \
              --out_dir ${out_dir}" \
          --ncpu=20 --mem=512000 --name=megamix/${model_type}/${language}/${size}/${n_component} --partition=learnfair --anaconda=/private/home/marvinlvn/.conda/envs/megamix
      done;
    done;
  done;
done;
