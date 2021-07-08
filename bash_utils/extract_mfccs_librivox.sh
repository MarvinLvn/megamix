#!/bin/bash

shopt -s expand_aliases
alias stool='/private/home/mriviere/FairInternal/stool/stool.py'

data_path=/private/home/marvinlvn/DATA/CPC_data/train
languages=(English_LibriVox_extracted_full_random French_LibriVox_extracted_full_random)
sizes=(8h 16h 32h 64h 128h 256h)

cd ~/megamix

for language in ${languages[*]}; do
  for size in ${sizes[*]}; do
    in_path=${data_path}/${language}/${size}/${size}_nb_0
    out_path=${data_path}/${language}/mfccs_${size}_nb_0.pt
    stool run speech/extract_features.py --args="--db ${in_path} --out ${out_path} --type mfcc" \
      --ncpu=10 --name=MFCC/${language}/${size} --partition=learnfair --anaconda=/private/home/marvinlvn/.conda/envs/megamix
  done;
done;