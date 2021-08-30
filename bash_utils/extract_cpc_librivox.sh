#!/bin/bash

shopt -s expand_aliases
alias stool='/private/home/mriviere/FairInternal/stool/stool.py'

data_path=/private/home/marvinlvn/DATA/CPC_data/train
#languages=(English_LibriVox_extracted_full_random French_LibriVox_extracted_full_random)
languages=(English_LibriVox_extracted_full_random_additive_min_snr_0_max_snr_15 French_LibriVox_extracted_full_random_additive_min_snr_0_max_snr_15 English_LibriVox_extracted_full_random_reverb_meta_reverb_additive_min_snr_0_max_snr_15 French_LibriVox_extracted_full_random_reverb_meta_reverb_additive_min_snr_0_max_snr_15)
sizes=(8h 16h 32h 64h 128h)
#sizes=(32h)
noise_study_path=/checkpoint/marvinlvn/noise_study
cd ~/megamix

for language in ${languages[*]}; do
  for size in ${sizes[*]}; do
    CPC_PATH=${noise_study_path}/${language}_${size}_nb_0
    BEST_EPOCH=$(python /private/home/marvinlvn/CPC_audio_jzay/utils/best_val_epoch.py --model_path ${CPC_PATH} | grep -oP "(?<=is : )([0-9]+)")
    CPC_PATH=${CPC_PATH}/checkpoint_${BEST_EPOCH}.pt

    in_path=${data_path}/${language}/${size}/${size}_nb_0
    out_path=${data_path}/${language}/cpc_${language}_${size}_nb_0_extracted_on_${language}_${size}_nb_0.pt

    stool run speech/extract_features.py --args="--db ${in_path} --out ${out_path} --type cpc \
      --cpc_path ${CPC_PATH}" \
      --ncpu=10 --ngpu=1 --name=CPC/${language}/${size} --partition=learnlab   --anaconda=/private/home/marvinlvn/.conda/envs/megamix
  done;
done;