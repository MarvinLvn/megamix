#!/bin/bash

#shopt -s expand_aliases
#alias stool='/private/home/mriviere/FairInternal/stool/stool.py'
#
#data_path=/private/home/marvinlvn/DATA/CPC_data/train
##model_types=(dpgmm gmm)
##model_types=(online_gmm dpgmm gmm kmeans)
#model_types=(skgmm)
##sizes=(8h 16h 32h 64h 128h 256h)
##sizes=(8h 128h)
##sizes=(16h 32h 64h)
##sizes=(8h 16h 32h 64h 128h)
#sizes=(64h)
##covs=(full diag spherical)
##covs=(full)
#covs=(spherical)
#
##languages=(English_LibriVox_extracted_full_random French_LibriVox_extracted_full_random)
##languages=(English_LibriVox_extracted_full_random French_LibriVox_extracted_full_random)
##languages=(English_LibriVox_extracted_full_random French_LibriVox_extracted_full_random)
##languages=(French_LibriVox_extracted_full_random)
#languages=(English_LibriVox_extracted_full_random_additive_min_snr_0_max_snr_15 French_LibriVox_extracted_full_random_additive_min_snr_0_max_snr_15 English_LibriVox_extracted_full_random_reverb_meta_reverb_additive_min_snr_0_max_snr_15 French_LibriVox_extracted_full_random_reverb_meta_reverb_additive_min_snr_0_max_snr_15)
#
## !!!! CAREFUL WITH NUMBER OF COMPONENTS
#n_components=(100 150)
##n_components=(500)
#cd /private/home/marvinlvn/megamix
#
#for model_type in ${model_types[*]}; do
#  for size in ${sizes[*]}; do
#    for language in ${languages[*]}; do
#      #PCA_PATH=/checkpoint/marvinlvn/megamix/PCA/cpc_${language}_${size}_nb_0_extracted_on_${language}_${size}_nb_0
#      for n_component in ${n_components[*]}; do
#        for cov in ${covs[*]}; do
#          if [ $model_type == 'online_gmm' ]; then
#            PARAMS_ONLINE="--kappa 0.5 --window 1024"
#          else
#            PARAMS_ONLINE=""
#          fi;
#          feat=${data_path}/${language}/cpc_${language}_${size}_nb_0_extracted_on_${language}_${size}_nb_0.pt
#          SUFFIX=$(echo "_$PARAMS_ONLINE" | sed 's/ /_/g' | sed 's/--//g' | sed 's/\./_/g')
#          if [ $model_type == 'skgmm' ]; then
#            SUFFIX="_${cov}_cov"
#            PARAMS_ONLINE="--cov_type ${cov}"
#           fi;
#          out_dir=/checkpoint/marvinlvn/megamix/noise_study/cpc/${model_type}$SUFFIX/${n_component}/${language}/${size}_nb_0
#
#          if [ ! -f $out_dir/checkpoint.h5 ] && [ ! -f $out_dir/checkpoint.pkl ]; then
#            stool run speech/model.py --args="--model_type ${model_type} --feat ${feat} --n_jobs 5 --n_components $n_component \
#              --out_dir ${out_dir} $PARAMS_ONLINE" \
#            --ncpu=20 --mem=512000 --name=megamix/cpc/${model_type}$SUFFIX/${language}/${size}/${n_component} --partition=learnlab --anaconda=/private/home/marvinlvn/.conda/envs/megamix
#          fi;
#        done;
#      done;
#    done;
#  done;
#done;

#!/bin/bash

shopt -s expand_aliases
alias stool='/private/home/mriviere/FairInternal/stool/stool.py'

data_path=/private/home/marvinlvn/DATA/CPC_data/train/Daylongs
#model_types=(dpgmm gmm)
#model_types=(online_gmm dpgmm gmm kmeans)
model_types=(skgmm)
#sizes=(8h 16h 32h 64h 128h 256h)
#sizes=(8h 128h)
#sizes=(16h 32h 64h)
#sizes=(8h 16h 32h 64h 128h)
#sizes=(10h 20h 40h 80h 150h)
sizes=(150h_speech_1050h_noiseh 150h_speech_150h_noiseh 150h_speech_450h_noiseh)
#covs=(full diag spherical)
#covs=(full)
covs=(spherical)

#languages=(English_LibriVox_extracted_full_random French_LibriVox_extracted_full_random)
#languages=(English_LibriVox_extracted_full_random French_LibriVox_extracted_full_random)
#languages=(English_LibriVox_extracted_full_random French_LibriVox_extracted_full_random)
#languages=(French_LibriVox_extracted_full_random)
#languages=(FRENCH_MAL_FEM_gt_1500ms ACLEW10K_MAL_FEM_gt_1500ms)
languages=(ACLEW10K_MAL_FEM_NOISE_gt_1500ms)
# !!!! CAREFUL WITH NUMBER OF COMPONENTS
n_components=(150)
#n_components=(500)
cd /private/home/marvinlvn/megamix

for model_type in ${model_types[*]}; do
  for size in ${sizes[*]}; do
    for language in ${languages[*]}; do
      #PCA_PATH=/checkpoint/marvinlvn/megamix/PCA/cpc_${language}_${size}_nb_0_extracted_on_${language}_${size}_nb_0
      for n_component in ${n_components[*]}; do
        for cov in ${covs[*]}; do
          if [ $model_type == 'online_gmm' ]; then
            PARAMS_ONLINE="--kappa 0.5 --window 1024"
          else
            PARAMS_ONLINE=""
          fi;
          feat=${data_path}/${language}/${language}_${size}.pt
          SUFFIX=$(echo "_$PARAMS_ONLINE" | sed 's/ /_/g' | sed 's/--//g' | sed 's/\./_/g')
          if [ $model_type == 'skgmm' ]; then
            SUFFIX="_${cov}_cov"
            PARAMS_ONLINE="--cov_type ${cov}"
           fi;
          out_dir=/checkpoint/marvinlvn/megamix/noise_study/daylong/cpc/${model_type}$SUFFIX/${n_component}/${language}/${size}

          if [ ! -f $out_dir/checkpoint.h5 ] && [ ! -f $out_dir/checkpoint.pkl ]; then
            echo stool run speech/model.py --args="--model_type ${model_type} --feat ${feat} --n_jobs 5 --n_components $n_component \
              --out_dir ${out_dir} $PARAMS_ONLINE" \
            --ncpu=20 --mem=512000 --name=megamix/cpc/${model_type}$SUFFIX/${language}/${size}/${n_component} --partition=learnlab --anaconda=/private/home/marvinlvn/.conda/envs/megamix
          fi;
        done;
      done;
    done;
  done;
done;