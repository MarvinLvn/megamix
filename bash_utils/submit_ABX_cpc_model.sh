
##model_types=(dpgmm gmm)
##model_types=(online_gmm_kappa_0_5_window_1024 dpgmm_ gmm_)
##model_types=(skgmm_full_cov skgmm_diag_cov skgmm_spherical_cov)
#model_types=(skgmm_spherical_cov skgmm_diag_cov)
#
## OOM for 256h during training :/
##sizes=(8h 16h 32h 64h 128h)
#sizes=(64h)
##sizes=(64h 128h)
##languages=(English_LibriVox_extracted_full_random French_LibriVox_extracted_full_random)
#languages=(English_LibriVox_extracted_full_random_additive_min_snr_0_max_snr_15 French_LibriVox_extracted_full_random_additive_min_snr_0_max_snr_15 English_LibriVox_extracted_full_random_reverb_meta_reverb_additive_min_snr_0_max_snr_15 French_LibriVox_extracted_full_random_reverb_meta_reverb_additive_min_snr_0_max_snr_15)
#zr2017_path=/private/home/marvinlvn/DATA/CPC_data/test/zerospeech2017/data/test
##test_languages=(english french)
#test_languages=(english french)
#model_dir=/checkpoint/marvinlvn/megamix/noise_study/cpc
##n_components=(10 50 500 800 1000)
#n_components=(10 50 500)
#noise_study_path=/checkpoint/marvinlvn/noise_study
#
#for model_type in ${model_types[*]}; do
#  for size in ${sizes[*]}; do
#    for language in ${languages[*]}; do
#      for test_language in ${test_languages[*]}; do
#        for n_component in ${n_components[*]}; do
#          DATA_PATH=${zr2017_path}/${test_language}/1s
#
#          # First line turned on : run ABX original (zr 2017)
#          # Second line turned on : run ABX clean (juliette)
#          ITEM_PATH=${DATA_PATH}/1s.item
#          #ITEM_PATH=/private/home/marvinlvn/DATA/CPC_data/test/zerospeech2017_ju/${test_language}/1s_clean/1s.item
#          FEAT_DIR=/checkpoint/marvinlvn/megamix/noise_study/feature/cpc/${model_type}/${n_component}/${language}/${size}/test_on_${test_language}
#          MODEL_PATH=${model_dir}/${model_type}/${n_component}/${language}/${size}_nb_0
#          if [ $model_type == online_gmm_kappa_0_5_window_1024 ]; then
#            model_type2=online_gmm
#          elif [[ $model_type == skgmm_* ]]; then
#            model_type2=skgmm
#          else
#            model_type2=${model_type/_/}
#          fi;
#
#          if [ $model_type2 == 'kmeans' ]; then
#            script=run_discrete_ABX.sh
#          else
#            script=run_ABX.sh
#          fi;
#
#          CPC_PATH=${noise_study_path}/${language}_${size}_nb_0
#          BEST_EPOCH=$(python /private/home/marvinlvn/CPC_audio_jzay/utils/best_val_epoch.py --model_path ${CPC_PATH} | grep -oP "(?<=is : )([0-9]+)")
#          CPC_PATH=${CPC_PATH}/checkpoint_${BEST_EPOCH}.pt
#          CPC_ARG="--cpc_path $CPC_PATH"
#          sbatch -o logs/cpc_${model_type}_${language}_${size}_${n_component}_${test_language}.txt ${script} ${DATA_PATH} ${FEAT_DIR} ${MODEL_PATH} ${ITEM_PATH} ${model_type2} cpc 0.01 kl_symmetric 16000 $CPC_ARG
#        done;
#      done;
#    done;
#  done;
#done;

#model_types=(dpgmm gmm)
#model_types=(online_gmm_kappa_0_5_window_1024 dpgmm_ gmm_)
#model_types=(skgmm_full_cov skgmm_diag_cov skgmm_spherical_cov)
model_types=(skgmm_spherical_cov)

# OOM for 256h during training :/
sizes=(10 20 40 80 150)
#languages=(English_LibriVox_extracted_full_random French_LibriVox_extracted_full_random)
languages=(FRENCH_MAL_FEM_gt_1500ms ACLEW10K_MAL_FEM_gt_1500ms)
zr2017_path=/gpfsscratch/rech/cfs/commun/zerospeech2017/data/test
#test_languages=(english french)
test_languages=(english french)
model_dir=/linkhome/rech/genscp01/uow84uh/CPC_audio_jzay
n_components=(100 150)

for model_type in ${model_types[*]}; do
  for size in ${sizes[*]}; do
    for language in ${languages[*]}; do
      for test_language in ${test_languages[*]}; do
        for n_component in ${n_components[*]}; do
          DATA_PATH=${zr2017_path}/${test_language}/1s

          #ITEM_PATH=/private/home/marvinlvn/DATA/CPC_data/test/zerospeech2017_ju/${test_language}/1s_clean/1s.item
          DATA_PATH=${zr2017_path}/${test_language}/1s
          if [ "$test_language" == "french" ]; then
            ITEM_PATH=${zr2017_path}/${test_language}/1s_clean_marvin/1s_clean_marvin.item
            FEAT_DIR=/gpfsscratch/rech/xdz/uow84uh/DATA/feature/megamix/daylong/cpc/${model_type}/${n_component}/${language}/${size}h/test_on_${test_language}_clean_marvin
          else
            ITEM_PATH=${zr2017_path}/${test_language}/1s/1s.item
            FEAT_DIR=/gpfsscratch/rech/xdz/uow84uh/DATA/feature/megamix/daylong/cpc/${model_type}/${n_component}/${language}/${size}h/test_on_${test_language}
          fi;

          MODEL_PATH=${model_dir}/daylong_clustering/cpc/${model_type}/${n_component}/${language}/${size}h
          if [ $model_type == online_gmm_kappa_0_5_window_1024 ]; then
            model_type2=online_gmm
          elif [[ $model_type == skgmm_* ]]; then
            model_type2=skgmm
          else
            model_type2=${model_type/_/}
          fi;

          if [ $model_type2 == 'kmeans' ]; then
            script=run_discrete_ABX.sh
          else
            script=run_ABX.sh
          fi;

          CPC_PATH=${model_dir}/${language}_${size}h
          BEST_EPOCH=$(python /linkhome/rech/genscp01/uow84uh/CPC_audio_jzay/utils/best_val_epoch.py --model_path ${CPC_PATH} | grep -oP "(?<=is : )([0-9]+)")
          CPC_PATH=${CPC_PATH}/checkpoint_${BEST_EPOCH}.pt
          CPC_ARG="--cpc_path $CPC_PATH"
          echo sbatch -o logs/cpc_${model_type}_${language}_${size}_${n_component}_${test_language}.txt run_ABX_jzay.sh ${DATA_PATH} ${FEAT_DIR} ${MODEL_PATH} ${ITEM_PATH} ${model_type2} cpc 0.01 kl_symmetric 16000 $CPC_ARG
        done;
      done;
    done;
  done;
done;