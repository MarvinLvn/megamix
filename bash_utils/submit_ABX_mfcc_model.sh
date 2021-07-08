
model_types=(dpgmm gmm)
# OOM for 256h during training :/
sizes=(8h 16h 32h 64h 128h)
languages=(English_LibriVox_extracted_full_random French_LibriVox_extracted_full_random)
zr2017_path=/private/home/marvinlvn/DATA/CPC_data/test/zerospeech2017/data/test
test_languages=(english french)
model_dir=/checkpoint/marvinlvn/megamix/mfccs
n_component=50
for model_type in ${model_types[*]}; do
  for size in ${sizes[*]}; do
    for language in ${languages[*]}; do
      for test_language in ${test_languages[*]}; do
        DATA_PATH=${zr2017_path}/${test_language}/1s
        ITEM_PATH=${DATA_PATH}/1s.item
        FEAT_DIR=/checkpoint/marvinlvn/megamix/feature/mfccs/${model_type}/${n_component}/${language}/${size}/test_on_${test_language}
        MODEL_PATH=${model_dir}/${model_type}/${n_component}/${language}/${size}_nb_0
        sbatch -o logs/${model_type}_${language}_${size}_${n_component}_${test_language}.txt run_ABX.sh ${DATA_PATH} ${FEAT_DIR} ${MODEL_PATH} ${ITEM_PATH} ${model_type} mfcc 0.01 kl_symmetric
      done;
    done;
  done;
done;