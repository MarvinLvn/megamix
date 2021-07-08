model_types=(gmm)
sizes=(64h)
languages=(English_LibriVox_extracted_full_random)
n_components=(10 50 100 200) # OOM for 500 and 1000
test_languages=(english)
zr2017_path=/private/home/marvinlvn/DATA/CPC_data/test/zerospeech2017/data/test
model_dir=/checkpoint/marvinlvn/megamix/mfccs

for model_type in ${model_types[*]}; do
  for size in ${sizes[*]}; do
    for language in ${languages[*]}; do
      for n_component in ${n_components[*]}; do
        for test_language in ${test_languages[*]}; do
          DATA_PATH=${zr2017_path}/${test_language}/1s
          ITEM_PATH=${DATA_PATH}/1s.item
          FEAT_DIR=/checkpoint/marvinlvn/megamix/feature/mfccs/${model_type}/${n_component}/${language}/${size}/test_on_${test_language}
          MODEL_PATH=${model_dir}/${model_type}/${n_component}/${language}/${size}_nb_0
          sbatch -o logs/${model_type}_${language}_${size}.txt run_ABX.sh ${DATA_PATH} ${FEAT_DIR} ${MODEL_PATH} ${ITEM_PATH} ${model_type} mfcc 0.01 kl_symmetric
        done;
      done;
    done;
  done;
done;
