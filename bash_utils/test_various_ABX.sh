
# This script aims at computing ABX error rate on the ZR 2017 CPC models
# for various parameters

lengths=(1s 10s 120s)
zr2017_path=/private/home/marvinlvn/DATA/CPC_data/test/zerospeech2017/data/test/english
CPC_ARG="--cpc_path /private/home/marvinlvn/zr-2021vg_baseline/zr2021_models/checkpoints/CPC-small-kmeans50/cpc_ls100/checkpoint_170.pt"

for length in ${lengths[*]}; do
  DATA_PATH=${zr2017_path}/${length}
  ITEM_PATH=${DATA_PATH}/${length}.item
  FEAT_DIR=/checkpoint/marvinlvn/megamix/feature/test_length/cpc/english/${length}_max_size_seq_${length}_slide_1
  MAX_SIZE_SEQ=${length/s/}
  MAX_SIZE_SEQ=$(($MAX_SIZE_SEQ*16000))
  #sbatch -o logs/test_ABX/cpc_${length}_max_size_seq_${length}_slide_1.txt cpc_varying_length.sh $DATA_PATH $FEAT_DIR $ITEM_PATH cpc 0.01 cosine $MAX_SIZE_SEQ 1 $CPC_ARG
done;

#slides=(0.25 0.5 0.75)
slides=(1)
for slide in ${slides[*]}; do
  DATA_PATH=${zr2017_path}/10s
  ITEM_PATH=${DATA_PATH}/10s.item
  FEAT_DIR=/checkpoint/marvinlvn/megamix/feature/test_length/cpc/english/10s_max_size_seq_1.28s_slide_${slide}
  MAX_SIZE_SEQ=20480
  sbatch -o logs/test_ABX/cpc_10s_max_size_seq_1.28s_slide_${slide}.txt cpc_varying_length.sh $DATA_PATH $FEAT_DIR $ITEM_PATH cpc 0.01 cosine $MAX_SIZE_SEQ $slide $CPC_ARG
done;