lengths=(1s 10s 120s)
zr2017_path=/private/home/marvinlvn/DATA/CPC_data/test/zerospeech2017/data/test/english
#feat_types=(mfcc cpc)
feat_types=(cpc)
for length in ${lengths[*]}; do
  for feat_type in ${feat_types[*]}; do
    if [ $feat_type == "cpc" ]; then
      CPC_ARG="--cpc_path /private/home/marvinlvn/zr-2021vg_baseline/zr2021_models/checkpoints/CPC-small-kmeans50/cpc_ls100/checkpoint_170.pt"
    else
      CPC_ARG=""
    fi;
      DATA_PATH=${zr2017_path}/${length}
      ITEM_PATH=${DATA_PATH}/${length}.item
      FEAT_DIR=/checkpoint/marvinlvn/megamix/feature/test_length/${feat_type}/english/${length}_max_frames_10240
      sbatch -o logs/test_length/${feat_type}_${length}_10240.txt cpc_varying_length.sh $DATA_PATH $FEAT_DIR $ITEM_PATH ${feat_type} 0.01 cosine 10240 1 $CPC_ARG
  done;
done;