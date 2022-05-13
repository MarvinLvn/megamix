#!/bin/bash
#SBATCH --account=xdz@gpu
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1                # nombre de GPU a reserver
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread

module load libsndfile
if [[ "$#" -lt 8 ]]; then
  echo "./eval/run_ABX.sh <DATA_PATH> <OUTPUT_PATH> <MODEL_PATH> <ITEM_PATH> <AUDIO_EXTENSION> <FEAT_SIZE> (--skip-embeddings)"
  exit
fi

cd /linkhome/rech/genscp01/uow84uh/CPC_audio_jzay/megamix
DATA_PATH=$1
PATH_FEATURE_DIR=$2
MODEL_PATH=$3
ITEM_PATH=$4
MODEL_TYPE=$5
FEAT_TYPE=$6
FEAT_SIZE=$7
DIST=$8
MAX_SIZE_SEQ=$9
CPC_ARG="${10} ${11}"

source activate megamix
if [[ ! $9 == "--skip-embeddings" ]]; then
    # 1) Extract embeddings
    start=`date +%s`
    python speech/extract_posteriors.py --input $DATA_PATH \
        --output $PATH_FEATURE_DIR --model $MODEL_PATH --model_type $MODEL_TYPE --type $FEAT_TYPE \
        --not_log $CPC_ARG --window 1024 --kappa 0.5 --max_size_seq $MAX_SIZE_SEQ
    end=`date +%s`
    runtime=$((end-start))
    echo "Took $runtime sec for extracting embeddings."
fi;

start=`date +%s`
# 2) Compute ABX
dirname1=$(basename $(dirname ${ITEM_PATH}))
dirname2=$(basename $(dirname ${DATA_PATH}))
bn_item_path=$(dirname $(basename $ITEM_PATH))
bn_item_path=${bn_item_path/./_}
OUTPUT_DIR=${MODEL_PATH}/ABX/${dirname1}_${dirname2}_${bn_item_path}_${DIST}
if [[ ! -d $OUTPUT_DIR ]]; then
    mkdir -p $OUTPUT_DIR
fi;

source activate libri-light
python /linkhome/rech/genscp01/uow84uh/fairseq_jzay/libri-light/eval/eval_ABX.py ${PATH_FEATURE_DIR} $ITEM_PATH \
    --file_extension .pt --out $OUTPUT_DIR --feature_size $FEAT_SIZE --cuda --distance_mode $DIST
#rm -rf $PATH_FEATURE_DIR

end=`date +%s`
runtime=$((end-start))
echo "Took $runtime sec for computing ABX on the 3 feature sets."