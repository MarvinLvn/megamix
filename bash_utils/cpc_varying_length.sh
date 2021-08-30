#!/bin/bash
#SBATCH --partition=learnfair
#SBATCH --nodes=1                     # nombre de noeud
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=20:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --mem=512000

if [[ "$#" -lt 6 ]]; then
  echo "./eval/run_ABX.sh <DATA_PATH> <OUTPUT_PATH> <MODEL_PATH> <ITEM_PATH> <AUDIO_EXTENSION> <FEAT_SIZE> (--skip-embeddings)"
  exit
fi

cd ~/megamix
DATA_PATH=$1
PATH_FEATURE_DIR=$2
ITEM_PATH=$3
FEAT_TYPE=$4
FEAT_SIZE=$5
DIST=$6
MAX_SIZE_SEQ=$7
SLIDE=$8
CPC_ARG="${9} ${10}"

source activate megamix
if [[ ! $9 == "--skip-embeddings" ]]; then
    # 1) Extract embeddings
    start=`date +%s`
    python speech/extract_posteriors.py --input $DATA_PATH \
        --output $PATH_FEATURE_DIR --type $FEAT_TYPE --max_size_seq $MAX_SIZE_SEQ --slide $SLIDE $CPC_ARG
    end=`date +%s`
    runtime=$((end-start))
    echo "Took $runtime sec for extracting embeddings."
fi;

start=`date +%s`
# 2) Compute ABX
dirname1=$(basename ${DATA_PATH})
dirname2=$(basename $(dirname ${DATA_PATH}))
bn_item_path=$(basename $ITEM_PATH)
bn_item_path=${bn_item_path/./_}
OUTPUT_DIR=${PATH_FEATURE_DIR}/ABX/${dirname1}_${dirname2}_${bn_item_path}_${DIST}

if [[ ! -d $OUTPUT_DIR ]]; then
    mkdir -p $OUTPUT_DIR
fi;

source activate libri-light
python /private/home/marvinlvn/fairseq_jzay/libri-light/eval/eval_ABX.py ${PATH_FEATURE_DIR} $ITEM_PATH \
    --file_extension .pt --out $OUTPUT_DIR --feature_size $FEAT_SIZE --cuda --distance_mode $DIST
rm $PATH_FEATURE_DIR/*.pt

end=`date +%s`
runtime=$((end-start))
echo "Took $runtime sec for computing ABX on the 3 feature sets."