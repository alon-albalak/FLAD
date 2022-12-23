GPU=$1
TARGET_DATASET=$2
AUX_DATASET=$3
MODEL=$4
WEIGHT_INIT_SAMPLES=$5

SEEDS=( 42 100 222 3456 5876 )

for SEED in ${SEEDS[@]}; do
    OUTPUT_DIR="outputs/train_logs/exp3/$SEED/$MODEL/${AUX_DATASET}/${TARGET_DATASET}/${WEIGHT_INIT_SAMPLES}"
    mkdir -p $OUTPUT_DIR

    CUDA_VISIBLE_DEVICES=$GPU python src/multirun_train_weighted_batch_exp3.py \
        --seed $SEED \
        --target_dataset $TARGET_DATASET \
        --aux_dataset $AUX_DATASET \
        --model $MODEL \
        --weight_initialization_samples $WEIGHT_INIT_SAMPLES \
        > $OUTPUT_DIR/log.log 2> $OUTPUT_DIR/err.log
done