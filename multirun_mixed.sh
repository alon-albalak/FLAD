GPU=$1
TARGET_DATASET=$2
AUX_DATASET=$3
MODEL=$4

SEEDS=( 42 100 222 3456 5876 )

for SEED in ${SEEDS[@]}; do
    OUTPUT_DIR="outputs/train_logs/mixed_train/$SEED/$MODEL/${AUX_DATASET}/${TARGET_DATASET}"
    mkdir -p $OUTPUT_DIR

    CUDA_VISIBLE_DEVICES=$GPU python src/multirun_train_mixed.py \
        --seed $SEED \
        --target_dataset $TARGET_DATASET \
        --aux_dataset $AUX_DATASET \
        --model $MODEL \
        > $OUTPUT_DIR/log.log 2> $OUTPUT_DIR/err.log
done