GPU=$1
TARGET_DATASET=$2

WEIGHT_INIT_SAMPLES=0
SEEDS=( 42 100 222 3456 5876 )
MODELS=( "google/t5-xl-lm-adapt" "bigscience/T0_3B" "google/t5-base-lm-adapt")
AUX_DATASETS=( "T0Mixture" "P3" )

for SEED in ${SEEDS[@]}; do
    for AUX_DATASET in ${AUX_DATASETS[@]}; do
        for MODEL in ${MODELS[@]}; do
            OUTPUT_DIR="outputs/train_logs/exp3/$SEED/$MODEL/${AUX_DATASET}/${TARGET_DATASET}/${WEIGHT_INIT_SAMPLES}"
            mkdir -p $OUTPUT_DIR

            echo $(date)
            echo "Running $SEED $MODEL $AUX_DATASET $TARGET_DATASET $WEIGHT_INIT_SAMPLES EXP3"
            echo "Saving log to ${OUTPUT_DIR}"

            CUDA_VISIBLE_DEVICES=$GPU python src/multirun_train_weighted_batch_exp3.py \
                --seed $SEED \
                --target_dataset $TARGET_DATASET \
                --aux_dataset $AUX_DATASET \
                --model $MODEL \
                --weight_initialization_samples $WEIGHT_INIT_SAMPLES \
                > $OUTPUT_DIR/log.log 2> $OUTPUT_DIR/err.log
        done
    done
done