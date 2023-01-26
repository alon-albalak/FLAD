GPU=$1
MODEL="google/t5-xl-lm-adapt"
MODEL_NAME="T5_LM_3B"
LOG_SAVE_EVAL_STEPS=10

delete_unnecesary_files(){
    FIND_DIR=$1

    find $FIND_DIR -name "pytorch_model*.bin" -delete
    find $FIND_DIR -name "spiece.model" -delete
    find $FIND_DIR -name "tokenizer_config.json" -delete
    find $FIND_DIR -name "special_tokens_map.json" -delete
    find $FIND_DIR -name "tokenizer.json" -delete
    find $FIND_DIR -name "pytorch_model.bin.index.json" -delete
}

target_only_train(){
    TARGET_DATASET=$1
    NUM_SHOT=$2
    LR=$3
    SEED=$4

    OUTPUT_DIR="outputs/${MODEL_NAME}/target_only/$SEED/${TARGET_DATASET}/$LR"
    mkdir -p $OUTPUT_DIR
    result_file="${OUTPUT_DIR}/eval_results.json"
    if [ -f "$result_file" ]; then
        echo "$result_file exists. Continuing"
    else
        echo $(date)
        echo "Training on $TARGET_DATASET seed ${SEED} with $NUM_SHOT shots and $LR learning rate."
        echo "Output dir: $OUTPUT_DIR"
        CUDA_VISIBLE_DEVICES=$GPU python3 src/run_train.py \
            --do_train \
            --do_eval \
            --model_name_or_path $MODEL \
            --train_strategy "target_only" \
            --target_dataset $TARGET_DATASET \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 128 \
            --gradient_accumulation_steps 2 \
            --gradient_checkpointing true \
            --output_dir $OUTPUT_DIR \
            --overwrite_output_dir true \
            --predict_with_generate true \
            --max_steps 1000 \
            --evaluation_strategy "steps" \
            --eval_steps $LOG_SAVE_EVAL_STEPS \
            --save_total_limit 1 \
            --save_steps $LOG_SAVE_EVAL_STEPS \
            --patience 10 \
            --eval_delay 50 \
            --load_best_model_at_end true \
            --metric_for_best_model "accuracy" \
            --logging_strategy "steps" \
            --logging_steps $LOG_SAVE_EVAL_STEPS \
            --log_samples_per_dataset true \
            --bf16 \
            --learning_rate $LR \
            --dataloader_num_workers 0 \
            --optim "adafactor" \
            --warmup_ratio 0.1 \
            --lr_scheduler_type "constant_with_warmup" \
            --num_shot $NUM_SHOT \
            --few_shot_random_seed $SEED \
            --tf32 true \
            > $OUTPUT_DIR/log.log 2> $OUTPUT_DIR/log.err
    fi
    delete_unnecesary_files $OUTPUT_DIR
}

TARGET_DATASETS=(
    "anli-r1"
    "anli-r2"
    "anli-r3"
    "cb"
    "copa"
    "h-swag"
    "rte"
    "story_cloze"
    "wic"
    "winogrande"
    "wsc"
)
NUM_SHOTS=(
    50
    50
    50
    32
    32
    20
    32
    70
    32
    50
    32
)
LRS=( 3e-4 1e-4 )
SEEDS=( 42 100 222 3456 5876 )

for i in "${!TARGET_DATASETS[@]}"; do
    TARGET_DATASET=${TARGET_DATASETS[$i]}
    NUM_SHOT=${NUM_SHOTS[$i]}
    for LR in ${LRS[@]}; do
        for SEED in ${SEEDS[@]}; do
            target_only_train $TARGET_DATASET $NUM_SHOT $LR $SEED
        done
    done
done