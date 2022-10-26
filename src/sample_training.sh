
LOG_SAVE_EVAL_STEPS=100
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 run_train.py \
CUDA_VISIBLE_DEVICES=0,1 deepspeed run_train.py \
    --deepspeed ds_config.json \
    --do_train \
    --do_eval \
    --auxiliary_dataset T0Mixture \
    --target_dataset rte \
    --model_name_or_path google/t5-v1_1-base \
    --output_dir outputs/debug \
    --overwrite_output_dir true \
    --predict_with_generate true \
    --max_steps 100000 \
    --evaluation_strategy steps \
    --eval_steps $LOG_SAVE_EVAL_STEPS \
    --save_total_limit 1 \
    --save_steps $LOG_SAVE_EVAL_STEPS \
    --patience 5 \
    --load_best_model_at_end true \
    --metric_for_best_model accuracy \
    --logging_strategy steps \
    --logging_steps $LOG_SAVE_EVAL_STEPS \
    --log_samples_per_dataset true \
    --bf16 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 3e-4 \
    --dataloader_num_workers 0 \
    --optim adafactor \
    --ddp_find_unused_parameters false