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
    --evaluation_strategy steps \
    --eval_steps 100 \
    --bf16 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 3e-4 \
    --save_strategy no \
    --dataloader_num_workers 4 \
    --optim adafactor \
    --ddp_find_unused_parameters false