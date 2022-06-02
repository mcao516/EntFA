#!/bin/bash

MODEL_NAME_OR_PATH=/huggingface/bart-large
OUTPUT_DIR=/BART_HF_models/TEST/

accelerate launch --config_file training_config.yaml train.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --per_device_train_batch_size 5 \
    --per_device_eval_batch_size 8 \
    --preprocessing_num_workers 16 \
    --num_warmup_steps 500 \
    --train_file train.json \
    --validation_file val.json \
    --learning_rate 5e-5 \
    --polyak_update_lr  0.001 \
    --gradient_accumulation_steps 1 \
    --num_beams 6 \
    --num_train_epochs 12 \
    --output_dir $OUTPUT_DIR;
    # --overwrite_cache true \
    # --dataset_name xsum \
    # --source_prefix "summarize: " \
    # --dataset_config "3.0.0" \
