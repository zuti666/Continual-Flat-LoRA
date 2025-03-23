#!/bin/bash
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"

echo $TRANSFORMERS_CACHE

port=$(shuf -i25000-30000 -n1)
 


ARGS=(
  --base_model_name google/vit-base-patch16-224-in21k
  --model_name_or_path google/vit-base-patch16-224-in21k

  --do_train True
  --train_method lora
  --lora_type lora
  --optimizer_type sgd

  --do_flatminal True
  --eval_after_train True
  --save_task_models True

  --task_id 0
  --data_dir CL_Image_CLass
  --task_config_dir cifar100
  --output_dir logs_lora_sgd_task0

  --per_device_train_batch_size 16
  --per_device_eval_batch_size 64
  --gradient_accumulation_steps 1
  --learning_rate 1e-3
  --num_train_epochs 2
  --run_name lora_sgd

  --overwrite_output_dir
  --lr_scheduler_type constant
  --warmup_steps 0
  --logging_strategy steps
  --logging_steps 10
  --evaluation_strategy no
  --save_strategy no
  --save_steps 1500
)

CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $port src/image_sam_new.py "${ARGS[@]}"
sleep 5
