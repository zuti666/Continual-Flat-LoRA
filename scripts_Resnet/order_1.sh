#!/bin/bash
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"

echo $TRANSFORMERS_CACHE

port=$(shuf -i25000-30000 -n1)
 
# bash scripts/order_1.sh> logs_and_outputs/order_1/logs/train_and_infer.log 2>&1 &



CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $port src/image_sam.py \
   --do_train True \
   --do_predict False \
   --do_flatminal False \
   --finetune True \
   --model_name_or_path microsoft/resnet-18 \
   --task_config_dir cifar100 \
   --data_dir  CL_Image_CLass\
   --output_dir logs-outputs_resnet18/order_1/outputs \
   --method "ewc" \
   --sam True \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 1 \
   --learning_rate 1e-03 \
   --num_train_epochs 10 \
   --run_name image_test \
   --add_task_name True \
   --add_dataset_name True \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --lamda_1 0.4

sleep 5
