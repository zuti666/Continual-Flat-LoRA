#!/bin/bash
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"

echo $TRANSFORMERS_CACHE

port=$(shuf -i25000-30000 -n1)
 
# bash scripts/order_1.sh> logs_and_outputs/order_1/logs/train_and_infer.log 2>&1 &



CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $port src/image_sam.py \
   --base_model_name google/vit-base-patch16-224-in21k \
   --model_name_or_path logs-outputs_ViT/order_1/outputs/models/task_0_model.pt \
   --do_train True \
   --do_predict False \
   --do_flatminal True \
   --finetune True \
   --lora False \
   --task_config_dir cifar100 \
   --data_dir  CL_Image_CLass\
   --task_id 1 \
   --save_task_models True \
   --eval_afterTrain True \
   --output_dir logs-outputs_ViT/order_1/outputs \
   --method None \
   --sam True \
   --class_incremental True \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 1 \
   --learning_rate 1e-03 \
   --num_train_epochs 2 \
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
