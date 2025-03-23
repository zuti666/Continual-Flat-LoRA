#!/bin/bash
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"

echo $TRANSFORMERS_CACHE

port=$(shuf -i25000-30000 -n1)
 
# bash scripts/order_1.sh> logs_and_outputs/order_1/logs/train_and_infer.log 2>&1 &

# 一、模型加载相关参数
# --base_model_name "google/vit-base-patch16-224-in21k"   # 加载 config 和结构（必填）
# --model_name_or_path "..."                               # 可以是 HF 模型名、.pt 路径或 adapter 文件夹

# 二、训练控制参数
# --do_train True                       # 是否训练
# --train_method "finetune"            # ['finetune', 'lora', 'continual']
# --finetune_strategy "ewc"            # ['full', 'ewc', 'er'] 仅 train_method=finetune 时生效
# --lora_type "lora"                   # ['lora', 'olora', 'nlora'] 仅 train_method=lora 时生效
# --optimizer_type "sam"               # ['sgd', 'sam']
# --output_dir "..."                   # 保存输出路径
# --save_task_models True              # 是否保存每轮任务的模型

# 三、数据加载参数
# --data_dir "./CL_Image_Classification"
# --task_config_dir "cifar100"
# --task_id 0                          # 当前任务 id；-1 表示所有任务



# 四、评估 & 分析
# --do_predict True                    # 是否评估
# --do_flatminal True                  # 是否进行损失曲面分析
# --eval_after_train True             # 每轮训练后是否评估

# 五 具体 训练数据设置



# ✅ 构建参数数组，保留参数注释说明（不要在续行中添加注释符，否则会导致解析错误）
# ✅ 构建参数数组（只保留被 argparse 正确识别的参数）
ARGS=(
  # ===== 模型加载相关 =====
  --base_model_name google/vit-base-patch16-224-in21k
  --model_name_or_path google/vit-base-patch16-224-in21k

  # ===== 模型训练设置 =====
  --do_train True
  --train_method finetune
  --finetune_strategy full
  --optimizer_type sam
  --save_task_models True
  --output_dir logs-outputs_ViT/order_1/outputs

  # ===== 数据加载设置 =====
  --data_dir CL_Image_CLass
  --task_config_dir cifar100
  --task_id 1

  # ===== 模型评估与分析 =====
  --do_predict False
  --do_flatminal True
  --eval_after_train True

  # ===== 通用训练超参数 =====
  --per_device_train_batch_size 16
  --per_device_eval_batch_size 64
  --gradient_accumulation_steps 1
  --learning_rate 1e-3
  --num_train_epochs 2
  --run_name image_test

  # ===== 训练过程策略 =====
  --overwrite_output_dir
  --lr_scheduler_type constant
  --warmup_steps 0
  --logging_strategy steps
  --logging_steps 10
  --evaluation_strategy no
  --save_strategy no
  --save_steps 1500
)

# ✅ 启动训练脚本
CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $port src/image_sam_new.py "${ARGS[@]}"
sleep 5