#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licensy tytes/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import sys
import json
import time
from dataclasses import dataclass, field
from typing import Optional

import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
import evaluate
from uie_collator import DataCollatorForImages

from existing_methods.agem import AGEM
from existing_methods.er import ER
from existing_methods.ewc import EWC

import tqdm
import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,  # add
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    TrainingArguments,
    set_seed, )
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint
from peft import get_peft_config, get_peft_model, LoraConfig
import os
import torch

from SAM_Trainer_new import SAMContinualTrainer
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig  # add

from peft.utils import PromptLearningConfig

from sam import SAM, enable_running_stats, disable_running_stats

from model.resnet import ResNetForImageClassification_WithLossMask
from model.vit import ViTForImageClassification_WithLossMask

from pt_data import load_imag_dataset

# off wandb
os.environ['WANDB_DISABLED'] = "True"
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
logger = logging.getLogger(__name__)
CURRENT_DIR = os.path.dirname(__file__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

@dataclass
class ModelArguments:
    base_model_name: str = field(default=None)
    model_name_or_path: str = field(default=None)
    lora_type: str = field(default=None)  # ['lora', 'olora', 'nlora']

@dataclass
class DataTrainingArguments:
    data_dir: str = field(default=None)
    task_config_dir: str = field(default=None)
    task_id: int = field(default=0)

@dataclass
class ImageTrainingArguments(TrainingArguments):
    do_train: bool = field(default=False)
    train_method: str = field(default="finetune")  # 'lora', 'finetune', 'continual'
    finetune_strategy: str = field(default="full")  # 'full', 'ewc', 'er'
    optimizer_type: str = field(default="sam")  # 'sgd' 或 'sam'
    save_task_models: bool = field(default=True)
    do_flatminal: bool = field(default=False)
    eval_after_train: bool = field(default=True)

    momentum: float = field(default=0.9)          # 仅用于 sgd 或 sam
    rho: float = field(default=0.05)              # sam 特有超参数
    gamma: float = field(default=1.0)  # ✅ 添加这个即可（默认 1.0 表示不变学习率）
   


def freeze_model_params(model, strategy):
    if strategy == "full":
        for _, param in model.named_parameters():
            param.requires_grad = True
    elif strategy == "ewc" or strategy == "er":
        for name, param in model.named_parameters():
            param.requires_grad = ("classifier" in name)

    elif strategy == "lora":
        for name, param in model.named_parameters():
            if not ("lora_" in name or "adapter" in name):  # 覆盖适配你实际模型的名称
                param.requires_grad = False
    else:
        raise ValueError(f"Unknown finetune strategy: {strategy}")

def build_lora_config(lora_type):
    base_cfg = dict(
        task_type=TaskType.IMAGE_CLASSIFICATION,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    if lora_type == "nlora":
        base_cfg["use_nlora"] = True
    elif lora_type == "olora":
        base_cfg["use_olora"] = True
    return LoraConfig(**base_cfg)

def main():
    # ===== 1. 解析参数 =====
    logger.info("🔧 解析命令行参数...")
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, ImageTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    logger.debug(f"✅ model_args: {model_args}")
    logger.debug(f"✅ data_args: {data_args}")
    logger.debug(f"✅ training_args: {training_args}")

    # Setup logging
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # 如果不想输出 log 信息将数值改为20即可
    # log_level = training_args.get_process_log_level()
    log_level = 10  # 或者 logging.DEBUG（数值 10）
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # ===== 2. 加载数据 =====
    logger.info("📦 开始加载任务划分与数据...")
    task_dir = os.path.join(data_args.data_dir, data_args.task_config_dir)
    task_split_path = os.path.join(task_dir, "task_split.json")
    saved_all_tasks = None
    if os.path.exists(task_split_path):
        with open(task_split_path) as f:
            task_all_split = json.load(f)
        if data_args.task_id != -1:
            saved_all_tasks = [task_all_split[data_args.task_id]]
            logger.info(f"🧩 使用已保存任务 task_id={data_args.task_id}, 类别={saved_all_tasks[0]}")
        else:
            saved_all_tasks = task_all_split
            logger.info("🧩 使用全部任务进行加载")
    else:
        os.makedirs(task_dir, exist_ok=True)
        logger.warning("⚠️ 未找到任务划分文件，将重新生成任务划分")

    logger.info("🔄 调用数据加载函数 load_imag_dataset...")
    dataloaders, task_split, num_classes, classes_per_task, subset_dataloaders = load_imag_dataset(
        dataset=data_args.task_config_dir,
        data_folder=task_dir,
        batch_size=5,
        val=training_args.do_eval,
        saved_tasks=saved_all_tasks,
        get_subset=False
    )
    logger.info(f"✅ 数据加载完成，num_classes={num_classes}, classes_per_task={classes_per_task}")

    # ===== 3. 加载模型结构（使用 num_classes）=====
    logger.info("🧠 开始加载模型结构...")
    logger.debug(f"📄 加载模型 config model.args:{model_args}")

    # Step 1: 加载 config
    
    logger.info(f"🔍 当前加载模型路径: {model_args.model_name_or_path}")
    logger.info(f"🔧 使用配置路径: {model_args.base_model_name}")
    if model_args.model_name_or_path.endswith(".pt"):
        assert model_args.base_model_name, "--base_model_name is required when loading a .pt file"
        config = AutoConfig.from_pretrained(model_args.base_model_name)
    elif "adapter" in model_args.model_name_or_path:
        config = PeftConfig.from_pretrained(model_args.model_name_or_path)
    else:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path)

    # Step 2: 设置模型结构（model_class）
    logger.debug("📐 设置模型结构 model_class ,model_args.base_model_name:{model_args.base_model_name}")
    if "resnet" in model_args.base_model_name.lower():
        model_class = ResNetForImageClassification_WithLossMask
    elif "vit" in model_args.base_model_name.lower():
        model_class = ViTForImageClassification_WithLossMask
    else:
        raise ValueError("Unsupported model class")

    # Step 3: 加载模型
    logger.info(f"📥 加载模型，方法 = {training_args.train_method}")
    if training_args.train_method == "finetune":
        if model_args.model_name_or_path.endswith(".pt"):
            logger.info("📦 从 .pt 文件加载微调模型")
            ckpt = torch.load(model_args.model_name_or_path, map_location="cpu")
            classifier_out_dim = ckpt.get("classifier_out_dim", 100)
            model = model_class.from_pretrained(model_args.base_model_name, config=config)
            model.load_state_dict(ckpt["model"])
        else:
            logger.info("🌐 从 HuggingFace 或本地路径加载模型")
            model = model_class.from_pretrained(model_args.model_name_or_path, config=config)
            classifier_out_dim = model.config.num_labels

        freeze_model_params(model, training_args.finetune_strategy)
        logger.info("🔒 Finetune 模型参数处理完成")

    elif training_args.train_method == "lora":
        logger.info("➕ 加载 LoRA 结构")
        base = model_class.from_pretrained(model_args.base_model_name, config=config)
        if "adapter" in model_args.model_name_or_path:
            model = PeftModel.from_pretrained(base, model_args.model_name_or_path)
            logger.info("✅ 加载已训练好的 adapter")
        else:
            lora_cfg = build_lora_config(training_args, model_args.lora_type)
            model = get_peft_model(base, lora_cfg)
            logger.info("🧪 初始化新的 LoRA adapter")

        freeze_model_params(model, training_args.finetune_strategy)
        logger.info("🔒 LoRA 模型参数处理完成")

    else:
        raise ValueError(f"Unsupported training method: {training_args.train_method}")

    # === 设置分类器 ===
    logger.info("🔧 设置分类头（classifier）")
    model.classifier = torch.nn.Linear(model.config.hidden_size, num_classes).to(training_args.device)
    model.config.num_labels = num_classes
    logger.debug(f"✅ 分类器结构: {model.classifier}")

    # ===== 4. 初始化 Trainer =====
    logger.info("🚀 初始化 SAMContinualTrainer...")
    trainer = SAMContinualTrainer(
        model=model,
        dataloaders=dataloaders,
        task_split=task_split,
        num_classes=num_classes,
        classes_per_task=classes_per_task,
        training_args=training_args,  # 使用命名参数传入完整 args
        criterion=None,
        algo=None,
        subset_dataloaders=subset_dataloaders,
        task_id=data_args.task_id
    )
    logger.info("✅ Trainer 初始化完成")

    # ===== 5. 启动训练 =====
    if training_args.do_train:
        logger.info("🚦 开始训练过程")
        if data_args.task_id != -1:
            logger.info(f"▶️ 训练单个任务：task_id={data_args.task_id}")
            trainer.train_onlyonetask(data_args.task_id)
        else:
            logger.info("▶️ 连续训练所有任务")
            trainer.train_cltask()

    # ===== 6. 执行 flatness 分析 =====
    if training_args.do_flatminal:
        logger.info("📊 开始损失曲面分析 compute_loss_landscape()...")
        trainer.compute_loss_landscape(eval_task_id=0, output_dir=training_args.output_dir)
        trainer.compute_loss_landscape_version2(eval_task_id=0, output_dir=training_args.output_dir)
        logger.info("✅ 损失曲面分析完成")




if __name__ == "__main__":
    main()
