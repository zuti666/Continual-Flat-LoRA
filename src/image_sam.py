#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
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
from uie_trainer_lora import ContinualTrainer
from SAM_Trainer import SAMContinualTrainer
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig  # add

from peft.utils import PromptLearningConfig

from sam import SAM, enable_running_stats,disable_running_stats

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
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                    "the model's position embeddings."
        },
    )
    # added for AutoCL
    # 修改这个参数
    lora_dim: Optional[int] = field(
        default=4,  # 8
        metadata={
            "help": "Intrinsic dimension of the latent space."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    lang: str = field(default=None, metadata={
                      "help": "Language id for multilingual model."})
    data_dir: str = field(
        default=None, metadata={"help": "The directory for saving the UIE train/dev/test splits."}
    )
    task_config_dir: str = field(
        default=None, metadata={"help": "The json file for config training and testing tasks"}
    )
    instruction_file: str = field(
        default=None, metadata={"help": "The instruction file for different tasks."}
    )
    instruction_strategy: Optional[str] = field(
        default='single', metadata={
            "help": "How many different instructions to use? Support 'single' and 'multiple' mode."
        }
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    input_record_file: str = field(
        default=None, metadata={"help": "file to record model input"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    
    max_source_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    # for decoder model, it means max_new_tokens
    max_target_length: Optional[int] = field(
        default=50,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    repetition_penalty: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "Penalty for repeat tokens in decode stage."
        },
    )
    num_beams: Optional[int] = field(
        default=1,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                    "which is used during ``evaluate`` and ``predict``."
        },
    )
    max_num_instances_per_task: int = field(
        default=10000, metadata={"help": "The maximum number of instances we will consider for each training task."}
    )
    max_num_instances_per_eval_task: int = field(
        default=200,
        metadata={
            "help": "The maximum number of instances we will consider for each validation/test task."}
    )

    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    # 添加参数，用来设置 flatminal 的最大数量
    max_flatminal_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )

    num_examples: Optional[int] = field(
        default=0,
        metadata={"help": "number of in-context positive examples."}
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    add_task_name: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to preappend task name before the task input."}
    )
    add_dataset_name: Optional[bool] = field(
        default=False,
        metadata={
            "help": "whether to preappend dataset name before the task input."}
    )
    


@dataclass
class UIETrainingArguments(Seq2SeqTrainingArguments):
    gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use computing time to gain more memory"}
    )
    denser_evaluation: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If specifid, the model will do more evaluation at the beginning of training."}
    )
    do_demo: bool = field(default=False, metadata={
                          "help": "Whether to run the model as a demo in the terminal."})
    lamda_1: float = field(default=0.5)
    lamda_2: float = field(default=0)

    # 尝试添加自定义参数 ，do_flatminal 用来指示是否执行flatminal的评估
    do_flatminal: bool = field(default=False, metadata={
                               "help": "Whether to do flatminal."})

    # 尝试添加自定义参数 ，flag_originLoRA 用来指示 当前方法是否为Lora 方法
    flag_originLoRA: bool = field(default=False, metadata={
                                  "help": "Whether to do flatminal."})

    # 尝试添加自定义参数 ，flag_modifiedNLoRA 用来指示 当前方法为NLora方法
    flag_modifiedNLoRA: bool = field(default=False, metadata={
                                     "help": "Whether to do flatminal."})

    # 尝试添加自定义参数 ，flag_modified_fullLoRA 用来指示 当前方法为NLora方法,是否对所有的lora部分进行评估
    flag_modifiedNLoRA_fullLoRA: bool = field(
        default=False, metadata={"help": "Whether to do flatminal."})

    # 尝试添加自定义参数 ，flag_modified_taskLoRA 用来指示 当前方法为NLora方法,是否只对与任务有关的LoRA进行评估
    flag_modifiedNLoRA_taskLoRA: bool = field(
        default=False, metadata={"help": "Whether to do flatminal."})

    # 尝试添加自定义参数 ，flag_disturb_fullModel 用来指示, 是否对整个模型的参数进行扰动
    flag_disturb_fullModel: bool = field(
        default=False, metadata={"help": "Whether to do flatminal."})

    # 尝试添加自定义参数 ，finetune 用来指示, 是否对整个模型进行 微调
    finetune: bool = field(
        default=False, metadata={"help": "Whether to do fintune."})

    # 为了调用实现 人家代码进行融合，对于他的参数 就直接从 之间的参数中替换或者重新定义
    # 1 args.epochs_per_task = training_args.num_train_epochs
    # 2 args.lr = training_args.learning_rate
    # 3 gamma


@dataclass
class ImageTrainingArguments(TrainingArguments):

    # 尝试添加自定义参数 ，do_flatminal 用来指示是否执行flatminal的评估
    do_flatminal: bool = field(default=False, metadata={
                               "help": "Whether to do flatminal."})
    
    # 尝试添加自定义参数 ，finetune 用来指示, 是否对整个模型进行 微调
    finetune: bool = field(
        default=False, metadata={"help": "Whether to do fintune."})
    
    # 尝试添加自定义参数 ，finetune 用来指示, 是否对整个模型进行 微调
    lora : bool = field(
        default=False, metadata={"help": "Whether to do fintune."})
    
    # 尝试添加自定义参数 ，base_model_name 用来指示对应车基础模型的名字
    base_model_name: str = field(
        default=None,
        metadata={"help": "Task split definition"},
    )
    

    # 尝试添加自定义参数 ，finetune 用来指示, 是否对整个模型进行 微调
    eval_afterTrain: bool = field(
        default=True, metadata={"help": "eval_afterTrain."})
    

    save_task_models: bool = field(
        default=True, metadata={"help": "eval_afterTrain."})
    




    
    lamda_1: float = field(default=0.5)
    lamda_2: float = field(default=0)



    task_id: int = field(
        default=-1,
        metadata={"help": "task use data set id"},
    )

    num_excluded_classes: int = field(
        default=267,
        metadata={"help": "Number of excluded classes"},
    )

    task_split: str = field(
        default=None,
        metadata={"help": "Task split definition"},
    )

    runs: int = field(
        default=1,
        metadata={"help": "Number of runs"},
    )

    dry_run: bool = field(
        default=False,
        metadata={"help": "Enable dry-run mode"},
    )

    pretrained: bool = field(
        default=False,
        metadata={"help": "Use pretrained model"},
    )

    run_hs: bool = field(
        default=False,
        metadata={"help": "Run additional hs step"},
    )

    layers: int = field(
        default=18,
        metadata={"help": "Number of layers in the model"},
    )

    pt_type: str = field(
        default=None,
        metadata={"help": "Pretraining type"},
    )

    momentum: float = field(
        default=0.0,
        metadata={"help": "Momentum for optimizer"},
    )

    dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout rate"},
    )

    lambd: int = field(
        default=1,
        metadata={"help": "Lambda parameter for EWC"},
    )

    mem_size: int = field(
        default=1,
        metadata={"help": "Memory size"},
    )

    method: str = field(
        default="sgd",
        metadata={"help": "Continual learning method"},
    )

    class_incremental: bool = field(
        default=False,
        metadata={"help": "Use class-incremental learning"},
    )

    sam: bool = field(
        default=False,
        metadata={"help": "Use Sharpness-Aware Minimization (SAM)"},
    )

    rho: float = field(
        default=0.05,
        metadata={"help": "Neighborhood parameter for SAM"},
    )

    gamma: float = field(
        default=1.0,
        metadata={"help": "lr decay. Use 1.0 for no decay"},
    )



def main():
    """-----------------Load AND process Args-----------------"""
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, ImageTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    """-----------------Set Logger-----------------"""
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

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.debug(f"Training/evaluation parameters {training_args}")

    """-----------------Detecting last checkpoint-----------------"""
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.debug(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    """-----------------Load and Set Dataset-----------------"""
    logger.debug('----------------Load and Set Dataset-----------------')

    
    task_dir = os.path.join(data_args.data_dir, data_args.task_config_dir)
    task_split_path = os.path.join(task_dir, "task_split.json")

    # 确保目录存在
    os.makedirs(task_dir, exist_ok=True)
    logger.debug(f'task_id: {training_args.task_id}, task_split_path: {task_split_path}, exists: {os.path.exists(task_split_path)}')

    # 检查是否已有任务划分
    if os.path.exists(task_split_path):
        saved_all_tasks = []
        with open(task_split_path, "r") as f:
            task_all_split = json.load(f)

        if training_args.task_id != -1 : # 只加载某一个任务
        
            if training_args.task_id < len(task_all_split):  # 确保 task_id 不超出范围
                saved_all_tasks.append(task_all_split[training_args.task_id])

                logger.info(f"加载已有任务划分，任务 {training_args.task_id} 类别: {saved_all_tasks[training_args.task_id]}")

            else:
                raise ValueError(f"task_id {training_args.task_id} 超出任务划分范围（最大值 {len(task_all_split) - 1}）")

        else: # Continual Learnig 训练所有任务

            for t in range(len(task_all_split)):
                saved_all_tasks.append(task_all_split[t])

                logger.info(f"加载已有任务划分，任务 {t} 类别: {saved_all_tasks[t]}")
        
    else:
        saved_all_tasks = None
        logger.info("未找到任务划分文件，重新生成任务划分。")

    # 加载数据 dataset, data_folder, batch_size, val, saved_tasks
    logger.debug(f'dataset={data_args.data_dir}, data_folder={task_dir}, batch_size={5}, val={training_args.do_eval}, saved_all_tasks={saved_all_tasks}')
    
    dataloaders, task_split, num_classes, classes_per_task, subset_dataloaders = load_imag_dataset(
        dataset=data_args.task_config_dir, data_folder=task_dir, batch_size=5, val=training_args.do_eval, saved_tasks=saved_all_tasks, get_subset=True
    )

    # 如果是新生成的任务划分，则保存到文件
    if not os.path.exists(task_split_path):
        with open(task_split_path, "w") as f:
            json.dump(task_split, f, indent=2)
        logger.info(f"新任务划分已保存至 {task_split_path}")

    # 获取任务 `task_id` 的训练和测试数据
    # if training_args.do_train:
    #     train_task_loader = dataloaders[task_id]["train"]
    #     train_dataset = dataloaders[task_id]["train"].dataset
    #     logger.info(f"成功加载任务 {task_id} 训练数据, 类别: {task_split[task_id]}")

    #     logger.info(f"训练集 `train_dataset` 类型: {type(train_dataset)}")
    #     logger.info(f"训练集大小: {len(train_dataset)}")
        
    #     # 取前 3 个样本
    #     for i in range(min(3, len(train_dataset))):
    #         sample = train_dataset[i]
    #         logger.info(f"🔹 训练样本 {i}: {sample}")


    #     labels_in_dataset = [train_dataset[i][1] for i in range(len(train_dataset))]
    #     logger.info(f"任务 {task_id} 的标签范围: {set(labels_in_dataset)}")


    # if training_args.do_eval:
    #     eval_task_loader = dataloaders[task_id]["test"]
    #     eval_dataset = dataloaders[task_id]["test"].dataset
    #     logger.info(f"成功加载任务 {task_id} 测试数据, 类别: {task_split[task_id]}")
    #     logger.info(f" 测试集 `eval_dataset` 类型: {type(eval_dataset)}")
    #     logger.info(f" 测试集大小: {len(eval_dataset)}")

    #     # 取前 3 个样本
    #     for i in range(min(3, len(eval_dataset))):
    #         sample = eval_dataset[i]
    #         logger.info(f" 测试样本 {i}: {sample}")


    if training_args.class_incremental:
        num_classes = classes_per_task

    img_data_collator = DataCollatorForImages()

    """  """

    # Load pretrained model and tokenizer

    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    # 选择模型类型
    logger.debug("*** 1 *** Load pretrained model and tokenizer")
    # 加载已经训练好的 Peft 模型
    if "adapter" in model_args.model_name_or_path:
        logger.info(f"*** 1 *** - adapter {model_args.model_name_or_path}")
        config = PeftConfig.from_pretrained(model_args.model_name_or_path)
    # 加载 初始模型
    else:
        logger.info(f"*** 1 *** - default {model_args.model_name_or_path}")
        config = AutoConfig.from_pretrained(model_args.model_name_or_path)

    # 选择 model_class，这里是自己定义的类别，需要自己引入，由于NloRA代码改进了Lora部分，所以需要修改原来的LoRA代码
    # 如果不采用NLoRA，可以直接采用  ResNetForImageClassification 或 iTForImageClassification
    logger.info("*** 2 *** Load model_class")
    if "resnet" in model_args.model_name_or_path.lower():
        model_class = ResNetForImageClassification_WithLossMask
        logger.info("*** 2 ***1 ResNetForImageClassification_WithLossMask")
    elif "vit" in model_args.model_name_or_path.lower():
        model_class = ViTForImageClassification_WithLossMask
        logger.info("*** 2 ***2 ViTForImageClassification_WithLossMask")
    else:
        raise ValueError("Model name must contain 'resnet' or 'vit'.")

    # 加载模型，处理 Adapter（LoRA）

    logger.info("*** 3 *** Load Model")
    if training_args.finetune:  # 如果是预训练 模型用来finetune，那么不需要添加LoRA 部分
        logger.info(f"*** 3 ***0 - finetune{model_args.model_name_or_path}")
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    else:  # 否则 添加LoRA部分

        if "adapter" in model_args.model_name_or_path:  # 如果加载训练好的 lora代码

            logger.info(
                f"*** 3 ***1 - adapter {model_args.model_name_or_path}")
            model = model_class.from_pretrained(config.base_model_name_or_path)
            model = PeftModel.from_pretrained(
                model, model_args.model_name_or_path)
        elif "resnet" in model_args.model_name_or_path.lower() or "vit" in model_args.model_name_or_path.lower():
            logger.info(
                f"*** 3 ***2 - loading {model_args.model_name_or_path}")
            model = model_class.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=model_args.lora_dim, lora_alpha=32, lora_dropout=0.1
            )
            model = get_peft_model(model, peft_config)

    # 如果模型包含 LoRA，则冻结主干权重

    logger.info("*** Model successfully loaded ***")
    logger.info(f"\n ViT 模型结构如下：\n{model}")
    logger.info(f"✅ 当前 ViT 分类头结构: {model.classifier}")


    # 根据训练数据的类别 替换 ViT 的分类层
    
    # 2️⃣ 替换 ViT 的 classifier 层
    model.classifier = torch.nn.Linear(model.config.hidden_size, num_classes).to(model.device)
    model.config.num_labels = num_classes  # 更新 config 以避免后续问题
    # 3️⃣ 初始化权重 (Xavier 初始化)
    torch.nn.init.xavier_uniform_(model.classifier.weight)
    torch.nn.init.zeros_(model.classifier.bias)

    # 4️⃣ 打印检查
    print(f"✅ ViT classifier 重置完成: {model.classifier}")
    logger.info(f"📐 分类头输出维度: {model.classifier.out_features}, 当前任务类别数: {num_classes}")
    






    # fix lora_A/B (bases of previous LoRA parameters, loaded in "load_adapter"[peft_momdel.py])
    # fine-tune loranew_A/B (initialized in "update_layer"[lora.py])
    # optional: lora_A/B is trainable but should not move too far from lorapre_A/B
    # (constrained in "training_step"[uie_trainer_lora.py])

    # modified
    # 设置使用的不同方法保存的adapter的名字
    # adapter_lora, adapter_Nlora
    # adapter_save_pth =  '/adapter'

    # 尝试输出修改之前模型的所有参数

    if training_args.do_train:

        for name, param in model.named_parameters():
            logger.debug(
                f'model.named_parameters() before traing set name:{name} , param.requires_grad:{param.requires_grad}')

        if training_args.finetune:
            adapter_save_pth = '/finetune'
            # 微调模型， 所有参数都能够进行训练
            for name, param in model.named_parameters():
                if name.find("shared") != -1:
                    param.requires_grad = False

        elif training_args.flag_originLoRA:
            # 如果需要训练模型 并且使用的方法是LoRA 方法
            adapter_save_pth = '/adapter_lora-4'
            # 根据LoRA 方法的设置，这里只有 lora_ 参数，也只更新lora_ 部分的参数
            for name, param in model.named_parameters():
                if name.find("lora_") != -1:
                    param.requires_grad = True
                # this module should always be frozen because we change the vocabulary
                elif name.find("shared") != -1:
                    param.requires_grad = False
        elif training_args.flag_modifiedNLoRA:
            # 如果需要训练模型 并且使用的方法是N_LoRA 方法
            adapter_save_pth = '/adapter_Nlora'
            # 根据N_LoRA 方法的设置，在进行Continual training 时候，只训练更新 task_LoRA，也就是loranew_ 部分
            for name, param in model.named_parameters():
                if name.find("loranew_") != -1:
                    param.requires_grad = True
                elif name.find("lora_") != -1:
                    param.requires_grad = False
                # this module should always be frozen because we change the vocabulary
                elif name.find("shared") != -1:
                    param.requires_grad = False
        else:
            adapter_save_pth = '/adapter'

    

    """ ----------------- 尝试训练模型-----------------"""
    #  ✅ 1. 加载评估指标（accuracy, F1-score, precision, recall）
    metric_accuracy = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")
    metric_precision = evaluate.load("precision")
    metric_recall = evaluate.load("recall")

    def compute_metrics(eval_pred):
        """
        计算模型的评估指标（Accuracy, F1-score, Precision, Recall）
        用于 Hugging Face Trainer 进行 ViT 图像分类评估
        """
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=1)  # 获取类别索引

        # ✅ 计算 Accuracy
        accuracy = metric_accuracy.compute(predictions=predictions, references=labels)

        # ✅ 计算 F1-score, Precision, Recall
        f1 = metric_f1.compute(predictions=predictions, references=labels, average="weighted")
        precision = metric_precision.compute(predictions=predictions, references=labels, average="weighted")
        recall = metric_recall.compute(predictions=predictions, references=labels, average="weighted")

        # ✅ 组合所有评估指标
        results = {
            "accuracy": round(accuracy["accuracy"], 4),
            "f1_score": round(f1["f1"], 4),
            "precision": round(precision["precision"], 4),
            "recall": round(recall["recall"], 4),
        }

        return results

    # 这里的训练模型并没有继承 Trasnsformer 进行方法的重载，为了方法的简单方便就直接使用之前论文提供的初始代码

    trainable_params = [
        n for n, p in model.named_parameters() if p.requires_grad]
    logger.info(
        f'***4***-ContinualTrainer(model)-- args= trainable_params:{trainable_params}')
    


    # 重新建立函数
    trainer = SAMContinualTrainer(
        model=model,
        dataloaders=dataloaders,
        task_split=task_split,
        args=training_args,
        num_classes=num_classes,
        classes_per_task=classes_per_task,
        criterion=None,  # 可选，自定义损失函数
        algo=None,       # 可选，提前传入 algo
        sam=training_args.sam,
        subset_dataloaders=subset_dataloaders
    )


    # 使用不同的方法作为 加速器
    logger.debug(f'begin traing training_args.do_train:{training_args.do_train},training_args.finetune:{training_args.finetune}')
            

    if training_args.do_train:
        if training_args.finetune:
            if training_args.task_id != -1 : # 只加载某一个任务

                logger.debug(f'train_onlyonetask, training_args.task_id:{training_args.task_id}')
    
                trainer.train_onlyonetask(training_args.task_id)    
            else: 
                trainer.train_cltask()
    
            
    if training_args.do_flatminal:
        # trainer.compute_loss_landscape_version2(eval_task_id=0,output_dir=training_args.output_dir)
        trainer.compute_loss_landscape(eval_task_id=0,output_dir=training_args.output_dir)


if __name__ == "__main__":
    main()
