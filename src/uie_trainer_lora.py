import torch
from transformers import GenerationConfig
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.trainer import *
from transformers.trainer_callback import TrainerCallback

from uie_collator import SUPPORTED_DECODER_MODELS, check_model
from uie_dataset_lora import ANSWER_PREFIX

from datetime import datetime
from torch.nn import functional as F
import copy
import h5py

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from sam import SAM, enable_running_stats,disable_running_stats
from existing_methods.agem import AGEM
from existing_methods.er import ER
from existing_methods.ewc import EWC


def skip_instructions(model, predictions_ids, tokenizer, ignore_idx=-100):
    predictions_ids = np.where(
        predictions_ids == ignore_idx, tokenizer.pad_token_id, predictions_ids)

    predictions = tokenizer.batch_decode(
        predictions_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    final_predictions = []
    if check_model(model.config._name_or_path, SUPPORTED_DECODER_MODELS):
        for pred in predictions:

            if ANSWER_PREFIX in pred:
                splits = pred.split(ANSWER_PREFIX)
                final_predictions.append(splits[-1].strip())
            else:
                final_predictions.append('')
    else:
        final_predictions = predictions

    return final_predictions


class DenserEvalCallback(TrainerCallback):

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):

        log_eval_steps = [1, 50, 100, 200]

        # Log
        if args.logging_strategy == IntervalStrategy.STEPS and state.global_step in log_eval_steps:
            control.should_log = True

        # Evaluate
        if args.evaluation_strategy == IntervalStrategy.STEPS and state.global_step in log_eval_steps:
            control.should_evaluate = True

        # Save
        # if args.save_strategy

        return control


class UIETrainer(Seq2SeqTrainer):
    def __init__(
        self,
        data_collator,
        model: Union["PreTrainedModel", nn.Module] = None,
        args: "TrainingArguments" = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        model_init: Optional[Callable[[], "PreTrainedModel"]] = None,
        compute_metrics: Optional[Callable[["EvalPrediction"], Dict]] = None,
        callbacks: Optional[List["TrainerCallback"]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        model_2: Optional[Union["PreTrainedModel", nn.Module]] = None,  # 添加第二个模型
        model_3: Optional[Union["PreTrainedModel", nn.Module]] = None,  # 添加第三个模型
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # 额外保存两个模型用于计算损失景观
        self.model_2 = model_2
        self.model_3 = model_3

    def create_optimizer_and_scheduler(self, num_training_steps: int):
            """
            自定义优化器和学习率调度器
            """
            # 初始化优化器
            if self.args.sam:
                self.optimizer = SAM(self.model.parameters(), torch.optim.SGD, rho=self.args.rho, lr=self.args.learning_rate)
            else:
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate, momentum=self.args.momentum)

            # 学习率调度器
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=num_training_steps // 3, gamma=0.1)

            logger.info(f"✅ 优化器: {self.optimizer}")
            logger.info(f"✅ 学习率调度器: {self.lr_scheduler}")


    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        # SageMaker MP 特殊处理
        if is_sagemaker_mp_enabled():
            # DeepSpeed 在 backward 里自动处理梯度缩放与累加
            loss_mb = smp_forward_backward(
                model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        l1_loss = 0.
        loranew_A_params = {}
        loranew_B_params = {}
        for name, param in self.model.named_parameters():
            if "loranew_A" in name:
                loranew_A_params[name.split("loranew_A")[0]] = param
            elif "loranew_B" in name:
                loranew_B_params[name.split("loranew_B")[0]] = param

        for key in loranew_A_params:
            if key in loranew_B_params:
                l1_loss += torch.norm(
                    torch.mm(loranew_A_params[key], loranew_B_params[key]), p=1)

        lamda_1 = self.args.lamda_1
        loss = loss + l1_loss * lamda_1

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()


        # ===================== 🔧 自定义优化器逻辑开始 =====================

        if hasattr(self, "optimizer") and self.optimizer is not None:
            if self.args.sam:
                # SAM 优化器的两步更新
                self.optimizer.first_step(zero_grad=True)
                with self.compute_loss_context_manager():
                    second_loss = self.compute_loss(model, inputs)
                    second_loss = second_loss + l1_loss * lamda_1
                second_loss.backward()
                self.optimizer.second_step(zero_grad=True)
            else:
                self.optimizer.step()
                self.optimizer.zero_grad()

        # ===================== 🔧 自定义优化器逻辑结束 =====================

        return loss.detach()

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:

            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None,  # inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = dataloader.batch_size

        logger.debug(f"***** Running {description} *****")
        if has_length(dataloader.dataset):
            logger.debug(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.debug("  Num examples: Unknown")
        logger.debug(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat(
                    (losses_host, losses), dim=0)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(
                    labels_host, labels, padding_index=-100)
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                preds_host = logits if preds_host is None else nested_concat(
                    preds_host, logits, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(
                args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate(
                        (all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(
                        all_preds, logits, padding_index=-100)
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(
                            all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate(
                (all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(
                all_preds, logits, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(
                all_labels, labels, padding_index=-100)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(
                dataset=eval_dataset, preds=all_preds, save_prefix=metric_key_prefix)
        else:
            metrics = {}

        metrics["global_step"] = self.state.global_step

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = self._gen_kwargs
        gen_kwargs["synced_gpus"] = True if is_deepspeed_zero3_enabled() else False

        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)

        generation_config = GenerationConfig(**gen_kwargs)

        # prepare generation inputs
        # some encoder-decoder models can have varying encder's and thus
        # varying model input names
        if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]

        generated_tokens = self.model.generate(
            input_ids=generation_inputs,
            generation_config=generation_config
        )

        bs, source_len = inputs['input_ids'].shape
        # in case the batch is shorter than max length, the output should be padded
        if check_model(self.model.config._name_or_path, SUPPORTED_DECODER_MODELS):
            max_length = source_len + gen_kwargs["max_new_tokens"]
        else:
            max_length = gen_kwargs["max_new_tokens"]

        if generated_tokens.shape[-1] < max_length:
            generated_tokens = self._pad_tensors_to_max_len(
                generated_tokens, max_length)

        with torch.no_grad():
            if has_labels:
                with self.autocast_smart_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = self.label_smoother(
                        outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(
                        outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_kwargs["max_new_tokens"]:
                labels = self._pad_tensors_to_max_len(
                    labels, gen_kwargs["max_new_tokens"])
        else:
            labels = None

        return (loss, generated_tokens, labels)

    def _compute_stats(self, eigenvalues):
        """改进点13：统一统计计算"""
        if not eigenvalues:
            return {"min": 0., "max": 0., "median": 0., "mean": 0.}

        arr = np.array(eigenvalues)
        return {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "median": float(np.median(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr))
        }

    def compute_loss_landscape(
        self,   flatminal_dataset: Dataset, output_dir, save_file_name="lossLandscape", x_range=(-1, 1), y_range=(-1, 1), num_points=20, max_batches=5,
        sample_batches=False, flag_lora=True, flag_Nlora_full=False, flag_FullModel=False,
    ):
        """
        计算损失景观，并分别计算:
        1️⃣ **完整模型（Base Model + LoRA Adapter）** 的损失景观
        2️⃣ **仅 LoRA 适配器（LoRA Adapter）** 的损失景观
        计算损失景观，并分别计算完整模型 & LoRA Adapter 的损失表面。
        针对大模型使用多 GPU，可以并行分配 (i, j) 坐标网格。
        flag_FullModel : 是否对模型的所有参数都进行扰动
        flag_lora :  是否对lora_进行扰动
        flag_Nlora_full： 是否对 lora_new 和 lora_ 都进行扰动
        """
        args = self.args
        device = args.device
        # 最终损失网络的数值
        loss_grid = np.zeros((num_points, num_points))

        # ✅ 记录日志信息
        logger.debug(f'***5***--5-2 **1 compute_loss_landscape  ')
        # logger.debug(f"***** Running Loss Landscape Calculation on expriment ,Nlora_full:{flag_Nlora_full}  lora:{flag_lora}*****")

        # ✅ 兼容 AMP 和分布式训练
        # 复制模型避免污染
        model = copy.deepcopy(self.model)
        model = self._wrap_model(model, training=False)

        # ✅ 处理 FP16/BF16 评估模式
        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        model = model.to(device=device)
        model.eval()  # 确保模型在 eval 模式

        # --------------------- 生成参数扰动阶段阶段 ---------------------
        # 确定需要扰动的参数名称
        # 根据不同的微调方法（Nlora或lora）确定需要保存原始值的参数

        new_Flag = '3.3'
        original_params_to_perturb = {}
        for name, param in model.named_parameters():

            # 注意 默认加载的模型已经添加了LoRA部分，如果是1 需要设置lora 中的代码 去掉 add_lora的代码

            if new_Flag == '1':  # 只更改 W
                if "shared" not in name:
                    param.requires_grad = True
                    original_params_to_perturb[name] = param.data.clone()

                surf_file = os.path.join(output_dir, f"{save_file_name}_1.h5")

            # 注意，2.1 加载的是初始模型，然后加载过后添加了LoRA部分，进行了初始化
            elif new_Flag == '2.1':  # 只更改 W
                if "lora_" in name and "shared" not in name:
                    param.requires_grad = False
                elif "lora_" not in name and "shared" not in name:
                    param.requires_grad = True
                    original_params_to_perturb[name] = param.data.clone()

                surf_file = os.path.join(
                    output_dir, f"{save_file_name}_2-1.h5")
            elif new_Flag == '2.2':  # 只更改 AB
                if "lora_" in name and "shared" not in name:
                    original_params_to_perturb[name] = param.data.clone()
                surf_file = os.path.join(
                    output_dir, f"{save_file_name}_2-2.h5")
            elif new_Flag == '2.3':  # 只更改 W AB
                if "shared" not in name:
                    original_params_to_perturb[name] = param.data.clone()
                surf_file = os.path.join(
                    output_dir, f"{save_file_name}_2-3.h5")

            # 注意，3.1 加载的是训练保存得到的adapter模型，里面具有训练后的LoRA部分
            elif new_Flag == '3.1':  # 只更改 W

                if "lora_" in name and "shared" not in name:
                    param.requires_grad = False
                elif "lora_" not in name and "shared" not in name:
                    param.requires_grad = True
                    original_params_to_perturb[name] = param.data.clone()

                surf_file = os.path.join(
                    output_dir, f"{save_file_name}_3-1_t1.h5")
            elif new_Flag == '3.2':  # 只更改 AB
                if "lora_" in name and "shared" not in name:
                    original_params_to_perturb[name] = param.data.clone()
                surf_file = os.path.join(
                    output_dir, f"{save_file_name}_3-2.h5")
            elif new_Flag == '3.3':  # 只更改 W AB
                if "shared" not in name:
                    original_params_to_perturb[name] = param.data.clone()
                surf_file = os.path.join(
                    output_dir, f"{save_file_name}_3-3.h5")
            
            

            # if flag_FullModel: # 如果是对所有参数都进行干扰，

            #     if "shared" not in name:
            #         original_params_to_perturb[name] = param.data.clone()

            #     surf_file = os.path.join(output_dir, f"{save_file_name}_fullModel-predictDataset.h5")
            # elif flag_lora:  # 当不使用全部模型的参数，并且使用标准lora方法时
            #     # 保存所有lora参数（lora_开头），且排除共享参数
            #     if "lora_" in name and "shared" not in name:
            #         original_params_to_perturb[name] = param.data.clone()
            #     surf_file = os.path.join(output_dir, f"{save_file_name}_lora_only-predictDataset.h5")

            # elif flag_Nlora_full: # 当不使用全部模型的参数，不使用标准lora方法，使用Nlora方法并且干扰所有相关 lora_,loranew_模型时候
            #     if ("loranew_" in name or "lora_" in name) and "shared" not in name:
            #         original_params_to_perturb[name] = param.data.clone()
            #     surf_file = os.path.join(output_dir, f"{save_file_name}_Nlora_full-predictDataset.h5")

            # else:
            #     if "loranew_" in name and "shared" not in name:
            #         original_params_to_perturb[name] = param.data.clone()
            #     surf_file = os.path.join(output_dir, f"{save_file_name}_Nlora_onlytask-predictDataset.h5")

        # 添加日志，打印所有被加入扰动的参数
        logger.debug(f"*** Perturbed Parameters ***")
        for name in original_params_to_perturb.keys():
            logger.debug(
                f"name: {name}, requires_grad: {model.state_dict()[name].requires_grad}")

        logger.debug(
            f"Output Dir = {output_dir}，Output File :{surf_file} Num points = {num_points}x{num_points} max_batches = {max_batches}")

        # --------------------- 扰动生成优化 ---------------------
        torch.manual_seed(42)  # 固定随机种子保证可重复性
        perturb_x, perturb_y = {}, {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name not in original_params_to_perturb:
                    continue

                # 直接生成归一化扰动（优化点2）
                # norm_factor = torch.norm(param) + 1e-8  # 计算神经网络参数的尺度
                # torch.manual_seed(seed_x)
                # d_x_perturb = torch.randn_like(param).to(device)  # 生成随机扰动
                # perturb_x[name] = (d_x_perturb / torch.norm(d_x_perturb)) * norm_factor  # 归一化并调整尺度
                else:
                    d_x = torch.randn_like(param)
                    d_x = (d_x / d_x.norm()) * (param.norm() + 1e-8)  # 直接归一化
                    perturb_x[name] = d_x.to(device)

                    # 生成d_x的正交方向的扰动
                    d_y = torch.randn_like(param)
                    d_y = d_y - torch.sum(d_y * d_x) * \
                        d_x / (d_x.norm()**2)  # 施密特正交化
                    d_y = (d_y / d_y.norm()) * (param.norm() + 1e-8)     # 归一化
                    perturb_y[name] = d_y.to(device)

        # 使用FP16存储扰动（优化点5）
        if args.fp16_full_eval or args.bf16_full_eval:
            perturb_x = {k: v.half() for k, v in perturb_x.items()}
            perturb_y = {k: v.half() for k, v in perturb_y.items()}

        # --------------------- 并行化网格计算（优化点3）---------------------
        x_coords = np.linspace(x_range[0], x_range[1], num_points)
        y_coords = np.linspace(y_range[0], y_range[1], num_points)

        # 分布式任务划分
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()  # 获取总进程数（GPU数量）
            rank = torch.distributed.get_rank()  # 获取当前进程的编号（从0开始）
            # 按行划分任务
            chunk = num_points // world_size
            start_idx = rank * chunk
            end_idx = (rank + 1) * chunk if rank != world_size - \
                1 else num_points
            logger.debug(
                f'***5***--5-2**4 distribute--yes world_size:{world_size} rank:{rank},start_idx:{start_idx},end_idx:{end_idx} ')
        else:
            # logger.debug(f'****5***--5-2**2-1 if**5***--5-2**5 distribute--no ')
            world_size = 1
            rank = 0
            start_idx, end_idx = 0, num_points
            logger.debug(
                f'***5***--5-2**4 distribute--no world_size:{world_size} rank:{rank},start_idx:{start_idx},end_idx:{end_idx} ')

        # --------------------- 数据准备阶段 ---------------------
        logger.debug(
            f'***5***--5-2**2 begin load data---{torch.distributed.is_initialized()} ')
        dataloader = self.get_eval_dataloader(flatminal_dataset)

        all_batches = []
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            all_batches.append(batch)

        logger.debug(f'***5***--5-2**2 begin load data---{len(all_batches)} ')

        # 合并所有批次为单个大批次（显存允许时）
        if not sample_batches and len(all_batches) > 0:
            try:
                # 尝试合并所有小批次为一个大批次
                big_batch = {
                    # 对每个特征键（如input_ids、labels）进行纵向拼接
                    k: torch.cat([b[k] for b in all_batches], dim=0)
                    for k in all_batches[0].keys()  # 假设所有批次结构相同
                }
                # 用合并后的大批次替换原始批次列表
                all_batches = [big_batch]  # 现在只包含一个合并后的批次
            except RuntimeError:
                # 显存不足时回退到原始小批次
                logger.warning("无法合并批次，保持原有批次数量")

# --------------------- 主计算循环优化 ---------------------

        for i in tqdm(range(start_idx, end_idx), desc=f"Rank {rank} Processing"):
            xv = x_coords[i]
            for j, yv in enumerate(y_coords):
                # 恢复参数时仅操作需要修改的部分（优化点1）
                for name in original_params_to_perturb:
                    model.state_dict()[name].copy_(
                        original_params_to_perturb[name])

                # 应用扰动
                with torch.no_grad():
                    for name in original_params_to_perturb:
                        param = model.state_dict()[name]
                        delta = xv * \
                            perturb_x[name].to(
                                param.dtype) + yv * perturb_y[name].to(param.dtype)
                        param.add_(delta)  # 原位操作减少内存分配

                # 计算损失
                total_loss = 0.0
                for batch in all_batches:
                    inputs = {k: v.to(device) for k, v in batch.items()}
                    # 支持混合精度
                    # with torch.cuda.amp.autocast(enabled=args.fp16_full_eval):
                    #     outputs = model(**inputs)
                    #     loss = F.cross_entropy(
                    #         outputs.logits.view(-1, outputs.logits.size(-1)),
                    #         inputs["labels"].view(-1)
                    #     )
                    with self.compute_loss_context_manager():
                        loss = self.compute_loss(model, inputs)
                    total_loss += loss.item()

                loss_grid[i, j] = total_loss / max(1, len(all_batches))  # 避免除零

                # 每10次迭代清理一次缓存（优化点6）
                if j % 10 == 0:
                    torch.cuda.empty_cache()

        # --------------------- 分布式结果收集 ---------------------
        if torch.distributed.is_initialized():
            # 收集所有进程的loss_grid

            # 计算 loss_grid
            # 将 loss_grid 转换为张量
            loss_grid_tensor = torch.tensor(loss_grid, device=device)
            # 创建一个列表，用于接收所有进程的 loss_grid
            all_loss = [torch.zeros_like(loss_grid_tensor)
                        for _ in range(world_size)]
            # 收集所有进程的 loss_grid
            torch.distributed.all_gather(all_loss, loss_grid_tensor)
            # 将收集到的结果拼接成一个完整的 loss_grid
            loss_grid_tensor = torch.stack(all_loss, dim=0).mean(dim=0)  # 取平均
            loss_grid = loss_grid_tensor.cpu().numpy()  # 转回 NumPy 数组后再保存
        else:
            all_loss = [loss_grid]

        # 仅rank 0进程保存结果
        if rank == 0:
            with h5py.File(surf_file, 'w') as f:
                f.create_dataset('xcoordinates', data=x_coords)
                f.create_dataset('ycoordinates', data=y_coords)
                f.create_dataset('train_loss', data=loss_grid)

            logger.debug(f"计算完成，结果保存至{surf_file}")

        return True

    def compute_loss_landscape_version2(
        self, flatminal_dataset, output_dir, save_file_name="lossLandscape",
        num_samples=100, max_batches=5, sharpness_p=100, sharpness_alpha=1.0,
        sample_batches=False, flag_lora=True, flag_Nlora_full=False, flag_FullModel=False
    ):
        """
        计算损失景观，并使用 Sharpness Metric 评估曲率（flatness）。

        关键改进：
        - 只在 **整个参数空间** 进行一次全局扰动，而不是每一层独立扰动。
        - 使用 Keskar et al. (2017) 的 **随机投影方法** 计算 Sharpness Metric。
        """

        args = self.args
        device = args.device

        # 记录日志信息
        logger.debug(f"*** Compute Loss Landscape with Sharpness Metric ***")

        # 复制模型，避免污染原模型
        model = copy.deepcopy(self.model)
        model = self._wrap_model(model, training=False)
        model.eval()
        model.to(device)

        # --------------------- 选择需要扰动的参数 ---------------------
        
        param_vector = []
        param_shapes = []
        param_names = []

        for name, param in model.named_parameters():
            if  "shared" not in name:
                param_vector.append(param.data.view(-1))
                param_shapes.append(param.shape)
                param_names.append(name)
        param_vector = torch.cat(param_vector)  # 合并为一个整体参数向量
        num_parameters = param_vector.numel()

        logger.debug(f"*** Total parameters selected: {num_parameters} ***")

       

        # --------------------- 计算损失景观 ---------------------
        # --------------------- 计算基准损失 L(w) ---------------------
        def compute_loss(model):
            """计算模型在完整数据集上的 loss"""
            total_loss = 0.0
            for batch in self.get_eval_dataloader(flatminal_dataset):
                inputs = {k: v.to(device) for k, v in batch.items()}
                with self.compute_loss_context_manager():
                    loss = self.compute_loss(model, inputs)
                total_loss += loss.item()
            return total_loss / max(1, len(flatminal_dataset))

        L_w = compute_loss(model)  # 计算原始模型损失
        logger.debug(f"Baseline loss L(w): {L_w:.6f}")

        # --------------------- 计算最大扰动损失 ---------------------

         # --------------------- 生成扰动矩阵 A ---------------------
        # torch.manual_seed(42)  # 固定随机种子
        A = torch.randn(num_parameters, sharpness_p, device=device)  # 生成投影矩阵
        A /= torch.norm(A, dim=0, keepdim=True)  # 归一化
        logger.debug(f"Generated projection matrix A with shape {A.shape}")
        
        loss_list = []  # 存储所有扰动后的损失值
        L_max = L_w  # 初始化 L_max（避免未定义错误）

        for _ in tqdm(range(num_samples), desc="Processing Samples"):


            

            # 生成随机扰动向量 Z
            z = torch.randn(sharpness_p, device=device)
            z = sharpness_alpha * z / (torch.norm(z) + 1e-8)  # 归一化

            # 计算 Az
            perturbed_w = A @ z  # 计算扰动后的参数向量
            perturbed_w = perturbed_w.view(-1)  # 保持一维

            # 添加扰动
            idx = 0
            for i, name in enumerate(param_names):
                param = model.state_dict()[name]
                numel = param_shapes[i].numel()
                delta = perturbed_w[idx: idx + numel].reshape(param_shapes[i])
                param.add_(delta)
                idx += numel

            # 计算扰动后的损失
            perturbed_loss = compute_loss(model)
            logger.debug(f'perturbed_loss:{perturbed_loss}')

            loss_list.append(perturbed_loss)  # 记录当前扰动的损失
            L_max = max(L_max, perturbed_loss)  # 记录最大扰动损失值

            # 释放 GPU 内存
            torch.cuda.empty_cache()

        # --------------------- 计算 Sharpness Metric ---------------------
        sharpness_metric = ((L_max - L_w) / (1 + L_w)) * 100  # 计算 Sharpness

        logger.debug(f"*** Sharpness Metric: {sharpness_metric:.4f} ***")

        # --------------------- 保存结果 ---------------------
        surf_file = os.path.join(output_dir, f"{save_file_name}_sharpness.h5")
        with h5py.File(surf_file, 'w') as f:
            f.create_dataset('L_w', data=L_w)  # 原始损失
            f.create_dataset('L_max', data=L_max)  # 最大扰动损失
            f.create_dataset('sharpness_metric', data=sharpness_metric)  # Sharpness 指标
            f.create_dataset('loss_list', data=np.array(loss_list))  # 保存所有扰动损失

        logger.debug(f"Loss Landscape and Sharpness saved at {surf_file}")

        return sharpness_metric



    def compute_loss_contours(
        self, eval_dataset, output_dir="output", save_file_name="lossLandscape",
        granularity=20, margin=0.2
    ):
        """
        计算局部损失景观，并保存到 HDF5 文件。

        - 计算 `LoRA` 适配器的增量参数 `ΔW2`, `ΔW3`。
        - 计算 `u = ΔW2`, `v = ΔW3 - 投影到 u 的部分`。
        - 在 `u, v` 方向生成扰动后的模型权重，并计算损失值。
        """

        if self.model_2 is None or self.model_3 is None:
            raise ValueError("必须提供 model_2 和 model_3 以计算损失景观。")

        device = self.args.device

        # --------------------- 1. 提取基础模型 W1（不包含 LoRA 部分） ---------------------
        base_param_vector = []
        param_shapes = []
        param_names = []

        for name, param in self.model.named_parameters():
            if "lora_" not in name:  # 只选择 Base Model 部分的参数
                base_param_vector.append(param.data.view(-1))
                param_shapes.append(param.shape)
                param_names.append(name)

        base_param_vector = torch.cat(base_param_vector)  # 形成完整参数向量
        num_parameters = base_param_vector.numel()
        logger.debug(f"Total parameters selected: {num_parameters}")

        # --------------------- 2. 提取 LoRA 参数增量 ΔW2, ΔW3 ---------------------
        def extract_lora_delta(model, base_model):
            """提取 LoRA 适配器的增量参数"""
            delta_vector = []
            for name, param in model.named_parameters():
                if "lora_" in name:  # 只提取 LoRA 适配器部分
                    delta_vector.append(param.data.view(-1))
            return torch.cat(delta_vector).to(device)

        ΔW2 = extract_lora_delta(self.model_2, self.model)  # W2 的 LoRA 适配器增量
        ΔW3 = extract_lora_delta(self.model_3, self.model)  # W3 的 LoRA 适配器增量

        # --------------------- 3. 计算正交方向 u, v ---------------------
        u = ΔW2 / torch.norm(ΔW2)  # 归一化方向向量 u
        v = ΔW3 - torch.dot(u, ΔW3) * u  # 使 v 正交于 u
        v /= torch.norm(v)  # 归一化 v

        # --------------------- 4. 生成网格点 ---------------------
        alphas = np.linspace(-margin, 1 + margin, granularity)
        betas = np.linspace(-margin, 1 + margin, granularity)
        loss_map = np.zeros((granularity, granularity))

        x_coords = np.zeros((granularity, granularity))
        y_coords = np.zeros((granularity, granularity))

        dataloader = get_eval_dataloader(eval_dataset)
        progress = tqdm(total=granularity * granularity)

        # --------------------- 5. 遍历网格计算损失 ---------------------
        for i, alpha in enumerate(alphas):
            for j, beta in enumerate(betas):
                # 计算扰动后的 LoRA 适配器参数
                perturbed_w = ΔW2 * alpha + ΔW3 * beta

                # 施加扰动
                idx = 0
                for k, name in enumerate(param_names):
                    param = self.model.state_dict()[name]
                    numel = param_shapes[k].numel()
                    delta = perturbed_w[idx: idx + numel].reshape(param_shapes[k])
                    param.add_(delta)  # 直接添加扰动
                    idx += numel

                # 计算扰动后的损失
                perturbed_loss = self.compute_loss_fn(self.model, dataloader, device)
                logger.debug(f'perturbed_loss:{perturbed_loss}')

                # 存储结果
                x_coords[i, j] = alpha
                y_coords[i, j] = beta
                loss_map[i, j] = perturbed_loss

                progress.update()
        progress.close()

        # --------------------- 6. 保存数据 ---------------------
        os.makedirs(output_dir, exist_ok=True)
        surf_file = os.path.join(output_dir, f"{save_file_name}.h5")

        with h5py.File(surf_file, 'w') as f:
            f.create_dataset('xcoordinates', data=x_coords)
            f.create_dataset('ycoordinates', data=y_coords)
            f.create_dataset('train_loss', data=loss_map)

        logger.info(f"Loss landscape data saved at {surf_file}")




    def compute_hessian_version1(
            self,
            flatminal_dataset,
            output_dir,
            name="hessian",
            max_batches=10,
            sample_batches=False,
            use_gpu=True,
            flag_lora=True, flag_Nlora_full=True):
        """
            改进版Hessian矩阵计算函数，主要优化：
            1. 完全消除参数污染风险
            2. 支持分布式训练环境
            3. 增强数值稳定性
            4. 内存效率优化
            5. 增加特征向量分析

            参数说明：
            - max_batches: 最大计算batch数（用于大数据集采样）
            - sample_batches: 是否随机采样batch（True=随机，False=顺序取前N个）

        """
        # --------------------- 初始化阶段 ---------------------
        # 根据模型判断，这里不妨使用手动

        logger.debug(f'***5***--5-3**1 begin init   ')
        logger.debug(
            f'***5***--5-3**1 use distribute ---- {torch.distributed.is_initialized()}')

        args = self.args
        device = args.device

        # 创建独立模型副本（关键改进点1：隔离原始模型）
        model = copy.deepcopy(self.model)
        model = self._wrap_model(model, training=False)

        # 混合精度处理
        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device

        # 设置要评估的参数
        # 设置需要进行评估的参数
        # if training_args.flag_disturb_fullModel:
        #     # 如果是要对模型的所有参数都进行评估
        #     for name, param in model.named_parameters():
        #         param.requires_grad = True

        # # 不改变整个模型的参数，只改变添加的lora部分的参数
        # elif training_args.flag_originLoRA:

        # # 如果需要评估模型 并且模型是LoRA方法训练得到的
        # # 则需要将 模型的LoRA设置为可以更新的，以供计算Hessian矩阵使用
        #     for name, param in model.named_parameters():
        #         if name.find("lora_") != -1:
        #             param.requires_grad = True
        #         # this module should always be frozen because we change the vocabulary
        #         elif name.find("shared") != -1:
        #             param.requires_grad = False
        # elif training_args.flag_modifiedNLoRA:
        # # 如果需要评估模型 并且模型是N_LoRA方法训练得到的
        #     if training_args.flag_modifiedNLoRA_fullLoRA:
        #         # 如果评估N_LoRA方法的所有LoRA （lora_和 loranew_）对模型的影响
        #         # 则将这两部分全部设置为可以更新的，以供计算Hessian矩阵使用
        #         for name, param in model.named_parameters():
        #             if name.find("loranew_") != -1:
        #                 param.requires_grad = True
        #             elif name.find("lora_") != -1:
        #                 param.requires_grad = True
        #             # this module should always be frozen because we change the vocabulary
        #             elif name.find("shared") != -1:
        #                 param.requires_grad = False
        #     elif training_args.flag_modifiedNLoRA_taskLoRA:
        #         # 如果只评估N_LoRA方法的  taskLoRA部分 （loranew_）对模型的影响
        #         # 则将这部分设置为可以更新的，以供计算Hessian矩阵使用
        #         for name, param in model.named_parameters():
        #             if name.find("loranew_") != -1:
        #                 param.requires_grad = True
        #             elif name.find("lora_") != -1:
        #                 param.requires_grad = False
        #             # this module should always be frozen because we change the vocabulary
        #             elif name.find("shared") != -1:
        #                 param.requires_grad = False

        # logger.debug(f'***5***-After set param   ')

        # # Debug 查看设置完后的对哪些参数进行扰动
        # for name, param in model.named_parameters():
        #     logger.debug(f'name:{name}, requires_grad:{param.requires_grad}')

        # 输出 batch 和参数的类型
        logger.debug(
            f'5***--5-3**1 Model device: {next(model.parameters()).device}, '
            f'Model param type: {next(model.parameters()).dtype}, '
        )

        if not self.is_in_train:
            logger.debug(
                f'args.fp16_full_eval:{args.fp16_full_eval}, args.bf16_full_eval:{args.bf16_full_eval}')
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        model = model.to(device=device)

        model.eval()

        # 输出 batch 和参数的类型
        logger.debug(
            f'5***--5-3**1 Model device: {next(model.parameters()).device}, '
            f'Model param type: {next(model.parameters()).dtype}, '
        )

        logger.debug(f'***5***--LORA Hessian**1 finish init ')

        # --------------------- 数据准备阶段 ---------------------
        logger.debug(f'***5***--5-3**2 begin load data   ')
        dataloader = self.get_eval_dataloader(flatminal_dataset)
        all_batches = list(dataloader)

        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            all_batches = all_batches[rank::world_size]  # 数据分片
            logger.debug(
                f'***5***--LORA Hessian**2 distribute yes, rank:{rank}, total_batches:{len(all_batches)}')

        else:
            logger.debug(f'***5***--5-3**4 distribute--no ')
            world_size = 1
            rank = 0

        # 数据采样处理
        logger.debug(
            f'***5***--5-3**2-1 if  {len(all_batches)} , {max_batches} ')
        if len(all_batches) > max_batches:
            if sample_batches:
                indices = np.random.choice(
                    len(all_batches), max_batches, replace=False)
                all_batches = [all_batches[i] for i in indices]
            else:
                all_batches = all_batches[:max_batches]

        # 分布式通信初始化（改进点9：分布式支持）
        if torch.distributed.is_initialized():
            logger.debug(f'***5***--5-3**4 distribute--yes ')
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            all_batches = all_batches[rank::world_size]  # 数据分片
            logger.debug(f'***5***--5-3**4 {len(all_batches)} ')
        else:
            logger.debug(f'***5***--5-3**4 distribute--no ')
            world_size = 1
            rank = 0

        logger.debug(f'***5***--5-3**2 finish load data ')

        # --------------------- 核心算法定义 ---------------------
        class HessianCalculator:
            """
            Hessian 计算器：
            - 计算 Hessian-Vector Product (HVP)
            - 使用 Lanczos 方法估计 Hessian 的特征值

            参数：
            - model: 计算 Hessian 的神经网络模型
            - device: 计算设备 (CPU/GPU)
            """

            def __init__(self, model, device, max_dim=10000):
                logger.debug(
                    f'***5***--5-3**3 class HessianCalculator init   ')
                self.model = model
                self.device = device
                self.criterion = torch.nn.CrossEntropyLoss()
                # self.max_dim = max_dim  # 限制 Hessian 计算的最大维度

            @staticmethod
            def _safe_normalize(v, eps=1e-12):
                """改进点4：安全归一化防止除零错误"""
                norm = torch.norm(v) + eps
                return v / norm

            def compute_hvp(self, batch, param_list=None):
                """计算Hessian-vector乘积函数 (HVP) """
                logger.debug(f'***5***--5-3**5 begin compute_hvp() ')
                self.model.zero_grad()  # 清空梯度
                batch = {k: v.to(self.device) for k, v in batch.items()}

                with torch.autograd.set_grad_enabled(True):

                    outputs = self.model(**batch)
                    loss = self.criterion(
                        outputs.logits.view(-1, outputs.logits.size(-1)),
                        batch["labels"].view(-1)
                    )

                # 根据传入参数的类型适配：字典或列表/元组
                # 选择需要计算 Hessian 的参数
                if isinstance(param_list, dict):
                    params = [p for p in param_list.values()
                              if p.requires_grad]
                elif isinstance(param_list, (list, tuple)):
                    params = [p for p in param_list if p.requires_grad]
                else:
                    params = []  # 若传入为 None 或其它类型，则空处理

                # 计算一阶梯度
                grads = torch.autograd.grad(loss, params, create_graph=True)

                def hvp_func(v):
                    """计算 Hessian 向量积,闭包函数保持计算图"""

                    split_sizes = [p.numel()
                                   for p in params]  # 计算每个 param 对应的大小
                    v_split = torch.split(v, split_sizes)  # 按照每个参数的形状拆分 v
                    v_reshaped = [v_i.view(p.shape) for v_i, p in zip(
                        v_split, params)]  # 重新调整 v_i 形状

                    # logger.debug(f"grads shape: {[g.shape for g in grads]}")
                    # logger.debug(f"v shape: {v.shape}")
                    # logger.debug(f"Total params count: {len(params)}, Total elements in params: {sum(split_sizes)}")
                    # logger.debug(f"🔹 First 5 split sizes: {split_sizes[:5]}")
                    # logger.debug(f"🔹 First 5 param shapes: {[p.shape for p in params[:5]]}")
                    # logger.debug(f"🔹 First 5 v_split shapes: {[v_i.shape for v_i in v_split[:5]]}")

                    # 计算 Hessian 作用
                    Hv = torch.autograd.grad(
                        grads, params, grad_outputs=v_reshaped,
                        retain_graph=True, allow_unused=True
                    )

                    # 拼接计算结果
                    Hv_flattened = torch.cat([
                        hv.contiguous().flatten() if hv is not None else torch.zeros_like(p).flatten()
                        for hv, p in zip(Hv, params)
                    ]).to(self.device)

                    # logger.debug(f"🔹 Hv computed successfully, shape: {Hv_flattened.shape}")
                    torch.cuda.empty_cache()  # 释放显存
                    return Hv_flattened

                return hvp_func

            def block_lanczos(self, hvp_func, dim, k=10, block_size=4):
                """
                分块Lanczos算法
                其中超参数 k 是迭代次数，block_size 是块的大小。

                """
                logger.debug(f'***5***--5-3**5 begin block_lanczos() ')
                # 初始化分块正交基
                Q = torch.zeros((k+1)*block_size, dim, device=self.device)
                T = torch.zeros(k*block_size, k*block_size, device=self.device)

                # 生成初始分块
                V = torch.randn(dim, block_size, device=self.device)
                V, _ = torch.linalg.qr(V)  # 正交化
                Q[:block_size] = V.T

                for i in range(k):
                    start_idx = i * block_size
                    # 计算Hessian作用
                    HV = torch.stack([hvp_func(Q[start_idx + j])
                                     for j in range(block_size)])

                    # 正交化过程
                    for j in range(start_idx, start_idx + block_size):
                        T[j, :j+1] = Q[:j+1] @ HV[j-start_idx]  # 计算三对角矩阵 T

                        # 🔹 在计算前检查形状
                        # logger.debug(f"🔹 Q[:j+1].shape: {Q[:j+1].shape}")
                        # logger.debug(f"🔹 Q[:j+1].T.shape: {Q[:j+1].T.shape}")
                        # logger.debug(f"🔹 T[j, :j+1].shape: {T[j, :j+1].shape}")
                        # logger.debug(f"🔹 T[j, :j+1].unsqueeze(1).shape: {T[j, :j+1].unsqueeze(1).shape}")
                        # logger.debug(f"🔹 HV[j-start_idx].shape: {HV[j-start_idx].shape}")

                        # ✅ 修正形状
                        HV[j-start_idx] -= (Q[:j+1].T @
                                            T[j, :j+1].unsqueeze(1)).squeeze()

                    # QR分解
                    V, R = torch.linalg.qr(HV.T)  # 正交化
                    Q[start_idx+block_size:start_idx+2*block_size] = V.T

                    # ✅ 修正错误：确保索引范围不会为空
                    if start_idx+block_size < T.shape[0]:
                        end_row = min(start_idx+2*block_size, T.shape[0])
                        end_col = min(start_idx+block_size, T.shape[1])

                        # logger.debug(f"🔹 Updating T matrix at [{start_idx+block_size}:{end_row}, {start_idx}:{end_col}]")

                        T[start_idx+block_size:end_row, start_idx:end_col] = R.T
                    else:
                        logger.debug(
                            f"❌ Skipping T update at [{start_idx+block_size}:{end_row}, {start_idx}:{end_col}] to prevent empty slice.")

                # 计算特征值
                T_np = T.cpu().numpy()
                eigvals = np.linalg.eigvalsh(T_np)
                return eigvals[-block_size:]  # 返回最大特征值

        # --------------------- 主计算流程 ---------------------
        logger.debug(f'***5***--5-3**3 begin main loop   ')
        # 只存储初始状态（不带梯度）,用于恢复模型状态
        original_params_to_calculate_hessian = {}
        if flag_lora:
            # 如果要评估的模型是使用lora方法训练得到的
            Flag_Nlora_task = False
            Flag_Nlora_full = False
            Flag_lora = True

            # 保存路径
            hessian_file_lora = os.path.join(
                output_dir, f"{name}_lora_only-predictDataset_lanczos.h5")

            dom_eigs_lora = []

            # 获取需要计算梯度 和 Hessian 矩阵的 参数信息
            # 对于lora方法，只计算lora_部分的 梯度 和 Hessian 矩阵
            lora_params = {}
            for name, param in model.named_parameters():
                if name.find("lora_") != -1:
                    lora_params[name] = param
                    original_params_to_calculate_hessian[name] = param.data.clone(
                    )

        elif flag_Nlora_full:
            # 如果要评估的模型是使用N_lora方法训练得到的
            # 并且这里要进行评估的对象是 模型所有与Lora相关的部分（newlora_ lora_）
            Flag_Nlora_task = False
            Flag_Nlora_full = True
            Flag_lora = False

            # 保存路径
            hessian_file_Nlora_fulllora = os.path.join(
                output_dir, f"{name}_Nlora_full-predictDataset.h5")

            # 获取需要计算梯度 和 Hessian 矩阵的 参数信息
            # 对于Nlora方法，
            # 由于是flag_Nlora_full ，所以计算lora_和 newlora_ 部分的 梯度 和 Hessian 矩阵
            dom_eigs_Nlora_lora = []
            Nlora_params_lora = {}
            for name, param in model.named_parameters():
                if name.find("loranew_") != -1:
                    # 当使用Nlora 方法时，需要对lora_ 和 loranew_ 进行区分
                    Nlora_params_lora[name] = param
                    original_params_to_calculate_hessian[name] = param.data.clone(
                    )
                elif name.find("lora_") != -1:
                    # 当使用lora 方法时，只有一个 lora的部分 进行扰动只考虑 lora 的部分，即只更新
                    Nlora_params_lora[name] = param
                    original_params_to_calculate_hessian[name] = param.data.clone(
                    )

        elif (not flag_Nlora_full):
            # 如果要评估的模型是使用N_lora方法训练得到的
            # 并且这里要进行评估的对象是 模型只与Lora相关的部分（newlora_ ）
            Flag_Nlora_task = True
            Flag_Nlora_full = False
            Flag_lora = False

            hessian_file_Nlora_tasklora = os.path.join(
                output_dir, f"{name}_Nlora_only-predictDataset.h5")

            # 获取需要计算梯度 和 Hessian 矩阵的 参数信息
            # 对于Nlora方法，
            # 由于是flag_Nlora_task ，所以只计算 newlora_ 部分的 梯度 和 Hessian 矩阵
            dom_eigs_Nlora_tasklora = []
            Nlora_params_tasklora = {}
            for name, param in model.named_parameters():
                if name.find("loranew_") != -1:
                    # 当使用Nlora 方法时，需要对lora_ 和 loranew_ 进行区分
                    # 进行扰动只考虑 loranew 的部分，即只更新  与任务有关的那一部分 lora
                    Nlora_params_tasklora[name] = param
                    original_params_to_calculate_hessian[name] = param.data.clone(
                    )

        calculator = HessianCalculator(model, device)

        try:
            for batch in tqdm(all_batches, desc=f"Rank {rank}: Processing"):

                # 恢复参数时仅操作需要修改的部分（优化点1）
                for name in original_params_to_calculate_hessian:
                    model.state_dict()[name].copy_(
                        original_params_to_calculate_hessian[name])

                # 输出 batch 和参数的类型
                logger.debug(
                    f'***5***--5-3**Model device: {next(model.parameters()).device}, '
                    f'Model param type: {next(model.parameters()).dtype}, '
                    f'Batch device: {next(iter(batch.values())).device}, '
                    f'Batch type: {next(iter(batch.values())).dtype}'
                )

                if Flag_lora:
                    logger.debug(f"Type of lora_params: {type(lora_params)}")

                    first_param = next(iter(lora_params.values()))
                    logger.debug(
                        f"First param dtype: {first_param.dtype}, device: {first_param.device}")

                    # logger.debug(f"Example entry in lora_params: {list(lora_params.items())[:5]}")  # 只打印前5个
                    logger.debug(
                        f"Type of original_params_to_calculate_hessian: {type(original_params_to_calculate_hessian)}")
                    # logger.debug(f"Example original_params_to_calculate_hessian: {list(original_params_to_calculate_hessian.items())[:5]}")  # 只打印前5个

                    hvp_lora = calculator.compute_hvp(batch, lora_params)
                    eigvals = calculator.block_lanczos(
                        hvp_lora, dim=sum(p.numel() for p in lora_params.values()))
                    dom_eigs_lora.extend(eigvals.tolist())
                    # tridiag = calculator.lanczos_algorithm(hvp_lora, dim=sum(p.numel() for p in lora_params.values()))
                    # dom_eigs_lora.extend(torch.linalg.eigvalsh(tridiag).tolist())

                # 计算LoRA Hessian
                elif Flag_Nlora_full:

                    logger.debug(
                        f"Type of Nlora_params_lora: {type(Nlora_params_lora)}")

                    first_param = next(iter(Nlora_params_lora.values()))
                    logger.debug(
                        f"First param dtype: {first_param.dtype}, device: {first_param.device}")

                    # logger.debug(f"Example entry in lora_params: {list(lora_params.items())[:5]}")  # 只打印前5个
                    logger.debug(
                        f"Type of original_params_to_calculate_hessian: {type(original_params_to_calculate_hessian)}")
                    # logger.debug(f"Example original_params_to_calculate_hessian: {list(original_params_to_calculate_hessian.items())[:5]}")  # 只打印前5个

                    hvp_Nlora_lora = calculator.compute_hvp(
                        batch, Nlora_params_lora)
                    eigvals = calculator.block_lanczos(hvp_Nlora_lora, dim=sum(
                        p.numel() for p in Nlora_params_lora.values()))
                    dom_eigs_Nlora_lora.extend(eigvals.tolist())

                else:

                    logger.debug(
                        f"Type of Nlora_params_tasklora: {type(Nlora_params_tasklora)}")

                    first_param = next(iter(Nlora_params_tasklora.values()))
                    logger.debug(
                        f"First param dtype: {first_param.dtype}, device: {first_param.device}")

                    # logger.debug(f"Example entry in lora_params: {list(lora_params.items())[:5]}")  # 只打印前5个
                    logger.debug(
                        f"Type of original_params_to_calculate_hessian: {type(original_params_to_calculate_hessian)}")
                    # logger.debug(f"Example original_params_to_calculate_hessian: {list(original_params_to_calculate_hessian.items())[:5]}")  # 只打印前5个

                    hvp_Nlora_tasklora = calculator.compute_hvp(
                        batch, Nlora_params_tasklora)
                    eigvals = calculator.block_lanczos(hvp_Nlora_tasklora, dim=sum(
                        p.numel() for p in Nlora_params_tasklora.values()))
                    dom_eigs_Nlora_tasklora.extend(eigvals.tolist())

                # 内存清理
                torch.cuda.empty_cache()

        except RuntimeError as e:
            logger.error(f"Hessian计算失败: {str(e)}")
            if "CUDA out of memory" in str(e):
                logger.warning("尝试启用梯度检查点...")

        # --------------------- 结果处理与保存 ---------------------
        logger.debug(f'***5***--5-3**6 begin save ')
        # 分布式结果聚合（改进点11）
        # 分布式结果聚合
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()

            # 选择需要聚合的变量
            if Flag_lora:
                dom_eigs_tensor = torch.tensor(dom_eigs_lora, device=device)
                hessian_file = hessian_file_lora
            elif Flag_Nlora_task:
                dom_eigs_tensor = torch.tensor(
                    dom_eigs_Nlora_tasklora, device=device)
                hessian_file = hessian_file_Nlora_tasklora
            elif Flag_Nlora_full:
                dom_eigs_tensor = torch.tensor(
                    dom_eigs_Nlora_lora, device=device)
                hessian_file = hessian_file_Nlora_fulllora
            else:
                dom_eigs_tensor, hessian_file = None, None

            # 仅当满足某个条件时才执行分布式聚合
            if dom_eigs_tensor is not None:
                all_dom_eigs = [torch.zeros_like(
                    dom_eigs_tensor) for _ in range(world_size)]
                torch.distributed.all_gather(all_dom_eigs, dom_eigs_tensor)
                dom_eigs = torch.cat(
                    all_dom_eigs, dim=0).cpu().numpy().tolist()

                # 计算统计指标
                stats = self._compute_stats(dom_eigs)

        # 仅 rank=0 进程保存 HDF5 结果，避免冲突
        if rank == 0 and hessian_file is not None:
            with h5py.File(hessian_file, "w") as hf:
                hf.attrs["created_at"] = datetime.now().isoformat()
                hf.attrs["model_type"] = type(model).__name__
                for k, v in stats.items():
                    hf.create_dataset(k, data=v)
                hf.create_dataset("dominant_eigs", data=np.array(dom_eigs))

            logger.debug(f"计算完成，结果保存至 {hessian_file} 文件")

        # 添加进程同步 & 关闭分布式进程
        if torch.distributed.is_initialized():
            logger.debug("所有进程同步中...")
            torch.distributed.barrier()  # 确保所有进程都完成再继续

            if torch.distributed.get_rank() == 0:
                logger.debug("所有进程已完成计算，开始关闭分布式进程...")

            torch.distributed.destroy_process_group()  # 释放 NCCL 资源
            logger.debug("分布式进程已正确关闭")
        return True


def extract_logits(out, task_ids, n_classes, device):
    """
    Extract logits corresponding to task_ids from out.

    Args:
        out: Predictions
        task_ids: Task ids
        n_classes: Number of classes per task
    """
    indices = (
        (torch.arange(n_classes * out.size(0)) % n_classes)
        .reshape(out.size(0), n_classes)
        .to(device=device)
    )
    indices = indices + (task_ids * n_classes).unsqueeze(1)
    return out[torch.arange(out.size(0)).unsqueeze(1), indices]



class ContinualTrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        args,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        data_collator=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        num_classes=100,
        classes_per_task=5,
        dataloaders=None,
        subset_dataloaders=None,
        task_id=0
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        self.task_id = task_id
        self.num_classes = num_classes
        self.classes_per_task = classes_per_task
        self.dataloaders = dataloaders
        self.subset_dataloaders = subset_dataloaders


        # ✅ 先初始化 `criterion`
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

        # ✅ 先创建优化器 & 调度器
        self.create_optimizer_and_scheduler(num_training_steps=10)  # 这里的 `num_training_steps` 可调整

        # ✅ 确保 `model_init` 和 `optimizers` 不能同时传入
        if self.model_init is not None and (self.optimizer is not None or self.lr_scheduler is not None):
            raise RuntimeError(
                "Passing a `model_init` is incompatible with providing the `optimizers` argument. "
                "You should subclass `Trainer` and override the `create_optimizer_and_scheduler` method."
            )

        # ✅ 现在 `self.criterion` 已经存在，可以安全初始化 `algo`
        self.algo = self._init_algo(args)

        # ✅ 记录 Trainer 重要信息
        self.log_trainer_info()

    def log_trainer_info(self):
        """输出 `Trainer` 相关的关键信息"""
        logger.info("🚀 Trainer 初始化完成！")
        logger.info(f"📌 任务 ID: {self.task_id}")
        logger.info(f"📌 设备: {self.args.device}")
        logger.info(f"📌 训练方法: {self.args.method}")
        logger.info(f"📌 学习率: {self.args.learning_rate}")
        logger.info(f"📌 任务类别数: {self.num_classes}, 每个任务类别: {self.classes_per_task}")
        logger.info(f"📌 SAM 训练: {'启用' if self.args.sam else '禁用'}, ρ: {self.args.rho if self.args.sam else 'N/A'}")
        logger.info(f"📌 使用优化器: {self.optimizer}")
        logger.info(f"📌 训练数据加载器: {'已加载' if self.dataloaders else '未加载'}")
        logger.info(f"📌 子集数据加载器: {'已加载' if self.subset_dataloaders else '未加载'}")
        logger.info(f"📌 训练集大小: {len(self.train_dataset) if self.train_dataset else 'N/A'}")
        logger.info(f"📌 测试集大小: {len(self.eval_dataset) if self.eval_dataset else 'N/A'}")


    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        自定义优化器和学习率调度器
        """
        # 初始化优化器
        if self.args.sam:
            self.optimizer = SAM(self.model.parameters(), torch.optim.SGD, rho=self.args.rho, lr=self.args.learning_rate)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate, momentum=self.args.momentum)

        # 学习率调度器
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=num_training_steps // 3, gamma=0.1)

        logger.info(f"✅ 自定义优化器: {self.optimizer}")
        logger.info(f"✅ 自定义学习率调度器: {self.lr_scheduler}")

    


    def _init_algo(self, args):
        """初始化训练方法"""
        if args.method == "er":
            return ER(args, self.num_classes)
        elif args.method == "agem":
            return AGEM(args, self.model, self.optimizer, self.criterion, self.classes_per_task, len(self.dataloaders))
        elif args.method == "ewc":
            return EWC(self.model, self.criterion)
        return None
    
    def train(self, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None, **kwargs):
        """
        重载 Huggingface Trainer 的 train() 方法，
        以支持 SAM 优化器（需要 closure）而避免二次调用 optimizer.step()
        """
        # === 初始化 ===
        self._memory_metrics = {}
        self.is_in_train = True

        # === 恢复状态 ===
        model = self._wrap_model(self.model, training=True)
        self._hp_search_setup(trial)

        # === 训练数据集 ===
        train_dataloader = self.get_train_dataloader()
        if not has_length(train_dataloader):
            raise ValueError("`train_dataloader` does not have length, which is required")

        num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
        num_train_epochs = self.args.num_train_epochs
        num_train_epochs = int(self.args.num_train_epochs)  # 🔧 fix: 转为 int
        max_steps = int(num_train_epochs * num_update_steps_per_epoch)

        self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # === 设置控制器和状态 ===
        self.state = TrainerState()
        self.control = TrainerControl()
        self.state.max_steps = max_steps
        self.state.epoch = 0

        # === 训练主循环 ===
        logger.info("\n🚀 Start Training...")
        for epoch in range(num_train_epochs):
            self.control = self.callback_handler.on_epoch_begin(self.args, self.state, self.control)
            self.train_one_epoch_with_sam(train_dataloader)
            self.state.epoch += 1
            self.control = self.callback_handler.on_epoch_end(self.args, self.state, self.control)

            if self.control.should_training_stop:
                break

        self.is_in_train = False
        return self.state
    

    def train_one_epoch_with_sam(self, dataloader):
        """
        每个 epoch 的训练逻辑，支持 SAM 优化器的 closure 调用
        """
        self.model.train()

        for step, inputs in enumerate(dataloader):
            self.control = self.callback_handler.on_step_begin(self.args, self.state, self.control)

            # === 执行自定义 training_step（已内置 SAM 支持） ===
            loss = self.training_step(self.model, inputs)

            # === 更新 scheduler（注意：不要调用 self.optimizer.step()，SAM 已包含 step） ===
            self.lr_scheduler.step()

            # === 日志记录和回调 ===
            self.state.global_step += 1
            self.state.log_history.append({"loss": loss.item(), "step": self.state.global_step})
            self.control = self.callback_handler.on_step_end(self.args, self.state, self.control)

            if self.control.should_training_stop or self.control.should_epoch_stop:
                break
    
    
    def training_step(self, model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        自定义训练步骤，适配 transformers.Trainer
        - 仅执行 **单个 step** 训练
        - 任务级别学习率衰减 (`gamma ** task_id`)
        - 兼容 `SAM`、`EWC`、`AGEM`
        """
        model.train()
        
        # 计算任务级别的 `gamma ** task_id` 学习率衰减
        if self.args.lr_scheduler_type == "constant":
            lr = max(self.args.learning_rate * (self.args.gamma ** self.task_id), 0.00005)
        else:
            lr = self.args.learning_rate  # 保持默认 `Trainer` 方式
        
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        # 取出 `pixel_values` 和 `labels`，确保适配 ViT
        pixel_values = inputs["pixel_values"].to(self.args.device)
        labels = inputs["labels"].to(self.args.device)

        # ⚙️ 构造 closure 函数：前向 + 反向 + 方法适配
        def closure():
            self.optimizer.zero_grad()

            outputs = model(pixel_values)
            logits = outputs.logits
            loss = self.criterion(logits, labels)

            # === 方法兼容处理 ===
            if self.args.method == "er":
                if self.task_id > 0:
                    mem_x, mem_y, mem_task_ids = self.algo.sample(
                        self.args.batch_size, exclude_task=None, pr=False
                    )
                    mem_pred = model(mem_x).logits
                    loss_mem = self.criterion(mem_pred, mem_y)
                    loss += loss_mem
                self.algo.add_reservoir(pixel_values, labels, None, self.task_id)

            elif self.args.method == "ewc":
                loss += self.args.lambd * self.algo.penalty(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 100)

            elif self.args.method == "agem":
                # ⚠️ AGEM 特殊处理：替换前向 + 梯度，不需要 backward
                self.algo.observe_agem(model, pixel_values, self.task_id, labels)
                return torch.tensor(0.0, device=self.args.device, requires_grad=True)

            loss.backward()
            return loss

        # === SAM 优化器路径 ===
        if self.args.sam:
            enable_running_stats(model)
            loss = closure()  # 第一次前向 + backward
            self.optimizer.step(closure=closure)  # SAM 自动处理 two-step
            disable_running_stats(model)

        # === 非 SAM 优化器路径 ===
        else:
            loss = closure()
            self.optimizer.step()

        return loss.detach()



    # def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
    #     """自定义训练步骤"""
    #     model.train()

    #     train_loader = self.dataloaders[self.task_id]["train"]
    #     subset_loader = self.subset_dataloaders[self.task_id] if self.subset_dataloaders else None

    #     # 计算学习率衰减，这里的学习率是每一个任务是固定的
    #     # lr = max(self.args.learning_rate * (self.args.lr_scheduler_type ** self.task_id), 0.00005)
    #     if self.args.lr_scheduler_type == "constant":
    #         lr = max(self.args.learning_rate  * (self.args.gamma ** self.task_id), 0.00005)
        
        
    #     for param_group in self.optimizer.param_groups:
    #         param_group["lr"] = lr

    #     # 训练循环
    #     logger.debug(f'***-----begin train')
    #     iterator = tqdm(range(self.args.num_train_epochs)) if self.args.num_train_epochs > 1 else range(self.args.num_train_epochs)
    #     for _ in iterator:
    #         self._train_single_epoch(model, train_loader, subset_loader)

    #     # 如果使用 EWC，则更新重要性参数
    #     if self.args.method == "ewc":
    #         loader = DataLoader(train_loader.dataset, batch_size=200, shuffle=True)
    #         self.algo.update(model, 0 if self.args.class_incremental else self.task_id, loader)

    #     # 保存模型
    #     os.makedirs(os.path.join(self.args.output_dir, "models"), exist_ok=True)
    #     torch.save({"model": model.state_dict()},
    #                os.path.join(self.args.output_dir, "models", f"task_{self.task_id}_model.pt"))
        
        

    # def _train_single_epoch(self, model, dataloader, subset_dataloader):
    #     """执行单个 epoch 训练"""
    #     model.train()
    #     for X, y in iter(dataloader):
    #         model.zero_grad()

    #         X, y = X.to(self.args.device), y.to(self.args.device)

    #         if self.args.sam:
    #             enable_running_stats(model)

    #         out = model(X, self.task_id)

    #         if self.args.method == "er":
    #             if self.task_id > 0:
    #                 mem_x, mem_y, mem_task_ids = self.algo.sample(self.args.batch_size, exclude_task=None, pr=False)
    #                 mem_pred = model(mem_x, None)
    #                 mem_pred = extract_logits(mem_pred, mem_task_ids, self.classes_per_task, self.args.device)
    #                 loss_mem = self.criterion(mem_pred, mem_y)
    #                 loss_mem.backward()
    #             self.algo.add_reservoir(X, y, None, self.task_id)

    #         elif self.args.method == "ewc":
    #             loss_ewc = self.args.lambd * self.algo.penalty(model)
    #             loss_ewc.backward()
    #             torch.nn.utils.clip_grad_norm_(model.parameters(), 100)

    #         elif self.args.method == "agem":
    #             model = self.algo.observe_agem(model, X, self.task_id, y)

    #         # 计算损失并优化
    #         if self.args.method != "agem":
    #             loss = self.criterion(out, y)
    #             loss.backward()

    #         if self.args.sam:
    #             self.optimizer.first_step(zero_grad=True)
    #             disable_running_stats(model)
    #             self.criterion(model(X, self.task_id), y).backward()
    #             self.optimizer.second_step(zero_grad=True)
    #         else:
    #             self.optimizer.step()