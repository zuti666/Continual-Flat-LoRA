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
        sample_batches=False, flag_lora=True, flag_Nlora_full= False , flag_FullModel =  False,
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

        new_Flag  = '3.1'
        original_params_to_perturb = {}
        for name, param in model.named_parameters():

            # 注意 默认加载的模型已经添加了LoRA部分，如果是1 需要设置lora 中的代码 去掉 add_lora的代码

            if new_Flag == '1': # 只更改 W
                if  "shared" not in name:
                    param.requires_grad = True
                    original_params_to_perturb[name] = param.data.clone()
                    
                surf_file = os.path.join(output_dir, f"{save_file_name}_1.h5")

            # 注意，2.1 加载的是初始模型，然后加载过后添加了LoRA部分，进行了初始化
            elif new_Flag == '2.1': # 只更改 W
                      
                if "lora_"  in name and "shared" not in name:
                    param.requires_grad = False
                elif "lora_" not in name and "shared" not in name:
                    param.requires_grad = True
                    original_params_to_perturb[name] = param.data.clone() 

                surf_file = os.path.join(output_dir, f"{save_file_name}_2-1.h5")
            elif new_Flag == '2.2': # 只更改 AB
                if "lora_"  in name and "shared" not in name:
                    original_params_to_perturb[name] = param.data.clone()
                surf_file = os.path.join(output_dir, f"{save_file_name}_2-2.h5")
            elif new_Flag == '2.3': # 只更改 W AB
                if  "shared" not in name:
                    original_params_to_perturb[name] = param.data.clone()
                surf_file = os.path.join(output_dir, f"{save_file_name}_2-3.h5")

            # 注意，3.1 加载的是训练保存得到的adapter模型，里面具有训练后的LoRA部分
            elif new_Flag == '3.1': # 只更改 W

                if "lora_"  in name and "shared" not in name:
                    param.requires_grad = False
                elif "lora_" not in name and "shared" not in name:
                    param.requires_grad = True
                    original_params_to_perturb[name] = param.data.clone() 

                surf_file = os.path.join(output_dir, f"{save_file_name}_3-1_t1.h5")
            elif new_Flag == '3.2': # 只更改 AB
                if "lora_"  in name and "shared" not in name:
                    original_params_to_perturb[name] = param.data.clone()
                surf_file = os.path.join(output_dir, f"{save_file_name}_3-2.h5")
            elif new_Flag == '3.3': # 只更改 W AB
                if  "shared" not in name:
                    original_params_to_perturb[name] = param.data.clone()
                surf_file = os.path.join(output_dir, f"{save_file_name}_3-3.h5")



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
            logger.debug(f"name: {name}, requires_grad: {model.state_dict()[name].requires_grad}")



        logger.debug(f"Output Dir = {output_dir}，Output File :{surf_file} Num points = {num_points}x{num_points} max_batches = {max_batches}")

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
                    model.state_dict()[name].copy_(original_params_to_perturb[name])

                # 应用扰动
                with torch.no_grad():
                    for name in original_params_to_perturb:
                        param = model.state_dict()[name]
                        delta = xv * \
                            perturb_x[name].to(param.dtype) + yv * perturb_y[name].to(param.dtype)
                        param.add_(delta)  # 原位操作减少内存分配

                # 计算损失
                total_loss = 0.0
                for batch in all_batches:
                    inputs = {k: v.to(device) for k, v in batch.items()}
                    with torch.cuda.amp.autocast(enabled=args.fp16_full_eval):  # 支持混合精度
                        outputs = model(**inputs)
                        loss = F.cross_entropy(
                            outputs.logits.view(-1, outputs.logits.size(-1)),
                            inputs["labels"].view(-1)
                        )
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
            loss_grid_tensor = torch.stack(all_loss, dim=0).mean(dim=0) # 取平均
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
        logger.debug(f'***5***--5-3**1 use distribute ---- {torch.distributed.is_initialized()}')
        
        args = self.args
        device = args.device


            
        # 创建独立模型副本（关键改进点1：隔离原始模型）
        model = copy.deepcopy(self.model)
        model = self._wrap_model(model, training=False)
        
        # 混合精度处理
        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device

        # 设置要评估的参数
         #设置需要进行评估的参数
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
            logger.debug(f'args.fp16_full_eval:{args.fp16_full_eval}, args.bf16_full_eval:{args.bf16_full_eval}')
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
            logger.debug(f'***5***--LORA Hessian**2 distribute yes, rank:{rank}, total_batches:{len(all_batches)}')

        else:
            logger.debug(f'***5***--5-3**4 distribute--no ')
            world_size = 1
            rank = 0
        
        # 数据采样处理
        logger.debug(f'***5***--5-3**2-1 if  {len(all_batches)} , {max_batches} ')
        if len(all_batches) > max_batches:
            if sample_batches:
                indices = np.random.choice(len(all_batches), max_batches, replace=False)
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
            def __init__(self, model, device,max_dim=10000):
                logger.debug(f'***5***--5-3**3 class HessianCalculator init   ')
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
                self.model.zero_grad() # 清空梯度
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
                    params = [p for p in param_list.values() if p.requires_grad]
                elif isinstance(param_list, (list, tuple)):
                    params = [p for p in param_list if p.requires_grad]
                else:
                    params = []  # 若传入为 None 或其它类型，则空处理

                # 计算一阶梯度
                grads = torch.autograd.grad(loss, params, create_graph=True)

                def hvp_func(v):
                    """计算 Hessian 向量积,闭包函数保持计算图"""
                    
                                
                    split_sizes = [p.numel() for p in params]  # 计算每个 param 对应的大小
                    v_split = torch.split(v, split_sizes)  # 按照每个参数的形状拆分 v
                    v_reshaped = [v_i.view(p.shape) for v_i, p in zip(v_split, params)]  # 重新调整 v_i 形状

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
                    HV = torch.stack([hvp_func(Q[start_idx + j]) for j in range(block_size)])
                    
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
                        HV[j-start_idx] -= (Q[:j+1].T @ T[j, :j+1].unsqueeze(1)).squeeze()
                    
                    # QR分解
                    V, R = torch.linalg.qr(HV.T) # 正交化
                    Q[start_idx+block_size:start_idx+2*block_size] = V.T

                    # ✅ 修正错误：确保索引范围不会为空
                    if start_idx+block_size < T.shape[0]:  
                        end_row = min(start_idx+2*block_size, T.shape[0])
                        end_col = min(start_idx+block_size, T.shape[1])

                        # logger.debug(f"🔹 Updating T matrix at [{start_idx+block_size}:{end_row}, {start_idx}:{end_col}]")

                        T[start_idx+block_size:end_row, start_idx:end_col] = R.T
                    else:
                        logger.debug(f"❌ Skipping T update at [{start_idx+block_size}:{end_row}, {start_idx}:{end_col}] to prevent empty slice.")


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
            hessian_file_lora = os.path.join(output_dir, f"{name}_lora_only-predictDataset_lanczos.h5")

            dom_eigs_lora = []

            # 获取需要计算梯度 和 Hessian 矩阵的 参数信息
            # 对于lora方法，只计算lora_部分的 梯度 和 Hessian 矩阵
            lora_params = {}
            for name, param in model.named_parameters():
                if name.find("lora_") != -1:
                        lora_params[name] = param
                        original_params_to_calculate_hessian[name] = param.data.clone()
        
        elif  flag_Nlora_full:
            # 如果要评估的模型是使用N_lora方法训练得到的
            # 并且这里要进行评估的对象是 模型所有与Lora相关的部分（newlora_ lora_）
            Flag_Nlora_task = False  
            Flag_Nlora_full = True 
            Flag_lora = False

            # 保存路径
            hessian_file_Nlora_fulllora = os.path.join(output_dir, f"{name}_Nlora_full-predictDataset.h5")
        
            # 获取需要计算梯度 和 Hessian 矩阵的 参数信息
            # 对于Nlora方法，
            # 由于是flag_Nlora_full ，所以计算lora_和 newlora_ 部分的 梯度 和 Hessian 矩阵
            dom_eigs_Nlora_lora = []
            Nlora_params_lora = {}
            for name, param in model.named_parameters():
                if name.find("loranew_") != -1:
                    # 当使用Nlora 方法时，需要对lora_ 和 loranew_ 进行区分
                    Nlora_params_lora[name] = param
                    original_params_to_calculate_hessian[name] = param.data.clone()
                elif name.find("lora_") != -1:
                # 当使用lora 方法时，只有一个 lora的部分 进行扰动只考虑 lora 的部分，即只更新
                    Nlora_params_lora[name] = param
                    original_params_to_calculate_hessian[name] = param.data.clone()
        
        elif (not flag_Nlora_full):
            # 如果要评估的模型是使用N_lora方法训练得到的
            # 并且这里要进行评估的对象是 模型只与Lora相关的部分（newlora_ ）
            Flag_Nlora_task = True 
            Flag_Nlora_full = False
            Flag_lora = False

            hessian_file_Nlora_tasklora = os.path.join(output_dir, f"{name}_Nlora_only-predictDataset.h5")
        

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
                    original_params_to_calculate_hessian[name] = param.data.clone()

        
        calculator = HessianCalculator(model, device)


        try:
            for batch in tqdm(all_batches, desc=f"Rank {rank}: Processing"):
                
                # 恢复参数时仅操作需要修改的部分（优化点1）
                for name in original_params_to_calculate_hessian:
                    model.state_dict()[name].copy_(original_params_to_calculate_hessian[name])
    
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
                    logger.debug(f"First param dtype: {first_param.dtype}, device: {first_param.device}")

                    # logger.debug(f"Example entry in lora_params: {list(lora_params.items())[:5]}")  # 只打印前5个
                    logger.debug(f"Type of original_params_to_calculate_hessian: {type(original_params_to_calculate_hessian)}")
                    # logger.debug(f"Example original_params_to_calculate_hessian: {list(original_params_to_calculate_hessian.items())[:5]}")  # 只打印前5个
                    
                    hvp_lora = calculator.compute_hvp(batch, lora_params)
                    eigvals = calculator.block_lanczos(hvp_lora, dim=sum(p.numel() for p in lora_params.values()))
                    dom_eigs_lora.extend(eigvals.tolist())
                    # tridiag = calculator.lanczos_algorithm(hvp_lora, dim=sum(p.numel() for p in lora_params.values()))
                    # dom_eigs_lora.extend(torch.linalg.eigvalsh(tridiag).tolist())

                # 计算LoRA Hessian
                elif Flag_Nlora_full:

                    logger.debug(f"Type of Nlora_params_lora: {type(Nlora_params_lora)}")

                    first_param = next(iter(Nlora_params_lora.values()))
                    logger.debug(f"First param dtype: {first_param.dtype}, device: {first_param.device}")

                    # logger.debug(f"Example entry in lora_params: {list(lora_params.items())[:5]}")  # 只打印前5个
                    logger.debug(f"Type of original_params_to_calculate_hessian: {type(original_params_to_calculate_hessian)}")
                    # logger.debug(f"Example original_params_to_calculate_hessian: {list(original_params_to_calculate_hessian.items())[:5]}")  # 只打印前5个
                    

                    hvp_Nlora_lora = calculator.compute_hvp(batch,Nlora_params_lora)
                    eigvals = calculator.block_lanczos(hvp_Nlora_lora, dim=sum(p.numel() for p in Nlora_params_lora.values()))
                    dom_eigs_Nlora_lora.extend(eigvals.tolist())
                

                else:

                    logger.debug(f"Type of Nlora_params_tasklora: {type(Nlora_params_tasklora)}")

                    first_param = next(iter(Nlora_params_tasklora.values()))
                    logger.debug(f"First param dtype: {first_param.dtype}, device: {first_param.device}")

                    # logger.debug(f"Example entry in lora_params: {list(lora_params.items())[:5]}")  # 只打印前5个
                    logger.debug(f"Type of original_params_to_calculate_hessian: {type(original_params_to_calculate_hessian)}")
                    # logger.debug(f"Example original_params_to_calculate_hessian: {list(original_params_to_calculate_hessian.items())[:5]}")  # 只打印前5个
                    
                    hvp_Nlora_tasklora = calculator.compute_hvp(batch,Nlora_params_tasklora)
                    eigvals = calculator.block_lanczos(hvp_Nlora_tasklora, dim=sum(p.numel() for p in Nlora_params_tasklora.values()))
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
            elif Flag_Nlora_task :
                dom_eigs_tensor = torch.tensor(dom_eigs_Nlora_tasklora, device=device)
                hessian_file = hessian_file_Nlora_tasklora
            elif Flag_Nlora_full:
                dom_eigs_tensor = torch.tensor(dom_eigs_Nlora_lora, device=device)
                hessian_file = hessian_file_Nlora_fulllora
            else:
                dom_eigs_tensor, hessian_file = None, None

            # 仅当满足某个条件时才执行分布式聚合
            if dom_eigs_tensor is not None:
                all_dom_eigs = [torch.zeros_like(dom_eigs_tensor) for _ in range(world_size)]
                torch.distributed.all_gather(all_dom_eigs, dom_eigs_tensor)
                dom_eigs = torch.cat(all_dom_eigs, dim=0).cpu().numpy().tolist()

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
