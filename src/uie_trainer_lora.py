import torch
from transformers import GenerationConfig
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.trainer import *
from transformers.trainer_callback import TrainerCallback

from uie_collator import SUPPORTED_DECODER_MODELS, check_model
from uie_dataset_lora import ANSWER_PREFIX

from datatime import datatime

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

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader.dataset):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

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

    def compute_loss_landscape(
        self,   eval_dataset: Dataset, output_dir, name="lossLandscape", x_range=(-1, 1), y_range=(-1, 1), num_points=20, max_batches=5,
        sample_batches=False, flag_lora=True, flag_Nlora_task=False,
    ):
        """
        计算损失景观，并分别计算:
        1️⃣ **完整模型（Base Model + LoRA Adapter）** 的损失景观
        2️⃣ **仅 LoRA 适配器（LoRA Adapter）** 的损失景观
        计算损失景观，并分别计算完整模型 & LoRA Adapter 的损失表面。
        针对大模型使用多 GPU，可以并行分配 (i, j) 坐标网格。
        """
        args = self.args
        device = args.device
        # 最终损失网络的数值
        loss_grid = np.zeros((num_points, num_points))

        # 根据模型判断，这里不妨使用手动
        if flag_lora:
            # 如果是 lora 方法，干扰 lora 部分
            Flag_Nlora_task = False
            Flag_Nlora_full = False
            Flag_lora = True
            surf_file_lora = os.path.join(
                output_dir, f"{name}_lora_only-evalDataset-orial.h5")
            surf_file = surf_file_lora
        elif flag_Nlora_task:
            Flag_Nlora_task = True
            Flag_Nlora_full = False
            Flag_lora = False
            surf_file_Nlora_tasklora = os.path.join(
                output_dir, f"{name}_Nlora_onlytask-predictDataset-2.h5")
            surf_file = surf_file_Nlora_tasklora
        else:
            Flag_Nlora_task = True
            Flag_Nlora_full = True
            Flag_lora = False
            surf_file_Nlora_full = os.path.join(
                output_dir, f"{name}_Nlora_full-predictDataset.h5")
            surf_file = surf_file_Nlora_full

        if not os.path.exists(surf_file):
            with open(surf_file, 'w') as f:
                pass  # 成功创建空文件

        # ✅ 记录日志信息
        logger.info(f'***5***--5-2 **1 compute_loss_landscape  ')
        logger.info(
            f"***** Running Loss Landscape Calculation on expriment Nlora_task:{Flag_Nlora_task} ,Nlora_full:{Flag_Nlora_full}  lora:{Flag_lora}*****")
        logger.info(
            f"Output Dir = {output_dir}，Output File :{surf_file} Num points = {num_points}x{num_points} max_batches = {max_batches}")

        # ✅ 兼容 AMP 和分布式训练
        # 复制模型避免污染
        model = copy.deepcopy(self.model)
        model = self._wrap_model(model, training=False)

        # ✅ 处理 FP16/BF16 评估模式
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=device)

        model = model.to(device=device)
        model.eval()  # 确保模型在 eval 模式

        # --------------------- 生成参数扰动阶段阶段 ---------------------
        # 确定需要扰动的参数名称
        # 根据不同的微调方法（Nlora或lora）确定需要保存原始值的参数
        original_params_to_perturb = {}
        for name, param in model.named_parameters():
            # 当使用Nlora方法时
            if Flag_Nlora_task and (not Flag_Nlora_full):
                # 只保存与任务相关的新增lora参数（loranew_开头），且排除共享参数
                if name.find("loranew_") != -1 and name.find("shared") == -1:
                    original_params_to_perturb[name] = param.data.clone()
            elif Flag_Nlora_task and (Flag_Nlora_full):
                if ("loranew_" in name or "lora_" in name) and "shared" not in name:
                    original_params_to_perturb[name] = param.data.clone()
            # 当使用标准lora方法时,或者是Nlora方法的full 进行评估
            elif Flag_lora:
                # 保存所有lora参数（lora_开头），且排除共享参数
                if name.find("lora_") != -1 and name.find("shared") == -1:
                    original_params_to_perturb[name] = param.data.clone()

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

                    # 生成正交扰动
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
            # all_batches = all_batches[rank::world_size]  # 数据分片
            logger.info(
                f'***5***--5-2**4 distribute--yes world_size:{world_size} rank:{rank},start_idx:{start_idx},end_idx:{end_idx} ')
        else:
            # logger.info(f'****5***--5-2**2-1 if**5***--5-2**5 distribute--no ')
            world_size = 1
            rank = 0
            start_idx, end_idx = 0, num_points
            logger.info(
                f'***5***--5-2**4 distribute--no world_size:{world_size} rank:{rank},start_idx:{start_idx},end_idx:{end_idx} ')

        # --------------------- 数据准备阶段 ---------------------
        logger.info(
            f'***5***--5-2**2 begin load data---{torch.distributed.is_initialized()} ')
        dataloader = self.get_eval_dataloader(eval_dataset)
        all_batches = list(dataloader)[:max_batches]
        logger.info(f'***5***--5-2**2 begin load data---{len(all_batches)} ')

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
                    with torch.cuda.amp.autocast(enabled=args.fp16):  # 支持混合精度
                        outputs = model(**inputs)
                        loss = F.cross_entropy(
                            outputs.logits.view(-1, outputs.logits.size(-1)),
                            inputs["labels"].view(-1)
                        )
                    total_loss += loss.item()

                loss_grid[i, j] = total_loss / len(all_batches)

                # 每5次迭代清理一次缓存（优化点6）
                if j % 5 == 0:
                    torch.cuda.empty_cache()

        # --------------------- 分布式结果收集 ---------------------
        if torch.distributed.is_initialized():
            # 收集所有进程的loss_grid
            # 将 loss_grid 转换为张量
            loss_grid_tensor = torch.tensor(loss_grid, device=device)
            # 创建一个列表，用于接收所有进程的 loss_grid
            all_loss = [torch.zeros_like(loss_grid_tensor)
                        for _ in range(world_size)]
            # 收集所有进程的 loss_grid
            torch.distributed.all_gather(all_loss, loss_grid_tensor)
            # 将收集到的结果拼接成一个完整的 loss_grid
            # loss_grid_tensor = torch.cat(all_loss, dim=0)
            loss_grid_tensor = torch.stack(all_loss, dim=0).mean(dim=0)
            loss_grid = loss_grid_tensor.cpu().numpy()  # 转回 NumPy 数组后再保存
        else:
            all_loss = [loss_grid]

        # 仅rank 0进程保存结果
        if rank == 0:
            with h5py.File(surf_file, 'w') as f:
                f.create_dataset('xcoordinates', data=x_coords)
                f.create_dataset('ycoordinates', data=y_coords)
                f.create_dataset('train_loss', data=loss_grid)

            logger.info(f"计算完成，结果保存至{surf_file}")

        return True
