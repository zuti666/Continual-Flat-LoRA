from collections import defaultdict
from tqdm import tqdm
import torch
import os
import torch.nn as nn

from sam import SAM, enable_running_stats, disable_running_stats
from existing_methods.agem import AGEM
from existing_methods.er import ER
from existing_methods.ewc import EWC
from scipy import optimize
from collections import defaultdict
import h5py
import json
import copy

import numpy as np

import sys
import logging
logger = logging.getLogger(__name__)
# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log_level = 10
# log_level = logging.DEBUG  # 或者 logging.DEBUG（数值 10）
logger.setLevel(log_level)
logger.warning(f'log_level:{log_level}')


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


class SAMContinualTrainer:
    def __init__(
        self,
        model,
        dataloaders,
        task_split,
        num_classes,
        classes_per_task,
        training_args,        # 💡 用明确命名代替 `args`
        criterion=None,
        algo=None,
        subset_dataloaders=None,
        task_id=None,
    ):
        self.model = model.to(training_args.device)
        self.dataloaders = dataloaders
        self.task_split = task_split
        self.args = training_args  # 仍保留原名用于内部兼容
        self.device = training_args.device
        self.num_classes = num_classes
        self.classes_per_task = classes_per_task
        self.criterion = criterion or torch.nn.CrossEntropyLoss().to(self.device)
        self.subset_dataloaders = subset_dataloaders
        self.task_id = task_id

       
        

        # ✅ 初始化优化器
        if training_args.optimizer_type.lower() == "sam":
            self.optimizer = SAM(
                self.model.parameters(),
                base_optimizer=torch.optim.SGD,
                rho=training_args.rho,
                lr=training_args.learning_rate,
                momentum=training_args.momentum,
            )
        elif training_args.optimizer_type.lower() == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=training_args.learning_rate,
                momentum=training_args.momentum,
            )
        else:
            raise ValueError(f"Unsupported optimizer_type: {training_args.optimizer_type}")

        # ✅ 初始化方法（仅在 continual 场景）
        self.finetune_strategy = training_args.finetune_strategy
        if training_args.train_method == "continual":
            if self.finetune_strategy == "er":
                self.algo = ER(training_args, num_classes)
            elif self.finetune_strategy == "agem":
                self.algo = AGEM(training_args, self.model, self.optimizer, self.criterion, classes_per_task, len(dataloaders))
            elif self.finetune_strategy == "ewc":
                self.algo = EWC(self.model, self.criterion)
            else:
                self.algo = None
        else:
            self.algo = None

        # ✅ 记录评估指标
        self.metrics = {
            "accuracies": defaultdict(list),
            "losses": defaultdict(list),
        }

        self.log_trainer_info()



    @staticmethod
    def calculate_run_metrics(acc_dict):
        num_tasks = len(acc_dict)
        average_accuracy = 0.0
        forgetting = 0.0
        learning_accuracy = acc_dict[0][0] if 0 in acc_dict and acc_dict[0] else 0.0

        for task_id in range(num_tasks):
            acc_list = acc_dict[task_id]
            average_accuracy += acc_list[-1]
            max_prev = max(acc_list[:-1]) if len(acc_list) > 1 else acc_list[0]
            forgetting += max_prev - acc_list[-1]

        average_accuracy /= num_tasks
        forgetting /= (num_tasks - 1) if num_tasks > 1 else 1

        return average_accuracy, forgetting, learning_accuracy


    def log_trainer_info(self):
        """输出 Trainer 初始化时的关键信息（结构化 & 分层清晰）"""
        logger.info("🚀 SAMContinualTrainer 初始化完成，配置信息如下：")

        # ===== 模型信息 =====
        logger.info("📦 模型相关信息：")
        logger.info(f"   ├─ 模型结构         ：{self.model.__class__.__name__}")
        logger.info(f"   ├─ 当前设备         ：{self.device}")
        logger.info(f"   └─ 分类头输出维度   ：{self.num_classes}")

        # ===== 优化器配置 =====
        logger.info("🧪 优化器与损失函数设置：")
        logger.info(f"   ├─ 优化器类型       ：{self.optimizer.__class__.__name__}")
        logger.info(f"   ├─ 学习率           ：{self.args.learning_rate}")
        logger.info(f"   ├─ 是否使用 SAM     ：{self.args.optimizer_type}")
        if self.args.optimizer_type.lower() == "sam":
            logger.info(f"   ├─ SAM ρ (rho) 值   ：{self.args.rho}")
        logger.info(f"   └─ 损失函数         ：{self.criterion.__class__.__name__}")

        # ===== 训练方法 =====
        logger.info("📘 训练方法与策略：")
        logger.info(f"   ├─ 训练方法         ：{self.args.train_method}")
        if self.args.train_method == "continual":
            logger.info(f"   └─ Continual 策略   ：{self.finetune_strategy}")
        elif self.args.train_method == "finetune":
            logger.info(f"   └─ Finetune 策略    ：{self.finetune_strategy}")
        else:
            logger.info(f"   └─ LoRA 类型         ：{getattr(self.args, 'lora_type', 'None')}")

        # ===== 数据加载状态 =====
        logger.info("📊 数据加载状态：")
        logger.info(f"   ├─ 主 dataloader     ：{'✅ 已加载' if self.dataloaders else '❌ 未加载'}")
        logger.info(f"   ├─ 子集 dataloader   ：{'✅ 已加载' if self.subset_dataloaders else '❌ 未加载'}")
        logger.info(f"   └─ 总任务数量       ：{len(self.task_split)}")

        # ===== 任务划分情况 =====
        logger.info("🧩 任务划分示例（最多显示前 3 个任务）：")
        for i, task in enumerate(self.task_split[:3]):
            logger.info(f"   ├─ 任务 {i} 类别     ：{task}")
        if len(self.task_split) > 3:
            logger.info(f"   └─ ... 共 {len(self.task_split)} 个任务")
        if self.task_id :
            logger.info(f"   ├─ 本次实验只使用任务 {self.task_id} 类别     ：{self.task_split[self.task_id]}")

        # ===== 其他参数输出（可选）=====
        logger.debug("🛠️ 所有训练参数 args（调试用）:")
        for k, v in vars(self.args).items():
            logger.debug(f"   - {k}: {v}")



    def train_onlyonetask(self, taskID):
        """这里是持续学习"""

        # for task_id, task in enumerate(self.task_split):
        logger.debug(f"只有一个任务 taskID :{taskID},self.args.num_train_epochs:{self.args.num_train_epochs},training_args.optimizer_type:{self.args.optimizer_type}")
        if taskID != -1:
            task_id = taskID
            lr = max(self.args.learning_rate *
                     (self.args.gamma ** task_id), 0.00005)
            
            for group in self.optimizer.param_groups:
                group["lr"] = lr

            train_loader = self.dataloaders[task_id]["train"]
            subset_loader = self.subset_dataloaders[task_id] if self.subset_dataloaders else None

            
            


            for epoch in tqdm(range(int(self.args.num_train_epochs)), desc=f"Task {task_id}"):

                logger.debug(f'train epoch: {epoch}')
                self.train_one_epoch(train_loader, task_id, subset_loader)

            if self.finetune_strategy == "ewc":
                loader = torch.utils.data.DataLoader(
                    train_loader.dataset, batch_size=200, shuffle=True)
                self.algo.update(self.model, task_id, loader)

            if self.args.save_task_models:
                logger.debug(f'✅ Finish Training, Save the model in {self.args.output_dir}')

                # 创建保存目录
                model_save_dir = os.path.join(self.args.output_dir, "models")
                os.makedirs(model_save_dir, exist_ok=True)

                # 构造模型保存文件名
                filename = f"task_{task_id}_method-{self.args.train_method}_opt-{self.args.optimizer_type}.pt"

                # 打包保存内容
                torch.save(
                    {
                        "model": self.model.state_dict(),
                        "classifier_out_dim": self.model.classifier.out_features,
                        "task_id": task_id,
                        "train_method": self.args.train_method,
                        "optimizer_type": self.args.optimizer_type,
                        "finetune_strategy": getattr(self, "finetune_strategy", None),
                        "classes": self.task_split[task_id],
                    },
                    os.path.join(model_save_dir, filename)
                )

                logger.info(f"✅ 模型已保存至: {os.path.join(model_save_dir, filename)}")

            if self.args.eval_after_train:
                logger.debug(f'Fininsh Training,Evaluate the model ,max_task_id={task_id + 1}')
                self.evaluate(max_task_id=task_id + 1)

    def train_cltask(self):

        logger.debug(f'Continual on all task:{self.task_split}')
        """这里是持续学习，在划分之后的所有task数据集上进行训练"""
        for task_id, task in enumerate(self.task_split):
            print(f"🎯 Task {task_id}: {task}")
            lr = max(self.args.learning_rate *
                     (self.args.gamma ** task_id), 0.00005)
            for group in self.optimizer.param_groups:
                group["lr"] = lr

            train_loader = self.dataloaders[task_id]["train"]
            subset_loader = self.subset_dataloaders[task_id] if self.subset_dataloaders else None

            for epoch in tqdm(range(int(self.args.num_train_epochs)), desc=f"Task {task_id}"):
                self.train_one_epoch(train_loader, task_id, subset_loader)

            if self.finetune_strategy == "ewc":
                loader = torch.utils.data.DataLoader(
                    train_loader.dataset, batch_size=200, shuffle=True)
                self.algo.update(self.model, task_id, loader)

            if self.args.save_task_models:
                os.makedirs(os.path.join(
                    self.args.output_dir, "models"), exist_ok=True)
                torch.save(
                    {"model": self.model.state_dict()},
                    os.path.join(self.args.output_dir, "models",
                                 f"task_{task_id}_model.pt"),
                )


            if self.args.eval_after_train:
                self.evaluate()

    def train_one_epoch(self, dataloader, task_id, subset_dataloader=None):

        logger.debug( f'train_one_epoch ,task_id:{task_id},self.args.train_method:{self.args.train_method} self.args.optimizer_type:{self.args.optimizer_type}')
        
        self.model.train()
        count = 0
        for X, y in iter(dataloader):
            count += 1
            self.model.zero_grad()
            if self.args.optimizer_type.lower() == "sam":
                enable_running_stats(self.model)

            X = X.to(self.device)
            y = y.to(self.device)
            out = self.model(X, task_id=task_id)
            if self.args.train_method == "er":
                if task_id > 0:
                    mem_x, mem_y, mem_task_ids = self.algo.sample(
                        self.batch_size, exclude_task=None, pr=False
                    )
                    mem_pred = self.model(mem_x, None)
                    mem_pred = extract_logits(
                        mem_pred, mem_task_ids, self.classes_per_task, self.device
                    )
                    loss_mem = self.criterion(mem_pred, mem_y)
                    loss_mem.backward()
                self.algo.add_reservoir(X, y, None, task_id)
            elif self.args.train_method == "ewc":
                loss_ewc = self.lambd * self.penalty(self.model)
                loss_ewc.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 100)
            elif self.args.train_method == "agem":
                self.model = self.algo.observe_agem(self.model, X, task_id, y)

            if self.args.train_method != "agem":
                loss = self.criterion(out, y)
                loss.backward()

            if self.args.optimizer_type.lower() == "sam":
                self.optimizer.first_step(zero_grad=True)
                disable_running_stats(self.model)
                self.criterion(self.model(X, task_id=task_id), y).backward()
                if self.args.train_method == "er" and task_id > 0:
                    mem_pred = self.model(mem_x, None)
                    mem_pred = extract_logits(
                        mem_pred, mem_task_ids, self.classes_per_task, self.device
                    )
                    self.criterion(mem_pred, mem_y).backward()
                self.optimizer.second_step(zero_grad=True)
            else:
                self.optimizer.step()


    def evaluate(self, max_task_id=None, logfile="log_eval_results.json"):
        """
        在 [0, max_task_id] 范围内的测试集上评估模型。
        
        参数：
            max_task_id: int，评估到第几个任务（包含该任务）。为 None 则评估全部任务。
            logfile: str，保存评估指标的 JSON 文件名。
        """
        logger.debug(f'evaluate, task_id:{max_task_id},save in {self.args.output_dir}')


        self.model.eval()
        full_metrics = {
            "accuracies": defaultdict(list),
            "losses": defaultdict(list),
        }

        task_average_accuracy = 0
        end_task_id = max_task_id if max_task_id is not None else len(self.task_split)

        with torch.no_grad():
            for eval_task_id in range(end_task_id):
                test_loader = self.dataloaders[eval_task_id]["test"]
                current_task_id =  eval_task_id

                total, correct, total_loss = 0, 0, 0.0
                for inputs, labels in test_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(inputs, task_id=current_task_id)
                    loss = self.criterion(outputs, labels)

                    preds = torch.argmax(outputs, dim=1)

                    total += labels.size(0)
                    correct += (preds == labels).sum().item()
                    total_loss += loss.item() * labels.size(0)

                accuracy = 100 * correct / total
                avg_loss = total_loss / total

                full_metrics["accuracies"][eval_task_id].append(accuracy)
                full_metrics["losses"][eval_task_id].append(avg_loss)
                task_average_accuracy += accuracy

                logger.info(
                    f"✅ Task {eval_task_id} - Accuracy: {accuracy:.2f}%, Loss: {avg_loss:.4f}"
                )

        # 计算整体指标
        average_accuracy, forgetting, learning_accuracy = self.calculate_run_metrics(
            full_metrics["accuracies"]
        )

        full_metrics["accuracies"] = dict(full_metrics["accuracies"])
        full_metrics["losses"] = dict(full_metrics["losses"])
        full_metrics["average_accuracy"] = average_accuracy
        full_metrics["forgetting"] = forgetting
        full_metrics["learning_accuracy"] = learning_accuracy

        os.makedirs(self.args.output_dir, exist_ok=True)

        # 构造带方法名的日志文件名
        filename = f"metrics_task{max_task_id}_{self.args.train_method}_opt-{self.args.optimizer_type}.json"
        save_path = os.path.join(self.args.output_dir, filename)

        with open(save_path, "w") as f:
            json.dump(full_metrics, f, indent=2)

        logger.info(f"📁 评估结果保存至: {save_path}")
        print(f"\n🎯 Final Average Accuracy: {average_accuracy:.2f}%")
        return full_metrics

    def compute_loss_landscape(
        self, eval_task_id: int, output_dir, save_file_name="lossLandscape",
        x_range=(-1, 1), y_range=(-1, 1), num_points=20, max_batches=5,
        sample_batches=False, flag_lora=True, flag_Nlora_full=False, flag_FullModel=False,
    ):
        args, device = self.args, self.device
        loss_grid = np.zeros((num_points, num_points))
        model = copy.deepcopy(self.model).to(device).eval()

        # 选择扰动参数
        original_params_to_perturb = {}
        for name, param in model.named_parameters():
            if "shared" not in name:
                original_params_to_perturb[name] = param.data.clone()
        surf_file = os.path.join(output_dir, f"{save_file_name}_task{eval_task_id}.h5")

        # 构建扰动方向
        torch.manual_seed(42)
        perturb_x, perturb_y = {}, {}
        for name in original_params_to_perturb:
            p = original_params_to_perturb[name]
            d_x = torch.randn_like(p)
            d_x = (d_x / d_x.norm()) * (p.norm() + 1e-8)
            d_y = torch.randn_like(p)
            d_y = d_y - torch.sum(d_y * d_x) * d_x / (d_x.norm()**2)
            d_y = (d_y / d_y.norm()) * (p.norm() + 1e-8)
            perturb_x[name] = d_x.to(device)
            perturb_y[name] = d_y.to(device)

        # 加载任务的测试数据
        test_loader = self.dataloaders[eval_task_id]["test"]
        all_batches = []
        for i, batch in enumerate(test_loader):
            if i >= max_batches:
                break
            all_batches.append(batch)

        # 坐标网格生成
        x_coords = np.linspace(x_range[0], x_range[1], num_points)
        y_coords = np.linspace(y_range[0], y_range[1], num_points)

        current_task_id =  eval_task_id

        for i in tqdm(range(num_points), desc=f"Grid X"):
            xv = x_coords[i]
            for j, yv in enumerate(y_coords):
                # 恢复原始参数
                for name in original_params_to_perturb:
                    model.state_dict()[name].copy_(original_params_to_perturb[name])

                # 添加扰动
                with torch.no_grad():
                    for name in original_params_to_perturb:
                        delta = xv * perturb_x[name] + yv * perturb_y[name]
                        model.state_dict()[name].add_(delta)

                # 计算该扰动点的平均损失
                total_loss = 0.0
                for inputs, labels in all_batches:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs, task_id=current_task_id)
                    loss = self.criterion(outputs, labels)
                    total_loss += loss.item()

                loss_grid[i, j] = total_loss / max(1, len(all_batches))

                if j % 10 == 0:
                    torch.cuda.empty_cache()

        # 保存结果
        os.makedirs(output_dir, exist_ok=True)

        filename = f"loss_surface_task{eval_task_id}_{self.args.train_method}_opt-{self.args.optimizer_type}.h5"
        surf_file = os.path.join(output_dir, filename)

        with h5py.File(surf_file, 'w') as f:
            f.create_dataset('xcoordinates', data=x_coords)
            f.create_dataset('ycoordinates', data=y_coords)
            f.create_dataset('train_loss', data=loss_grid)

        logger.info(f"✅ Loss surface for Task {eval_task_id} saved at: {surf_file}")

        return True


    def compute_loss_landscape_version2(
        self, eval_task_id: int, output_dir, save_file_name="lossLandscape2",
        num_samples=100, max_batches=5, sharpness_p=100, sharpness_alpha=1.0,
        ):
        args, device = self.args, self.device
        model = copy.deepcopy(self.model).to(device).eval()
        

        param_vector, param_shapes, param_names = [], [], []
        for name, param in model.named_parameters():
            if "shared" not in name:
                param_vector.append(param.data.view(-1))
                param_shapes.append(param.shape)
                param_names.append(name)
        param_vector = torch.cat(param_vector)
        num_parameters = param_vector.numel()

        def compute_loss(model):
            total_loss = 0.0
            test_loader = self.dataloaders[eval_task_id]["test"]

            for i, batch in enumerate(test_loader):
                if i >= max_batches:
                    break

                X, y = batch
                X = X.to(device)
                y = y.to(device)

                out = model(X, task_id=eval_task_id)
                # logger.debug(f'计算损失，查看样本预测结果，真实标签：{y},预测结果{out}')

                if self.args.train_method == "er" and eval_task_id > 0:
                    mem_x, mem_y, mem_task_ids = self.algo.sample(
                        self.batch_size, exclude_task=None, pr=False
                    )
                    mem_pred = model(mem_x.to(device), None)
                    mem_pred = extract_logits(
                        mem_pred, mem_task_ids, self.classes_per_task, self.device
                    )
                    loss_mem = self.criterion(mem_pred, mem_y.to(device))
                    loss_er = self.criterion(out, y)
                    loss = loss_er + loss_mem  # 加权可以根据需要修改
                else:
                    loss = self.criterion(out, y)

                total_loss += loss.item()

            return total_loss / max(1, i + 1)

        L_w = compute_loss(model)
        logger.info(f"🎯 Baseline loss (task {eval_task_id}): {L_w:.4f}")

        A = torch.randn(num_parameters, sharpness_p, device=device)
        A /= torch.norm(A, dim=0, keepdim=True)
        loss_list, L_max = [], L_w

        for _ in tqdm(range(num_samples), desc="Sharpness Samples"):
            z = torch.randn(sharpness_p, device=device)
            z = sharpness_alpha * z / (torch.norm(z) + 1e-8)
            perturbed_w = A @ z

            idx = 0
            for i, name in enumerate(param_names):
                param = model.state_dict()[name]
                numel = param_shapes[i].numel()
                delta = perturbed_w[idx: idx + numel].reshape(param_shapes[i])
                param.add_(delta)
                idx += numel

            perturbed_loss = compute_loss(model)
            loss_list.append(perturbed_loss)
            L_max = max(L_max, perturbed_loss)
            torch.cuda.empty_cache()

        sharpness_metric = ((L_max - L_w) / (1 + L_w)) * 100
        logger.info(f"📈 Sharpness (task {eval_task_id}): {sharpness_metric:.4f}")

        
        os.makedirs(output_dir, exist_ok=True)
        filename = f"sharpness_task{eval_task_id}_{self.args.train_method}_opt-{self.args.optimizer_type}.h5"
        surf_file = os.path.join(output_dir, filename)

        with h5py.File(surf_file, 'w') as f:
            f.create_dataset('L_w', data=L_w)
            f.create_dataset('L_max', data=L_max)
            f.create_dataset('sharpness_metric', data=sharpness_metric)
            f.create_dataset('loss_list', data=np.array(loss_list))

        logger.info(f"📁 Sharpness landscape 保存成功到 {surf_file}")

        return sharpness_metric



    #------------------复制实现另一种效果-----------------


    def run_sharpness_eval(self, model_checkpoints, dataloaders, p=100):
        """
        对前若干个模型 checkpoint 执行 sharpness 计算。
        参数：
            model_checkpoints: dict[int, str]，每个任务的 checkpoint 路径
            dataloaders: dict[int, dict[str, DataLoader]]，每个任务的 dataloader
            p: 投影维度
        """
        device = self.device
        model_list = {i: model_checkpoints[i] for i in range(min(5, len(model_checkpoints)))}
        num_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        A = 1 if p == 0 else self._get_projection_matrix(num_parameters, p)

        results = defaultdict(dict)
        for model_idx in model_list:
            task_idx = model_idx
            self._load_checkpoint(self.model, model_list[model_idx])
            sharpness_fn = self.create_sharpness_fn(dataloaders[task_idx]["test"], task_idx, device)
            for epsilon in [1e-3, 5e-4]:
                s = self._sharpness_core(self.model, sharpness_fn, A, epsilon, p)
                logger.info(f"🔍 Model {model_idx} ε={epsilon} Sharpness: {s:.4f}")
                results[model_idx][epsilon] = s
        return dict(results)







    def _sharpness_core(self, model, criterion_fn, A, epsilon=1e-3, p=0, bounds=None):
        model = copy.deepcopy(model)
        run_fn = self._create_run_model(model, A, criterion_fn)
        if bounds is None:
            bounds = self._compute_bounds(model, A, epsilon)
        dim = sum(p.numel() for p in model.parameters()) if p == 0 else p
        opt_result = optimize.minimize(
            lambda x: run_fn(x),
            np.zeros(dim),
            method="L-BFGS-B",
            bounds=bounds,
            jac=True,
            options={"maxiter": 10},
        )
        flat_diffs = opt_result.x if A == 1 else A @ opt_result.x
        model_copy = copy.deepcopy(model)
        self._apply_diffs(model_copy, flat_diffs)
        L_max = criterion_fn(model_copy)["loss"]
        L_orig = criterion_fn(model)["loss"]
        return 100 * (L_max - L_orig) / (1 + L_orig)

    def _get_projection_matrix(self, num_parameters, p):
        A = np.random.randn(num_parameters, p).astype(np.float32)
        A /= np.linalg.norm(A, axis=0, keepdims=True)
        return A

    def _compute_bounds(self, model, A, epsilon):
        dim = sum(p.numel() for p in model.parameters()) if isinstance(A, int) else A.shape[1]
        return [(-epsilon, epsilon) for _ in range(dim)]

    def _create_run_model(self, model, A, criterion_fn):
        base_params = self.flatten_parameters(model).cpu().numpy()
        def run_fn(delta):
            delta = delta.astype(np.float32)
            diff = delta if A == 1 else A @ delta
            perturbed = base_params + diff
            model_copy = copy.deepcopy(model)
            self._assign_flat_parameters(model_copy, perturbed)
            metrics = criterion_fn(model_copy)
            return metrics["loss"], metrics["gradients"].cpu().numpy()
        return run_fn

    def flatten_parameters(self, model):
        return torch.cat([p.view(-1) for p in model.parameters() if p.requires_grad])

    def flatten_gradients(self, model):
        return torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])

    def _assign_flat_parameters(self, model, flat_tensor):
        pointer = 0
        for p in model.parameters():
            if not p.requires_grad:
                continue
            numel = p.numel()
            p.data = flat_tensor[pointer:pointer + numel].view_as(p).clone().to(p.device)
            pointer += numel

    def _apply_diffs(self, model, diff_tensor):
        flat = self.flatten_parameters(model).cpu().numpy()
        updated = flat + diff_tensor
        self._assign_flat_parameters(model, updated)



    def create_eval_fn(self, task_id, calculate_gradient=False):
        def eval_fn(model, dataloader, device):
            model.eval()
            total_loss = 0
            loss_fn = torch.nn.CrossEntropyLoss(reduction="sum").to(device)
            num_correct = 0
            model.zero_grad()
            torch.set_grad_enabled(calculate_gradient)
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                output = model(X, task_id)
                preds = torch.argmax(output, dim=1)
                num_correct += (preds == y).sum().item()
                loss = loss_fn(output, y) / len(dataloader.dataset)
                if calculate_gradient:
                    loss.backward()
                total_loss += loss.item()
            accuracy = num_correct / len(dataloader.dataset)
            metrics = {"loss": total_loss, "accuracy": accuracy}
            if calculate_gradient:
                metrics["gradients"] = self.flatten_gradients(model)
            return metrics
        return eval_fn

    def create_sharpness_fn(self, dataloader, task_id, device):
        full_loss_fn = self.create_eval_fn(task_id, calculate_gradient=True)
        return lambda model: full_loss_fn(model, dataloader, device)



