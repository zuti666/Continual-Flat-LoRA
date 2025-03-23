import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.vit.modeling_vit import ViTForImageClassification, ViTPreTrainedModel
from transformers.modeling_outputs import ImageClassifierOutput
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from typing import Optional, Union


class ViTForImageClassification_WithLossMask(ViTForImageClassification):
    def __init__(self, config):
        """
        初始化 ViTForImageClassificationWithLossMask。

        Args:
            config: `transformers` 预训练 ViT 模型的配置对象 (`ViTConfig`)。
        """
        super().__init__(config)
        self.num_labels = config.num_labels  # 任务类别数
        self.vit = self.vit  # 继承 `ViTForImageClassification` 的主干网络
        self.classifier = self.classifier  # 继承 `classifier` 头部

        # 兼容 `transformers` 预训练模型
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        loss_mask: Optional[torch.Tensor] = None,  # 新增 `loss_mask`
        task_id: Optional[int] = None,  # ✅ 添加 task_id 参数
    ) -> Union[tuple, ImageClassifierOutput]:
        """
        自定义 forward 方法，增加 `loss_mask`，用于控制哪些样本计算 loss。

        Args:
            pixel_values (torch.Tensor): 输入图像 (batch_size, 3, H, W)。
            head_mask (torch.Tensor, optional): 头部 mask。
            labels (torch.Tensor, optional): 目标标签 (batch_size,)。
            output_attentions (bool, optional): 是否返回注意力权重。
            output_hidden_states (bool, optional): 是否返回隐藏状态。
            interpolate_pos_encoding (bool, optional): 是否对位置编码进行插值。
            return_dict (bool, optional): 是否以字典形式返回结果。
            loss_mask (torch.Tensor, optional): 形状 `(batch_size,)`，值为 0 的样本不计算 loss。

        Returns:
            ImageClassifierOutput: 包含 logits 和 loss 的输出。
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.classifier(sequence_output[:, 0, :])


         # 🧠 SAM 优化器 compatibility: 如果没传 labels，直接返回 logits
        if labels is None:
            return logits

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and labels.dtype in (torch.long, torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss(reduction='none')  # `reduction='none'` 以支持 `loss_mask`
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss(reduction='none')
                loss = loss_fct(logits, labels)

            # 应用 loss_mask，仅计算部分样本的 loss
            if loss_mask is not None:
                loss = loss * loss_mask.view(-1)  # 仅保留 `loss_mask` 位置的损失
                loss = loss.sum() / loss_mask.sum()  # 归一化处理
            else:
                loss = loss.mean()  # 默认计算均值 loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
