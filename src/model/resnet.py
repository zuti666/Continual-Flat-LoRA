import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional, List, Tuple, Union
from transformers.models.resnet.modeling_resnet import ResNetForImageClassification
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.modeling_outputs import (
    BackboneOutput,
    BaseModelOutputWithNoAttention,
    BaseModelOutputWithPoolingAndNoAttention,
    ImageClassifierOutputWithNoAttention,
)

class ResNetForImageClassification_WithLossMask(ResNetForImageClassification):
    def __init__(self, config):
        """
        初始化 ResNetForImageClassificationWithLossMask 继承自 ResNetForImageClassification。

        Args:
            config: `transformers` 预训练 ResNet 模型的配置对象 (`ResNetConfig`)。
        """
        super().__init__(config)
        self.num_labels = config.num_labels  # 任务类别数
        self.resnet = self.resnet  # 继承 `ResNetForImageClassification` 的主干网络
        self.classifier = self.classifier  # 继承 `classifier` 头部

        # 兼容 `transformers` 预训练模型
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        loss_mask: Optional[torch.Tensor] = None,
    ) :
        """
        自定义 forward 方法，增加 `loss_mask`，用于控制哪些样本计算 loss。

        Args:
            pixel_values (torch.FloatTensor): 输入图像 (batch_size, 3, H, W)。
            labels (torch.LongTensor, optional): 目标标签 (batch_size,)。
            output_hidden_states (bool, optional): 是否返回隐藏状态。
            return_dict (bool, optional): 是否以字典形式返回结果。
            loss_mask (torch.Tensor, optional): 形状 `(batch_size,)`，值为 0 的样本不计算 loss。

        Returns:
            ImageClassifierOutputWithNoAttention: 包含 logits 和 loss 的输出。
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.resnet(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)

        pooled_output = outputs.pooler_output if return_dict else outputs[1]
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            # 处理不同类型的分类任务
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
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output

        return ImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )

