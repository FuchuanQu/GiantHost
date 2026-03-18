"""
Multi-Task Learning Model for Viral Contig Classification
多任务学习模型：基于 DualMLP backbone

Tasks:
- Task 1: Virus Taxonomy Classification (virus_order)
  - 支持半监督学习：使用 train + no_label 数据
- Task 2: MoE-style Two-Stage Host Classification
  - Stage 1: Coarse host classification (host_group_1): Vertebrata, Invertebrata, Protozoa, Algae
  - Stage 2: Fine-grained host classification with expert heads (host_group_2)
  - 仅使用有宿主标签的 train 数据

Architecture:
- Shared dual-tower backbone for dense (kmer+traits) and sparse (GVOG) features
- Task-specific heads with learnable routing mechanism

Training Strategy:
- Semi-supervised multi-task learning
- Virus taxonomy: trained on all data (train + no_label MAGs with virus_order)
- Host classification: trained only on labeled data (train)
- Teacher forcing for expert selection during training
- Soft/hard routing during inference
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.preprocessing import StandardScaler
from src.core.base_model import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SharedBackbone(nn.Module):
    """
    Shared dual-tower backbone for processing heterogeneous features.
    
    Tower A: Dense features (K-mer + Gene Content Traits)
        - BatchNorm for input normalization
        - Lower dropout (0.1) as features are dense
    
    Tower B: Sparse features (GVOG one-hot)
        - Higher dropout (0.5) to prevent overfitting
        - No BatchNorm as input is sparse
    """
    
    def __init__(self, dense_input_dim, sparse_input_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3,
                 dropout_dense: float = 0.1, dropout_sparse: float = 0.5):
        super(SharedBackbone, self).__init__()
        
        # --- Tower A: Dense features (K-mer + Traits) ---
        self.tower_dense = nn.Sequential(
            nn.Linear(dense_input_dim, hidden_dim_1),
            nn.BatchNorm1d(hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout_dense),

            nn.Linear(hidden_dim_1, hidden_dim_1),
            nn.BatchNorm1d(hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout_dense),

            nn.Linear(hidden_dim_1, hidden_dim_3),
            nn.ReLU()
        )
        
        # --- Tower B: Sparse features (GVOG) ---
        self.tower_sparse = nn.Sequential(
            nn.Linear(sparse_input_dim, hidden_dim_2),
            nn.BatchNorm1d(hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout_sparse),
            
            # nn.Linear(hidden_dim_2, hidden_dim_2),
            # nn.BatchNorm1d(hidden_dim_2),
            # nn.ReLU(),
            # nn.Dropout(dropout_sparse),
            
            nn.Linear(hidden_dim_2, hidden_dim_3),
            nn.ReLU()
        )
        
        self.combine = nn.Sequential(
            nn.Linear(hidden_dim_3 * 2, hidden_dim_3 * 2),
            nn.BatchNorm1d(hidden_dim_3 * 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.output_dim = hidden_dim_3 * 2
        
    def forward(self, x_dense, x_sparse):
        out_dense = self.tower_dense(x_dense)
        out_sparse = self.tower_sparse(x_sparse)
        combined_tensor = torch.cat((out_dense, out_sparse), dim=1)
        return self.combine(combined_tensor)


class VirusTaxonomyHead(nn.Module):
    """病毒分类学任务头 (virus_order)"""
    
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(VirusTaxonomyHead, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        return self.classifier(x)


class HostGroupRouter(nn.Module):
    """第一阶段路由器：将样本分配到 4 个宿主大类"""
    
    def __init__(self, input_dim, hidden_dim, num_groups=4):
        super(HostGroupRouter, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_groups)
        )
        self.num_groups = num_groups
        
    def forward(self, x):
        return self.classifier(x)


class HostExpertHead(nn.Module):
    """每个宿主大类对应的专家分类头 (host_group_2)"""
    
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(HostExpertHead, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        return self.classifier(x)


class GradientReversalFunction(torch.autograd.Function):
    """Standard GRL used in DANN: identity forward, -lambda * grad in backward."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    """nn.Module wrapper for GradientReversalFunction."""

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = float(lambda_)

    def set_lambda(self, lambda_: float):
        self.lambda_ = float(lambda_)

    def forward(self, x, lambda_: Optional[float] = None):
        if lambda_ is None:
            lambda_ = self.lambda_
        return GradientReversalFunction.apply(x, float(lambda_))


class DomainHead(nn.Module):
    """Domain discriminator head for DANN (host_label vs no_label)."""

    def __init__(self, input_dim, hidden_dim, num_domains=2, grl_lambda: float = 1.0):
        super().__init__()
        self.grl = GradientReversalLayer(lambda_=grl_lambda)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_domains),
        )

    def forward(self, x, grl_lambda: Optional[float] = None):
        x = self.grl(x, grl_lambda)
        return self.classifier(x)

class GradientScaler(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None

def scale_gradient(x, scale):
    return GradientScaler.apply(x, scale)

class MoEHostClassifier(nn.Module):
    """
    类似 MoE 的两阶段宿主分类器
    - 阶段1: 路由器预测 host_group_1
    - 阶段2: 根据 host_group_1 使用对应的专家头预测 host_group_2
    """
    
    def __init__(self, input_dim, hidden_dim, num_groups, num_classes_per_group, inference_routing="hard"):
        """
        Args:
            input_dim: backbone 输出维度
            hidden_dim: 隐藏层维度
            num_groups: host_group_1 的类别数 (默认 4)
            num_classes_per_group: dict, 每个 host_group_1 对应的 host_group_2 类别数
                                   e.g., {'Vertebrata': 5, 'Invertebrata': 3, ...}
            inference_routing: "soft" 或 "hard"。soft 模式下所有专家输出统一维度
        """
        super(MoEHostClassifier, self).__init__()
        
        self.num_groups = num_groups
        self.num_classes_per_group = num_classes_per_group
        self.inference_routing = inference_routing
        
        # 第一阶段: 路由器
        self.router = HostGroupRouter(input_dim, hidden_dim, num_groups)
        
        # 第二阶段: 每个 group 一个专家头
        self.experts = nn.ModuleDict()
        if inference_routing == "soft":
            # Soft routing: 所有专家输出统一维度 (全部 host_group_2 类别)
            total_classes = sum(num_classes_per_group.values())
            self.total_expert_classes = total_classes
            for group_name in num_classes_per_group:
                self.experts[group_name] = HostExpertHead(input_dim, hidden_dim, total_classes)
            logger.info(f"    [Soft Routing] All experts output unified dim = {total_classes}")
        else:
            # Hard routing: 每个专家输出各自 group 的类别数
            self.total_expert_classes = None
            for group_name, num_classes in num_classes_per_group.items():
                self.experts[group_name] = HostExpertHead(input_dim, hidden_dim, num_classes)
        
        # 存储 group 名称到索引的映射
        self.group_names = list(num_classes_per_group.keys())
        
    def forward(self, x, group_labels=None, return_all_experts=False):
        """
        Args:
            x: backbone 特征 [batch_size, input_dim]
            group_labels: host_group_1 标签 (训练时使用真实标签)
            return_all_experts: 是否返回所有专家的输出 (用于推理)
            
        Returns:
            router_logits: 第一阶段路由器输出 [batch_size, num_groups]
            expert_outputs: dict of expert outputs, 或者 aggregated output
        """
        # 第一阶段: 路由预测
        router_logits = self.router(x)
        
        # # TODO : 根据需要实现软/硬路由
        # if return_all_experts:
        #     # 推理时，计算所有专家的输出
        #     expert_outputs = {}
        #     for group_name in self.group_names:
        #         expert_outputs[group_name] = self.experts[group_name](x)
        #     return router_logits, expert_outputs
        
        # 训练时，根据 group_labels 选择对应的专家
        # TODO: 梯度停止
        feat_scaled = scale_gradient(x, 0.2)
        expert_outputs = {}
        for group_name in self.group_names:
            expert_outputs[group_name] = self.experts[group_name](feat_scaled)
        
        return router_logits, expert_outputs


class MultiTaskViralNet(nn.Module):
    """多任务病毒分类网络"""
    
    def __init__(
        self,
        dense_input_dim,
        sparse_input_dim,
        hidden_dim_1,
        hidden_dim_2,
        hidden_dim_3,
        num_virus_orders,
        num_host_groups,
        num_classes_per_group,
        enable_virus_taxonomy: bool = True,
        inference_routing: str = "hard",
        enable_domain_adversarial: bool = True,
        domain_hidden_dim: Optional[int] = None,
        domain_grl_lambda: float = 1.0,
    ):
        super(MultiTaskViralNet, self).__init__()
        
        self.enable_virus_taxonomy = enable_virus_taxonomy
        
        # 共享 backbone
        self.backbone = SharedBackbone(
            dense_input_dim, sparse_input_dim,
            hidden_dim_1, hidden_dim_2, hidden_dim_3
        )
        
        backbone_out_dim = self.backbone.output_dim
        task_hidden_dim = hidden_dim_3
        
        # 任务1: 病毒分类学 (virus_order)
        self.virus_taxonomy_head = None
        if self.enable_virus_taxonomy:
            self.virus_taxonomy_head = VirusTaxonomyHead(
                backbone_out_dim, task_hidden_dim, num_virus_orders
            )
        
        # 任务2: MoE 宿主分类
        self.host_moe_classifier = MoEHostClassifier(
            backbone_out_dim, task_hidden_dim,
            num_host_groups, num_classes_per_group,
            inference_routing=inference_routing
        )

        # 域判别头: host_label(1) vs no_label(0)
        self.enable_domain_adversarial = enable_domain_adversarial
        self.domain_head = None
        if self.enable_domain_adversarial:
            self.domain_head = DomainHead(
                input_dim=backbone_out_dim,
                hidden_dim=domain_hidden_dim or task_hidden_dim,
                num_domains=2,
                grl_lambda=domain_grl_lambda,
            )
        
    def forward(
        self,
        x_dense,
        x_sparse,
        host_group_labels=None,
        return_all_experts=False,
        grl_lambda: Optional[float] = None,
        return_features: bool = False,
    ):
        """
        Args:
            x_dense: 稠密特征
            x_sparse: 稀疏特征
            host_group_labels: host_group_1 标签 (训练时)
            return_all_experts: 推理时返回所有专家输出
            
        Returns:
            virus_logits: 病毒分类学 logits
            host_router_logits: 宿主大类 logits
            host_expert_outputs: 宿主细分类 outputs
        """
        # 提取共享特征
        shared_features = self.backbone(x_dense, x_sparse)
        
        # 任务1: 病毒分类
        virus_logits = None
        if self.enable_virus_taxonomy and self.virus_taxonomy_head is not None:
            virus_logits = self.virus_taxonomy_head(shared_features)
        
        # 任务2: MoE 宿主分类
        host_router_logits, host_expert_outputs = self.host_moe_classifier(
            shared_features, host_group_labels, return_all_experts
        )
        
        domain_logits = None
        if self.enable_domain_adversarial and self.domain_head is not None:
            domain_logits = self.domain_head(shared_features, grl_lambda=grl_lambda)

        if return_features:
            return virus_logits, host_router_logits, host_expert_outputs, domain_logits, shared_features

        return virus_logits, host_router_logits, host_expert_outputs, domain_logits


class MultiTaskMLPModel(BaseModel):
    """多任务学习模型封装"""
    
    def __init__(self, config):
        super(MultiTaskMLPModel, self).__init__(config)
        
        # 从 config 读取 host classification 层级配置 (不再硬编码)
        host_cfg = config.get('multi_task', {}).get('tasks', {}).get('host_classification', {})
        self.stage_1_classes = list(host_cfg.get('stage_1', {}).get('classes', []))
        self.stage_1_col = host_cfg.get('stage_1', {}).get('label_column', 'host_group_1')
        self.stage_2_col = host_cfg.get('stage_2', {}).get('label_column', 'host_group_2')
        
        # 从 config 获取参数
        self.input_dim1 = config['model']['params']['input_dim1']
        self.input_dim2 = config['model']['params']['input_dim2']
        self.hidden_dim1 = config['model']['params']['hidden_dim1']
        self.hidden_dim2 = config['model']['params']['hidden_dim2']
        self.hidden_dim3 = config['model']['params']['hidden_dim3']
        
        # 任务相关参数
        self.enable_virus_taxonomy = config.get('multi_task', {}).get('tasks', {}).get('virus_taxonomy', {}).get('enable', True)
        self.virus_taxonomy_label_column = config.get('multi_task', {}).get('tasks', {}).get('virus_taxonomy', {}).get('label_column', 'virus_order')
        self.num_virus_orders = config['model']['params'].get('num_virus_orders', 0)
        if not self.enable_virus_taxonomy:
            # 当禁用 taxonomy 任务时，不强制配置 num_virus_orders
            self.num_virus_orders = self.num_virus_orders or 0
        self.num_host_groups = config['model']['params'].get('num_host_groups', len(self.stage_1_classes))
        self.num_classes_per_group = config['model']['params']['num_classes_per_group']
        
        # 训练参数
        self.learning_rate = config['model']['params']['learning_rate']
        self.weight_decay = config['model']['params'].get('weight_decay', 0.0)
        self.num_epochs = config['model']['params']['num_epochs']
        self.batch_size = config['model']['params']['batch_size']

        # Domain-adversarial (DANN) 参数
        self.enable_domain_adversarial = config['model']['params'].get('enable_domain_adversarial', True)
        self.domain_loss_weight = float(config['model']['params'].get('domain_loss_weight', 1.0))
        self.domain_hidden_dim = int(config['model']['params'].get('domain_hidden_dim', self.hidden_dim3))
        self.domain_grl_lambda_max = float(config['model']['params'].get('domain_grl_lambda_max', 1.0))
        self.domain_grl_gamma = float(config['model']['params'].get('domain_grl_gamma', 10.0))
        
        # 任务权重 (支持动态权重调度)
        # 默认使用线性递减: 初始 (0.5, 0.5, 0) -> 最终 (0, 0, 1)
        self.task_weights_start = config['model']['params'].get('task_weights_start', {
            'virus_taxonomy': 0.3,
            'host_group_1': 0.6,
            'host_group_2': 0.1
        })
        self.task_weights_end = config['model']['params'].get('task_weights_end', {
            'virus_taxonomy': 0.1,
            'host_group_1': 0.45,
            'host_group_2': 0.45
        })
        # 兼容旧配置: 如果设置了固定 task_weights，则不使用动态调度
        self.use_dynamic_weights = config['model']['params'].get('use_dynamic_weights', True)
        if not self.use_dynamic_weights:
            self.task_weights = config['model']['params'].get('task_weights', {
                'virus_taxonomy': 0,
                'host_group_1': 0.7,
                'host_group_2': 0.3
            })
        # 如果禁用 taxonomy 任务，将其权重强制为 0
        if not self.enable_virus_taxonomy:
            self.task_weights_start['virus_taxonomy'] = 0.0
            self.task_weights_end['virus_taxonomy'] = 0.0
            if not self.use_dynamic_weights:
                self.task_weights['virus_taxonomy'] = 0.0
        
        # 路由策略
        self.inference_routing = config['model']['params'].get('inference_routing', 'soft')
        
        # 构建模型
        self.model = MultiTaskViralNet(
            self.input_dim1, self.input_dim2,
            self.hidden_dim1, self.hidden_dim2, self.hidden_dim3,
            self.num_virus_orders,
            self.num_host_groups,
            self.num_classes_per_group,
            enable_virus_taxonomy=self.enable_virus_taxonomy,
            inference_routing=self.inference_routing,
            enable_domain_adversarial=self.enable_domain_adversarial,
            domain_hidden_dim=self.domain_hidden_dim,
            domain_grl_lambda=self.domain_grl_lambda_max,
        ).to(self.device)
        
        # 损失函数 (使用 ignore_index=-1 自动忽略无标签样本)
        self.criterion_virus = nn.CrossEntropyLoss(ignore_index=-1)
        self.criterion_host_router = nn.CrossEntropyLoss(ignore_index=-1)
        self.criterion_host_expert = nn.CrossEntropyLoss(ignore_index=-1)
        self.criterion_host_experts = {}  # per-expert weighted losses (populated by set_class_weights)
        self.criterion_domain = nn.CrossEntropyLoss()
        self._class_weights = None
        
        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Label encoders (将在训练时设置，或通过 set_label_encoders 预先设置)
        self.virus_order_encoder = None
        self.host_group_1_encoder = None
        self.host_group_2_encoders = {}  # 每个 group 一个 encoder
        self.unified_host_g2_encoder = None  # soft routing 统一编码器
        self._encoders_preset = False    # 标记是否已预设 encoders
    
    def _get_task_weights(self, epoch: int) -> Dict[str, float]:
        """
        获取当前 epoch 的任务权重
        
        支持动态权重调度：线性从 task_weights_start 过渡到 task_weights_end
        
        Args:
            epoch: 当前训练轮次
            
        Returns:
            dict with current weights for each task
        """
        if not self.use_dynamic_weights:
            return self.task_weights
        
        # 线性插值: weight = start + (end - start) * progress
        # progress: 0 -> 1 (从第一个 epoch 到最后一个 epoch)
        if self.num_epochs <= 1:
            progress = 1.0
        else:
            progress = epoch / (self.num_epochs - 1)
        
        current_weights = {}
        for key in self.task_weights_start:
            start_w = self.task_weights_start[key]
            end_w = self.task_weights_end[key]
            current_weights[key] = start_w + (end_w - start_w) * progress
        
        return current_weights

    def _get_dann_lambda(self, epoch: int, batch_idx: int, total_batches: int) -> float:
        """DANN schedule: lambda = max_lambda * (2 / (1 + exp(-gamma * p)) - 1)."""
        total_steps = max(self.num_epochs * max(total_batches, 1), 1)
        current_step = epoch * max(total_batches, 1) + batch_idx
        p = current_step / max(total_steps - 1, 1)
        lambda_base = 2.0 / (1.0 + np.exp(-self.domain_grl_gamma * p)) - 1.0
        return float(self.domain_grl_lambda_max * lambda_base)
    
    def set_label_encoders(self, encoders: Dict):
        """
        设置预先初始化的 label encoders（用于优化 LOOCV 性能）
        
        Args:
            encoders: dict with:
                - 'virus_order': LabelEncoder for virus taxonomy
                - 'host_group_1': LabelEncoder for host group 1
                - 'host_group_2': Dict[group_name -> LabelEncoder] for each host_group_1
        """
        self.virus_order_encoder = encoders.get('virus_order')
        self.host_group_1_encoder = encoders.get('host_group_1')
        self.host_group_2_encoders = encoders.get('host_group_2', {})
        self._encoders_preset = True
        logger.debug("    [Optimization] Using pre-initialized label encoders")

    def set_class_weights(self, class_weights: Dict):
        """
        设置类别权重并重建带权重的损失函数
        
        Args:
            class_weights: dict with:
                - 'host_group_1': {class_name: weight_value, ...}
                - 'host_group_2': {g1_class_name: {g2_class_name: weight_value, ...}, ...}
        """
        self._class_weights = class_weights
        
        # ---- host_group_1 router loss ----
        g1_weights = class_weights.get('host_group_1', {})
        if g1_weights and self.host_group_1_encoder is not None:
            weight_tensor = torch.ones(len(self.host_group_1_encoder.classes_), device=self.device)
            for i, cls_name in enumerate(self.host_group_1_encoder.classes_):
                if cls_name in g1_weights:
                    weight_tensor[i] = g1_weights[cls_name]
            self.criterion_host_router = nn.CrossEntropyLoss(weight=weight_tensor, ignore_index=-1)
            logger.info(f"    [ClassWeight] host_group_1 weights: {dict(zip(self.host_group_1_encoder.classes_, weight_tensor.cpu().tolist()))}")
        
        # ---- per-expert host_group_2 losses ----
        g2_weights = class_weights.get('host_group_2', {})
        self.criterion_host_experts = {}  # group_name -> weighted CrossEntropyLoss
        if g2_weights:
            if self.inference_routing == "soft" and self.unified_host_g2_encoder is not None:
                # Soft routing: 单一统一权重的 loss
                all_g2_w = {}
                for group_w in g2_weights.values():
                    all_g2_w.update(group_w)
                weight_tensor = torch.ones(len(self.unified_host_g2_encoder.classes_), device=self.device)
                for i, cls_name in enumerate(self.unified_host_g2_encoder.classes_):
                    if cls_name in all_g2_w:
                        weight_tensor[i] = all_g2_w[cls_name]
                self.criterion_host_expert = nn.CrossEntropyLoss(weight=weight_tensor, ignore_index=-1)
                logger.info(f"    [ClassWeight] host_group_2 unified weights: {dict(zip(self.unified_host_g2_encoder.classes_, weight_tensor.cpu().tolist()))}")
            else:
                # Hard routing: per-expert weighted losses
                for group_name, group_w in g2_weights.items():
                    if group_name in self.host_group_2_encoders:
                        encoder = self.host_group_2_encoders[group_name]
                        weight_tensor = torch.ones(len(encoder.classes_), device=self.device)
                        for i, cls_name in enumerate(encoder.classes_):
                            if cls_name in group_w:
                                weight_tensor[i] = group_w[cls_name]
                        self.criterion_host_experts[group_name] = nn.CrossEntropyLoss(
                            weight=weight_tensor, ignore_index=-1
                        )
                        logger.info(f"    [ClassWeight] host_group_2[{group_name}] weights: {dict(zip(encoder.classes_, weight_tensor.cpu().tolist()))}")
        
        logger.info(f"    [ClassWeight] Class weights applied to loss functions")
        
    def _setup_label_encoders(self, y_virus_taxonomy: np.ndarray, y_host: Optional[Dict[str, np.ndarray]] = None):
        """
        设置标签编码器
        
        如果已通过 set_label_encoders 预设，则跳过此步骤。
        
        Args:
            y_virus_taxonomy: 所有样本的 virus taxonomy 标签 (train + no_label)
                              可能是 virus_order 或 virus_family
                              可能包含 None、NaN 或 'Unknown' 值，会被过滤掉
            y_host: 有宿主标签的数据，包含 'host_group_1' 和 'host_group_2'
                   如果为 None，则不设置 host encoders
        """
        # 如果已预设 encoders，直接返回
        if self._encoders_preset and (self.virus_order_encoder is not None or not self.enable_virus_taxonomy):
            logger.debug("    [Optimization] Skipping label encoder setup (already preset)")
            # Soft routing: 构建统一编码器
            if self.inference_routing == "soft" and self.host_group_2_encoders and self.unified_host_g2_encoder is None:
                self._build_unified_host_g2_encoder()
            return
        
        from sklearn.preprocessing import LabelEncoder
        
        # 定义无效标签值
        invalid_labels = {'Unknown', 'unknown', 'UNKNOWN', '', 'NA', 'N/A', 'nan'}
        
        # virus taxonomy encoder (仅使用有效标签，排除 None/NaN/空字符串/Unknown)
        if self.enable_virus_taxonomy:
            self.virus_order_encoder = LabelEncoder()  # 保留变量名以兼容
            valid_virus_labels = [
                v for v in y_virus_taxonomy 
                if v is not None and pd.notna(v) and str(v).strip() not in invalid_labels
            ]
            if len(valid_virus_labels) == 0:
                raise ValueError("No valid virus taxonomy labels found for training")
            self.virus_order_encoder.fit(valid_virus_labels)
            logger.info(f"    Virus taxonomy classes ({len(self.virus_order_encoder.classes_)}): {list(self.virus_order_encoder.classes_)}")
        else:
            self.virus_order_encoder = None
        
        # host encoders (仅使用有标签数据)
        if y_host is not None:
            # host_group_1 encoder (使用 config 中定义的类别顺序)
            self.host_group_1_encoder = LabelEncoder()
            self.host_group_1_encoder.fit(self.stage_1_classes)
            
            # 为每个 host_group_1 创建 host_group_2 encoder
            self.host_group_2_encoders = {}
            for group_name in self.stage_1_classes:
                mask = y_host['host_group_1'] == group_name
                if mask.any():
                    unique_classes = np.unique(y_host['host_group_2'][mask])
                    encoder = LabelEncoder()
                    encoder.fit(unique_classes)
                    self.host_group_2_encoders[group_name] = encoder
                    logger.info(f"    Host group {group_name} classes ({len(unique_classes)}): {list(unique_classes)}")
            
            # Soft routing: 构建统一 host_group_2 编码器
            if self.inference_routing == "soft" and self.host_group_2_encoders:
                self._build_unified_host_g2_encoder()
        
    def _build_unified_host_g2_encoder(self):
        """构建统一的 host_group_2 编码器 (用于 soft routing)"""
        from sklearn.preprocessing import LabelEncoder
        all_g2_classes = []
        for group_name in self.stage_1_classes:
            if group_name in self.host_group_2_encoders:
                all_g2_classes.extend(self.host_group_2_encoders[group_name].classes_)
        self.unified_host_g2_encoder = LabelEncoder()
        self.unified_host_g2_encoder.fit(all_g2_classes)
        logger.info(f"    Unified host_group_2 encoder ({len(self.unified_host_g2_encoder.classes_)} classes): {list(self.unified_host_g2_encoder.classes_)}")

    def train(self, X, y, graph_structure=None):
        """
        训练多任务模型 (支持半监督学习)
        
        Args:
            X: dict with 'dense' and 'sparse' features
               包含所有样本 (train + no_label)
            y: dict with:
               - 'virus_taxonomy': 所有样本的病毒分类标签 (train + no_label)
                              可能是 virus_order 或 virus_family，可能包含 None/NaN
               - 'host_group_1': 有标签样本的宿主大类 (仅 train，no_label 部分为 None 或 NaN)
               - 'host_group_2': 有标签样本的宿主细分类 (仅 train)
               - 'has_host_label': bool array 标识哪些样本有宿主标签
        """
        # 分离有宿主标签和无宿主标签的样本
        # scaler = StandardScaler()
        # X['dense'] = scaler.fit_transform(X['dense'])
        has_host_label = y.get('has_host_label', None)
        
        # 支持 'virus_taxonomy' (新) 和 'virus_order' (旧) 两种键名
        virus_tax_key = 'virus_taxonomy' if 'virus_taxonomy' in y else 'virus_order'
        
        if has_host_label is None:
            # 如果没有提供 has_host_label，检查 host_group_1 是否有 NaN
            has_host_label = ~pd.isna(y['host_group_1']) if 'host_group_1' in y else np.ones(len(y[virus_tax_key]), dtype=bool)
        
        labeled_mask = np.array(has_host_label, dtype=bool)
        
        # 定义无效标签值（将被视为无标签）
        invalid_labels = {'Unknown', 'unknown', 'UNKNOWN', '', 'NA', 'N/A', 'nan'}
        
        # 检查 virus taxonomy 有效性 (处理 None/NaN/空字符串/Unknown)
        virus_tax_arr = np.array(y[virus_tax_key], dtype=object)
        
        # 优先使用传入的 has_virus_label（如果存在）
        has_virus_label = y.get('has_virus_label', None)
        if not self.enable_virus_taxonomy:
            has_virus_label = np.zeros(len(virus_tax_arr), dtype=bool)
        elif has_virus_label is None:
            # 自动检测无效标签
            has_virus_label = np.array([
                v is not None and pd.notna(v) and str(v).strip() not in invalid_labels
                for v in virus_tax_arr
            ], dtype=bool)
        else:
            has_virus_label = np.array(has_virus_label, dtype=bool)
        
        n_total = len(virus_tax_arr)
        n_host_labeled = labeled_mask.sum()
        n_virus_labeled = has_virus_label.sum()
        n_unlabeled = n_total - n_host_labeled
        
        logger.info(f"    Semi-supervised training:")
        logger.info(f"      - Total samples: {n_total}")
        if self.enable_virus_taxonomy:
            logger.info(f"      - Virus taxonomy labels: {n_virus_labeled}")
        else:
            logger.info("      - Virus taxonomy task disabled")
        logger.info(f"      - Host labels: {n_host_labeled}")
        
        # 设置标签编码器
        # virus taxonomy: 仅使用有有效标签的数据
        # host: 仅使用有标签数据
        y_host_labeled = None
        if n_host_labeled > 0:
            y_host_labeled = {
                'host_group_1': np.array(y['host_group_1'])[labeled_mask],
                'host_group_2': np.array(y['host_group_2'])[labeled_mask]
            }
        self._setup_label_encoders(virus_tax_arr, y_host_labeled)
        
        # 编码样本的 virus taxonomy 标签
        # 对于无效标签，使用 -1 作为占位符
        y_virus = np.full(n_total, -1, dtype=np.int64)
        if self.enable_virus_taxonomy and n_virus_labeled > 0:
            valid_virus_labels = virus_tax_arr[has_virus_label]
            y_virus[has_virus_label] = self.virus_order_encoder.transform(valid_virus_labels)
        
        # 编码有标签样本的 host 标签
        y_host_g1 = np.full(n_total, -1, dtype=np.int64)  # -1 表示无标签
        y_host_g2 = np.full(n_total, -1, dtype=np.int64)
        y_domain = np.where(labeled_mask, 1, 0).astype(np.int64)  # 1=host_label, 0=no_label
        
        if n_host_labeled > 0:
            host_g1_labeled = self.host_group_1_encoder.transform(y_host_labeled['host_group_1'])
            y_host_g1[labeled_mask] = host_g1_labeled
            
            # 编码 host_group_2
            if self.inference_routing == "soft" and self.unified_host_g2_encoder is not None:
                # Soft routing: 使用统一编码
                for i, (g1, g2) in enumerate(zip(y_host_labeled['host_group_1'], y_host_labeled['host_group_2'])):
                    global_idx = np.where(labeled_mask)[0][i]
                    y_host_g2[global_idx] = self.unified_host_g2_encoder.transform([g2])[0]
            else:
                # Hard routing: 使用 per-group 编码
                for i, (g1, g2) in enumerate(zip(y_host_labeled['host_group_1'], y_host_labeled['host_group_2'])):
                    if g1 in self.host_group_2_encoders:
                        global_idx = np.where(labeled_mask)[0][i]
                        y_host_g2[global_idx] = self.host_group_2_encoders[g1].transform([g2])[0]
                        # logger.debug(f"Encoding host_group_2 for sample {global_idx}: group {g1}, label {g2} -> {y_host_g2[global_idx]}")
        
        # 创建数据集 (标签中 -1 表示无标签，CrossEntropyLoss 会自动忽略)
        X1, X2 = X['dense'], X['sparse']
        
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X1, dtype=torch.float32, device=self.device),
            torch.tensor(X2, dtype=torch.float32, device=self.device),
            torch.tensor(y_virus, dtype=torch.long, device=self.device),
            torch.tensor(y_host_g1, dtype=torch.long, device=self.device),
            torch.tensor(y_host_g2, dtype=torch.long, device=self.device),
            torch.tensor(y_domain, dtype=torch.long, device=self.device),
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
        
        # 训练循环
        self.model.train()
        for epoch in range(self.num_epochs):
            # 计算当前 epoch 的动态任务权重
            current_weights = self._get_task_weights(epoch)
            
            epoch_loss = 0.0
            epoch_loss_virus = 0.0
            epoch_loss_host_g1 = 0.0
            epoch_loss_host_g2 = 0.0
            epoch_loss_domain = 0.0
            epoch_domain_correct = 0
            epoch_domain_total = 0
            epoch_grl_lambda_sum = 0.0
            epoch_grl_lambda_last = 0.0
            no_label_embed_count = 0
            no_label_embed_sum = None
            no_label_embed_sq_sum = None
            n_batches = 0
            
            for batch_idx, (batch_X1, batch_X2, batch_y_virus, batch_y_g1, batch_y_g2, batch_y_domain) in enumerate(dataloader):
                self.optimizer.zero_grad()

                current_grl_lambda = self._get_dann_lambda(epoch, batch_idx, len(dataloader)) if self.enable_domain_adversarial else 0.0
                epoch_grl_lambda_sum += float(current_grl_lambda)
                epoch_grl_lambda_last = float(current_grl_lambda)
                
                # 前向传播 (对所有样本)
                virus_logits, host_router_logits, host_expert_outputs, domain_logits, shared_features = self.model(
                    batch_X1,
                    batch_X2,
                    batch_y_g1,
                    return_all_experts=False,
                    grl_lambda=current_grl_lambda,
                    return_features=True,
                )

                # ========== Collapse monitor (no_label only, fast O(n*d)) ==========
                no_label_mask = (batch_y_domain == 0)
                if no_label_mask.any():
                    no_label_feats = shared_features[no_label_mask].detach()
                    if no_label_embed_sum is None:
                        no_label_embed_sum = torch.zeros(no_label_feats.size(1), device=no_label_feats.device)
                        no_label_embed_sq_sum = torch.zeros(no_label_feats.size(1), device=no_label_feats.device)
                    no_label_embed_sum += no_label_feats.sum(dim=0)
                    no_label_embed_sq_sum += (no_label_feats * no_label_feats).sum(dim=0)
                    no_label_embed_count += int(no_label_feats.size(0))
                
                # ========== Task 1: Virus taxonomy ==========
                # CrossEntropyLoss(ignore_index=-1) 会自动忽略标签为 -1 的样本
                loss_virus = torch.tensor(0.0, device=self.device)
                if self.enable_virus_taxonomy and virus_logits is not None and self.virus_order_encoder is not None:
                    loss_virus = self.criterion_virus(virus_logits, batch_y_virus)
                
                # ========== Task 2: Host classification ==========
                # Stage 1: Router loss (ignore_index=-1 自动处理无标签样本)
                loss_host_g1 = self.criterion_host_router(host_router_logits, batch_y_g1)
                
                # Stage 2: Expert loss
                loss_host_g2 = torch.tensor(0.0, device=self.device)
                num_active_groups = 0
                
                if self.inference_routing == "soft":
                    # Soft routing: 所有专家 logits 按 router softmax 加权平均，然后计算统一 loss
                    router_probs_detached = F.softmax(host_router_logits.detach(), dim=1)
                    aggregated_logits = torch.zeros_like(next(iter(host_expert_outputs.values())))
                    for group_idx, group_name in enumerate(self.host_group_1_encoder.classes_):
                        if group_name in host_expert_outputs:
                            aggregated_logits = aggregated_logits + router_probs_detached[:, group_idx:group_idx+1] * host_expert_outputs[group_name]
                    
                    host_mask = (batch_y_g2 != -1)
                    if host_mask.sum() > 0:
                        loss_host_g2 = self.criterion_host_expert(aggregated_logits[host_mask], batch_y_g2[host_mask])
                        if not torch.isnan(loss_host_g2):
                            num_active_groups = 1
                else:
                    # Hard routing: 每个专家只处理对应 group 的样本
                    for group_idx, group_name in enumerate(self.host_group_1_encoder.classes_):
                        # 选择属于该 group 且有标签的样本 (batch_y_g1 == group_idx 且 batch_y_g2 != -1)
                        group_mask = (batch_y_g1 == group_idx)
                        if group_mask.sum() > 0:
                            expert_logits = host_expert_outputs[group_name][group_mask]
                            expert_labels = batch_y_g2[group_mask]
                            
                            # 使用 per-expert 加权 loss（如有），否则使用默认 loss
                            expert_criterion = (
                                self.criterion_host_experts.get(group_name, self.criterion_host_expert)
                                if hasattr(self, 'criterion_host_experts') and self.criterion_host_experts
                                else self.criterion_host_expert
                            )
                            loss_expert = expert_criterion(expert_logits, expert_labels)
                            
                            # 只有当 loss 有效时才累加
                            if not torch.isnan(loss_expert):
                                loss_host_g2 = loss_host_g2 + loss_expert
                                num_active_groups += 1
                
                # 归一化专家损失
                if num_active_groups > 0:
                    loss_host_g2 = loss_host_g2 / num_active_groups

                # ========== DANN: Domain adversarial ==========
                loss_domain = torch.tensor(0.0, device=self.device)
                if self.enable_domain_adversarial and domain_logits is not None:
                    loss_domain = self.criterion_domain(domain_logits, batch_y_domain)
                    domain_pred = torch.argmax(domain_logits, dim=1)
                    epoch_domain_correct += (domain_pred == batch_y_domain).sum().item()
                    epoch_domain_total += batch_y_domain.numel()
                
                # 总损失 (加权组合)
                # 注意: 当某个任务没有有效样本时，对应 loss 为 0 或 nan
                loss = torch.tensor(0.0, device=self.device)
                
                if self.enable_virus_taxonomy and not torch.isnan(loss_virus):
                    loss = loss + current_weights.get('virus_taxonomy', 0.0) * loss_virus
                if not torch.isnan(loss_host_g1):
                    loss = loss + current_weights.get('host_group_1', 0.0) * loss_host_g1
                if num_active_groups > 0:
                    loss = loss + current_weights.get('host_group_2', 0.0) * loss_host_g2
                if self.enable_domain_adversarial and not torch.isnan(loss_domain):
                    loss = loss + self.domain_loss_weight * loss_domain
                
                # 反向传播
                if loss > 0:
                    loss.backward()
                    self.optimizer.step()
                
                # 累计损失 (用于日志)
                epoch_loss += loss.item()
                if not torch.isnan(loss_virus):
                    epoch_loss_virus += loss_virus.item()
                if not torch.isnan(loss_host_g1):
                    epoch_loss_host_g1 += loss_host_g1.item()
                if num_active_groups > 0:
                    epoch_loss_host_g2 += loss_host_g2.item()
                if self.enable_domain_adversarial and not torch.isnan(loss_domain):
                    epoch_loss_domain += loss_domain.item()
                n_batches += 1
            
            # 计算平均损失
            avg_loss = epoch_loss / max(n_batches, 1)
            avg_loss_virus = epoch_loss_virus / max(n_batches, 1)
            avg_loss_host_g1 = epoch_loss_host_g1 / max(n_batches, 1)
            avg_loss_host_g2 = epoch_loss_host_g2 / max(n_batches, 1)
            avg_loss_domain = epoch_loss_domain / max(n_batches, 1)
            avg_grl_lambda = epoch_grl_lambda_sum / max(n_batches, 1)
            domain_acc = epoch_domain_correct / max(epoch_domain_total, 1)

            if no_label_embed_count > 0 and no_label_embed_sum is not None and no_label_embed_sq_sum is not None:
                no_label_mean = no_label_embed_sum / no_label_embed_count
                no_label_var = (no_label_embed_sq_sum / no_label_embed_count) - (no_label_mean * no_label_mean)
                no_label_var = torch.clamp(no_label_var, min=0.0)
                no_label_var_mean = float(no_label_var.mean().item())
                no_label_std_mean = float(torch.sqrt(no_label_var + 1e-12).mean().item())
                var_sum = float(no_label_var.sum().item())
                var_sq_sum = float((no_label_var * no_label_var).sum().item())
                no_label_participation_ratio = (var_sum * var_sum / (var_sq_sum + 1e-12)) if var_sq_sum > 0 else 0.0
            else:
                no_label_var_mean = 0.0
                no_label_std_mean = 0.0
                no_label_participation_ratio = 0.0
            
            if epoch % 1 == 0 or epoch == self.num_epochs - 1:
                logger.info(
                    f"     Epoch [{epoch}/{self.num_epochs}], "
                    f"Loss: {avg_loss:.4f} "
                    f"(Virus: {avg_loss_virus:.4f}, "
                    f"Host_G1: {avg_loss_host_g1:.4f}, "
                    f"Host_G2: {avg_loss_host_g2:.4f}, "
                    f"Domain: {avg_loss_domain:.4f}) | "
                    f"DomainAcc: {domain_acc:.4f} | "
                    f"Weights: V={current_weights.get('virus_taxonomy', 0.0):.2f}, "
                    f"G1={current_weights.get('host_group_1', 0.0):.2f}, "
                    f"G2={current_weights.get('host_group_2', 0.0):.2f}, "
                    f"D={self.domain_loss_weight:.2f} | "
                    f"GRL_lambda(avg/last)={avg_grl_lambda:.4f}/{epoch_grl_lambda_last:.4f} | "
                    f"NoLabelCollapse(var_mean/std_mean/pr)={no_label_var_mean:.6f}/{no_label_std_mean:.6f}/{no_label_participation_ratio:.2f}"
                )

    def predict_proba(self, X, graph_structure=None):
        """
        预测并返回所有任务的 logits
        
        Returns:
            dict with:
                'virus_order': logits for virus taxonomy
                'host_group_1': logits for host group 1 (router)
                'host_group_2': dict of logits for each expert
                'host_group_2_aggregated': aggregated host_group_2 predictions (soft routing)
        """
        self.model.eval()
        with torch.no_grad():
            inputs1 = torch.tensor(X['dense'], dtype=torch.float32).to(self.device)
            inputs2 = torch.tensor(X['sparse'], dtype=torch.float32).to(self.device)
            
            virus_logits, host_router_logits, host_expert_outputs, _ = self.model(
                inputs1, inputs2, return_all_experts=True
            )
            
            result = {
                'host_group_1': host_router_logits.cpu().numpy(),
                'host_group_2': {
                    k: v.cpu().numpy() for k, v in host_expert_outputs.items()
                }
            }
            if self.enable_virus_taxonomy and virus_logits is not None:
                virus_output = virus_logits.cpu().numpy()
                result['virus_taxonomy'] = virus_output
                if self.virus_taxonomy_label_column not in ['virus_taxonomy']:
                    result[self.virus_taxonomy_label_column] = virus_output
                # 保持向后兼容
                result['virus_order'] = virus_output
            
            # Aggregate host_group_2 predictions using soft routing
            # This combines expert predictions weighted by router probabilities
            router_probs = F.softmax(host_router_logits, dim=1)  # [batch, num_groups]
            
            # Build unified host_group_2 prediction space
            aggregated_probs = self._aggregate_expert_predictions(
                router_probs, host_expert_outputs
            )
            result['host_group_2_aggregated'] = aggregated_probs.cpu().numpy()
            
            return result
    
    def _aggregate_expert_predictions(
        self, 
        router_probs: torch.Tensor, 
        expert_outputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Aggregate expert predictions using routing (weighted by gate probabilities).
        
        Soft routing (inference_routing=="soft"): 所有专家输出统一维度，
            final_logits = Σ softmax(router)[g] * expert[g]_logits
        Hard routing: 每个专家输出不同维度，按 argmax(router) 选择专家后映射到统一空间
        
        Args:
            router_probs: [batch_size, num_groups] - gate probabilities
            expert_outputs: Dict[group_name -> [batch_size, num_classes]]
        
        Returns:
            aggregated_probs: [batch_size, total_unique_host_group_2_classes]
        """
        batch_size = router_probs.size(0)
        device = router_probs.device
        
        if self.inference_routing == "soft":
            # Soft routing: 所有专家输出统一维度，直接加权平均 logits
            aggregated_logits = torch.zeros_like(next(iter(expert_outputs.values())))
            for group_idx, group_name in enumerate(self.host_group_1_encoder.classes_):
                if group_name in expert_outputs:
                    aggregated_logits = aggregated_logits + router_probs[:, group_idx:group_idx+1] * expert_outputs[group_name]
            return F.softmax(aggregated_logits, dim=1)
        else:
            # Hard routing: 每个专家输出不同维度，需要映射到统一空间
            router_label_enc = router_probs.argmax(dim=1)
            router_label = self.host_group_1_encoder.inverse_transform(router_label_enc.cpu())
            
            if not hasattr(self, '_unified_host_g2_classes'):
                self._build_unified_class_mapping()
            
            num_unified_classes = len(self._unified_host_g2_classes)
            aggregated = torch.zeros(batch_size, num_unified_classes, device=device)
            
            for group_idx, group_name in enumerate(self.host_group_1_encoder.classes_):
                if group_name not in expert_outputs or group_name not in self.host_group_2_encoders:
                    continue
                group_mask = (router_label == group_name)
                expert_logits = expert_outputs[group_name]
                weighted_probs = F.softmax(expert_logits, dim=1) * torch.tensor(group_mask, dtype=torch.float32, device=device).unsqueeze(1)
                encoder = self.host_group_2_encoders[group_name]
                for local_idx, class_name in enumerate(encoder.classes_):
                    unified_idx = self._class_to_unified_idx[class_name]
                    aggregated[:, unified_idx] += weighted_probs[:, local_idx]
            return aggregated
    
    def _build_unified_class_mapping(self):
        """Build mapping from local expert classes to unified global classes."""
        if self.inference_routing == "soft" and self.unified_host_g2_encoder is not None:
            # Soft routing: 使用统一编码器的类别顺序
            self._unified_host_g2_classes = list(self.unified_host_g2_encoder.classes_)
            self._class_to_unified_idx = {c: i for i, c in enumerate(self._unified_host_g2_classes)}
        else:
            # Hard routing: 按 stage_1_classes 顺序拼接
            self._unified_host_g2_classes = []
            self._class_to_unified_idx = {}
            for group_name in self.stage_1_classes:
                if group_name in self.host_group_2_encoders:
                    for class_name in self.host_group_2_encoders[group_name].classes_:
                        if class_name not in self._class_to_unified_idx:
                            self._class_to_unified_idx[class_name] = len(self._unified_host_g2_classes)
                            self._unified_host_g2_classes.append(class_name)
    
    def predict_host_group_2(self, X, use_hard_routing=False):
        """
        Two-stage prediction for host_group_2.
        
        Args:
            X: Input features dict with 'dense' and 'sparse'
            use_hard_routing: If True, use hard routing (select highest probability expert)
                              If False, use soft routing (weighted combination)
            
        Returns:
            predictions: Predicted host_group_2 labels
            confidences: Prediction confidence scores
        """
        proba_result = self.predict_proba(X)
        
        router_logits = proba_result['host_group_1']
        router_probs = F.softmax(torch.tensor(router_logits), dim=1).numpy()
        
        if use_hard_routing:
            # Hard routing: select expert with highest router probability
            selected_groups = np.argmax(router_probs, axis=1)
            predictions = []
            confidences = []
            
            for i, group_idx in enumerate(selected_groups):
                group_name = self.stage_1_classes[group_idx]
                expert_logits = proba_result['host_group_2'][group_name][i]
                expert_probs = F.softmax(torch.tensor(expert_logits), dim=0).numpy()
                
                pred_idx = np.argmax(expert_probs)
                # Soft routing 时专家输出统一维度，使用统一编码器解码
                if self.inference_routing == "soft" and self.unified_host_g2_encoder is not None:
                    pred_label = self.unified_host_g2_encoder.inverse_transform([pred_idx])[0]
                else:
                    pred_label = self.host_group_2_encoders[group_name].inverse_transform([pred_idx])[0]
                confidence = expert_probs[pred_idx] * router_probs[i, group_idx]
                
                predictions.append(pred_label)
                confidences.append(confidence)
                
            return np.array(predictions), np.array(confidences)
        else:
            # Soft routing: use aggregated predictions
            aggregated_probs = proba_result['host_group_2_aggregated']
            pred_indices = np.argmax(aggregated_probs, axis=1)
            confidences = np.max(aggregated_probs, axis=1)
            
            # Build unified class list if not exists
            if not hasattr(self, '_unified_host_g2_classes'):
                self._build_unified_class_mapping()
            
            predictions = [self._unified_host_g2_classes[idx] for idx in pred_indices]
            return np.array(predictions), confidences
    
    def predict_virus_taxonomy(self, X):
        """预测 virus taxonomy (virus_order 或 virus_family)"""
        if not self.enable_virus_taxonomy:
            raise ValueError("Virus taxonomy task is disabled in the current configuration")
        proba_result = self.predict_proba(X)
        virus_logits = proba_result.get('virus_taxonomy') or proba_result.get(self.virus_taxonomy_label_column) or proba_result.get('virus_order')
        pred_indices = np.argmax(virus_logits, axis=1)
        return self.virus_order_encoder.inverse_transform(pred_indices)
    
    # 别名，保持向后兼容
    predict_virus_order = predict_virus_taxonomy
    
    def predict_host_group_1(self, X):
        """预测 host_group_1"""
        proba_result = self.predict_proba(X)
        router_logits = proba_result['host_group_1']
        pred_indices = np.argmax(router_logits, axis=1)
        return self.host_group_1_encoder.inverse_transform(pred_indices)
    
    def get_internal_model(self):
        return self.model
    
    def get_label_encoders(self):
        """
        返回所有标签编码器，用于结果解码
        
        Returns:
            dict with:
                'virus_taxonomy': LabelEncoder for virus taxonomy (order/family)
                'host_group_1': LabelEncoder for host group 1
                'host_group_2': Dict[group_name -> LabelEncoder] for each host_group_1
        """
        return {
            'virus_taxonomy': self.virus_order_encoder,  # 通用名
            self.virus_taxonomy_label_column: self.virus_order_encoder,
            'virus_order': self.virus_order_encoder,     # 向后兼容
            'host_group_1': self.host_group_1_encoder,
            'host_group_2': self.host_group_2_encoders
        }
    
    def get_unified_host_g2_classes(self) -> List[str]:
        """Get the unified list of host_group_2 class names."""
        if not hasattr(self, '_unified_host_g2_classes'):
            self._build_unified_class_mapping()
        return self._unified_host_g2_classes
    
    def save(self, path: str):
        """
        Save model state, encoders, and configuration.
        
        Args:
            path: Path to save the model checkpoint
        """
        # Build unified class mapping if not exists
        if hasattr(self, '_unified_host_g2_classes'):
            unified_classes = self._unified_host_g2_classes
            class_to_idx = self._class_to_unified_idx
        else:
            unified_classes = None
            class_to_idx = None
        
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'virus_order_encoder': self.virus_order_encoder,
            'host_group_1_encoder': self.host_group_1_encoder,
            'host_group_2_encoders': self.host_group_2_encoders,
            'unified_host_g2_encoder': self.unified_host_g2_encoder,
            'unified_host_g2_classes': unified_classes,
            'class_to_unified_idx': class_to_idx,
            'stage_1_classes': self.stage_1_classes,
            'stage_1_col': self.stage_1_col,
            'stage_2_col': self.stage_2_col,
            'config': self.config
        }
        torch.save(state, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """
        Load model state, encoders, and configuration.
        
        Args:
            path: Path to the saved model checkpoint
        """
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.virus_order_encoder = state['virus_order_encoder']
        self.host_group_1_encoder = state['host_group_1_encoder']
        self.host_group_2_encoders = state['host_group_2_encoders']
        self.unified_host_g2_encoder = state.get('unified_host_g2_encoder', None)
        
        # 恢复 config-driven 层级信息
        if 'stage_1_classes' in state:
            self.stage_1_classes = state['stage_1_classes']
        if 'stage_1_col' in state:
            self.stage_1_col = state['stage_1_col']
        if 'stage_2_col' in state:
            self.stage_2_col = state['stage_2_col']
        
        if state.get('unified_host_g2_classes') is not None:
            self._unified_host_g2_classes = state['unified_host_g2_classes']
            self._class_to_unified_idx = state['class_to_unified_idx']
        
        logger.info(f"Model loaded from {path}")
    
    def get_task_metrics(self, X, y_true: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """
        Compute evaluation metrics for all tasks.
        
        Args:
            X: Input features
            y_true: Dict with true labels for 'virus_order', 'host_group_1', 'host_group_2'
        
        Returns:
            Dict with accuracy and other metrics for each task
        """
        from sklearn.metrics import accuracy_score, f1_score
        
        metrics = {}
        tax_key = self.virus_taxonomy_label_column
        
        # Task 1: Virus order
        if self.enable_virus_taxonomy and self.virus_order_encoder is not None and tax_key in y_true and y_true[tax_key] is not None:
            virus_preds = self.predict_virus_taxonomy(X)
            metrics[tax_key] = {
                'accuracy': accuracy_score(y_true[tax_key], virus_preds),
                'f1_macro': f1_score(y_true[tax_key], virus_preds, average='macro', zero_division=0)
            }
        
        # Task 2a: Host group 1
        if 'host_group_1' in y_true and y_true['host_group_1'] is not None:
            host_g1_preds = self.predict_host_group_1(X)
            metrics['host_group_1'] = {
                'accuracy': accuracy_score(y_true['host_group_1'], host_g1_preds),
                'f1_macro': f1_score(y_true['host_group_1'], host_g1_preds, average='macro', zero_division=0)
            }
        
        # Task 2b: Host group 2 (using soft routing)
        if 'host_group_2' in y_true and y_true['host_group_2'] is not None:
            host_g2_preds, _ = self.predict_host_group_2(X, use_hard_routing=False)
            metrics['host_group_2_soft'] = {
                'accuracy': accuracy_score(y_true['host_group_2'], host_g2_preds),
                'f1_macro': f1_score(y_true['host_group_2'], host_g2_preds, average='macro', zero_division=0)
            }
            
            # Task 2b: Host group 2 (using hard routing)
            host_g2_preds_hard, _ = self.predict_host_group_2(X, use_hard_routing=True)
            metrics['host_group_2_hard'] = {
                'accuracy': accuracy_score(y_true['host_group_2'], host_g2_preds_hard),
                'f1_macro': f1_score(y_true['host_group_2'], host_g2_preds_hard, average='macro', zero_division=0)
            }
        
        return metrics
