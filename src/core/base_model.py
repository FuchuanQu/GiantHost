from abc import ABC, abstractmethod
from src.utils.logger import get_logger
logger = get_logger(__name__)

class BaseModel(ABC):
    def __init__(self, config):
        self.config = config
        self.device = config['general']['device']
    @abstractmethod
    def train(self, X, y, graph_structure=None):
        """
        X: 特征矩阵 或 字典(对于双输入模型)
        y: 标签
        graph_structure: 邻接矩阵/EdgeIndex (对于GNN/LabelSpread)
        """
        pass

    @abstractmethod
    def predict_proba(self, X, graph_structure=None):
        """返回 logits 或 probabilities"""
        pass
    
    @abstractmethod
    def get_internal_model(self):
        """返回底层的 torch model 或 sklearn model，用于 Captum 分析"""
        pass