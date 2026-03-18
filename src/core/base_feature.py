from abc import ABC, abstractmethod
import pandas as pd
import os
from pathlib import Path
from src.utils.logger import get_logger
logger = get_logger(__name__)


class BaseFeatureExtractor(ABC):
    def __init__(self, config):
        self.config = config
        release_root = Path(__file__).resolve().parents[2]
        self.prodigal_path = str(release_root / "scripts" / "parallel-prodigal-gv.py")
        self.data = {}

    @abstractmethod
    def load(self, file_path):
        """读取特征文件"""
        pass

    @abstractmethod
    def generate(self, fasta_path, output_path):
        """如果文件不存在，基于fasta生成特征"""
        pass

    def get_feature(self, fasta_path, filename, source):
        """模板方法：先尝试读取，失败则生成"""
        if source in self.data:
            return self.data[source]  # 已经加载过，直接返回缓存数据
        else:            
            if not os.path.exists(filename):
                logger.info(f"      Feature file {filename} not found. Generating...")
                self.generate(fasta_path, filename)
            self.data[source] = self.load(filename)
            return self.data[source]