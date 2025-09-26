"""
基础推理器

定义推理器的通用接口。
"""

from abc import ABC, abstractmethod
from typing import Dict, Iterator, Any, Optional

from ..core.models import InferenceStats


class BaseInferencer(ABC):
    """基础推理器抽象类"""
    
    def __init__(self):
        """初始化推理器"""
        self.model = None
        self.is_loaded = False
    
    @abstractmethod
    def load_model(self, model_path: str, config: Dict[str, Any]) -> None:
        """加载模型"""
        pass
    
    @abstractmethod
    def generate_stream(self, prompt: str) -> Iterator[str]:
        """流式生成文本"""
        pass
    
    @abstractmethod
    def generate_complete(self, prompt: str) -> str:
        """非流式生成完整文本"""
        pass
    
    @abstractmethod
    def unload_model(self) -> None:
        """卸载模型"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        pass
    
    def is_model_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.is_loaded
    
    def get_inference_stats(self) -> Optional[InferenceStats]:
        """
        获取最近一次推理的统计信息
        
        Returns:
            Optional[InferenceStats]: 推理统计信息，如果没有进行过推理则返回None
        """
        return None