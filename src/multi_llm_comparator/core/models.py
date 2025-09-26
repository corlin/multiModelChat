"""
数据模型定义

定义系统中使用的核心数据结构。
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class ModelType(Enum):
    """模型类型枚举"""
    PYTORCH = "pytorch"
    GGUF = "gguf"
    OPENAI_API = "openai_api"


@dataclass
class ModelInfo:
    """模型信息数据类"""
    id: str
    name: str
    path: str
    model_type: ModelType
    size: int
    config: Dict[str, Any]


@dataclass
class InferenceStats:
    """推理统计信息"""
    start_time: float
    end_time: Optional[float]
    token_count: int
    tokens_per_second: Optional[float]


@dataclass
class InferenceResult:
    """推理结果数据类"""
    model_id: str
    content: str
    is_complete: bool
    error: Optional[str]
    stats: InferenceStats


@dataclass
class ModelConfig:
    """模型配置参数"""
    # 通用参数
    temperature: float = 0.7
    max_tokens: int = 512
    top_p: float = 0.9
    
    # PyTorch特定参数
    do_sample: bool = True
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    
    # GGUF特定参数  
    top_k: int = 40
    repeat_penalty: float = 1.1
    n_ctx: int = 2048
    n_threads: Optional[int] = None
    use_gpu: bool = True
    
    # 内存管理参数
    low_cpu_mem_usage: bool = True
    torch_dtype: str = "auto"
    
    # OpenAI API特定参数
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model_name: Optional[str] = None
    stream: bool = True
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0