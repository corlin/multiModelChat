"""
核心模块

包含数据模型、配置管理、参数验证和异常定义。
"""

from .models import ModelInfo, ModelType, ModelConfig, InferenceResult, InferenceStats
from .config import ConfigManager
from .validators import ParameterValidator, RealTimeValidator
from .exceptions import (
    MultiLLMComparatorError,
    ModelLoadError,
    ModelNotFoundError,
    InferenceError,
    ConfigurationError,
    MemoryError,
    ValidationError,
    ModelSelectionError
)

__all__ = [
    # Models
    "ModelInfo",
    "ModelType", 
    "ModelConfig",
    "InferenceResult",
    "InferenceStats",
    
    # Configuration
    "ConfigManager",
    
    # Validation
    "ParameterValidator",
    "RealTimeValidator",
    
    # Exceptions
    "MultiLLMComparatorError",
    "ModelLoadError",
    "ModelNotFoundError", 
    "InferenceError",
    "ConfigurationError",
    "MemoryError",
    "ValidationError",
    "ModelSelectionError",
]