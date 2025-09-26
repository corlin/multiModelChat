"""
自定义异常定义

定义系统中使用的自定义异常类。
"""


class MultiLLMComparatorError(Exception):
    """基础异常类"""
    pass


class ModelLoadError(MultiLLMComparatorError):
    """模型加载错误"""
    pass


class ModelNotFoundError(MultiLLMComparatorError):
    """模型未找到错误"""
    pass


class InferenceError(MultiLLMComparatorError):
    """推理过程错误"""
    pass


class ConfigurationError(MultiLLMComparatorError):
    """配置错误"""
    pass


class MemoryError(MultiLLMComparatorError):
    """内存管理错误"""
    pass


class ValidationError(MultiLLMComparatorError):
    """参数验证错误"""
    pass


class ModelSelectionError(MultiLLMComparatorError):
    """模型选择错误"""
    pass