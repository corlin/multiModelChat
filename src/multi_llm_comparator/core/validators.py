"""
参数验证器

实现不同模型类型的参数验证规则和实时验证功能。
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import fields

from .models import ModelConfig, ModelType
from .exceptions import ValidationError


class ParameterValidator:
    """参数验证器"""
    
    # 参数范围定义
    PARAMETER_RANGES = {
        # 通用参数
        "temperature": (0.0, 2.0),
        "max_tokens": (1, 8192),
        "top_p": (0.0, 1.0),
        
        # PyTorch特定参数
        "pad_token_id": (0, 100000),
        "eos_token_id": (0, 100000),
        
        # GGUF特定参数
        "top_k": (1, 100),
        "repeat_penalty": (0.0, 2.0),
        "n_ctx": (1, 32768),
        "n_threads": (1, 64),
    }
    
    # 模型类型特定参数
    PYTORCH_PARAMETERS = {
        "temperature", "max_tokens", "top_p", "do_sample", 
        "pad_token_id", "eos_token_id", "low_cpu_mem_usage", "torch_dtype"
    }
    
    GGUF_PARAMETERS = {
        "temperature", "max_tokens", "top_p", "top_k", 
        "repeat_penalty", "n_ctx", "n_threads", "use_gpu"
    }
    
    # 有效的torch_dtype值
    VALID_TORCH_DTYPES = {
        "auto", "float16", "bfloat16", "float32", "int8", "int4"
    }
    
    def __init__(self):
        """初始化验证器"""
        pass
    
    def validate_parameter(
        self, 
        param_name: str, 
        value: Any, 
        model_type: Optional[ModelType] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        验证单个参数
        
        Args:
            param_name: 参数名称
            value: 参数值
            model_type: 模型类型（可选）
            
        Returns:
            (is_valid, error_message)
        """
        try:
            # 检查参数是否适用于指定的模型类型
            if model_type:
                if not self._is_parameter_compatible(param_name, model_type):
                    return False, f"参数 '{param_name}' 不适用于 {model_type.value} 模型"
            
            # 类型验证
            if not self._validate_type(param_name, value):
                expected_type = self._get_expected_type(param_name)
                return False, f"参数 '{param_name}' 类型错误，期望 {expected_type}，得到 {type(value).__name__}"
            
            # 范围验证
            if not self._validate_range(param_name, value):
                range_info = self._get_range_info(param_name)
                return False, f"参数 '{param_name}' 超出有效范围 {range_info}"
            
            # 特殊值验证
            if not self._validate_special_values(param_name, value):
                valid_values = self._get_valid_values(param_name)
                return False, f"参数 '{param_name}' 值无效，有效值: {valid_values}"
            
            return True, None
            
        except Exception as e:
            return False, f"验证参数 '{param_name}' 时发生错误: {str(e)}"
    
    def validate_config(
        self, 
        config: Union[ModelConfig, Dict[str, Any]], 
        model_type: Optional[ModelType] = None
    ) -> Tuple[bool, List[str]]:
        """
        验证完整的模型配置
        
        Args:
            config: 模型配置对象或字典
            model_type: 模型类型（可选）
            
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        try:
            # 安全地转换为字典格式
            if isinstance(config, ModelConfig):
                config_dict = config.__dict__.copy()
            elif isinstance(config, dict):
                config_dict = config.copy()
            else:
                errors.append(f"配置类型错误: 期望ModelConfig或dict，得到{type(config)}")
                return False, errors
            
            # 如果指定了模型类型，只验证该类型相关的参数
            if model_type:
                relevant_params = self._get_relevant_parameters(model_type)
                config_dict = {k: v for k, v in config_dict.items() if k in relevant_params}
            
            # 验证每个参数
            for param_name, value in config_dict.items():
                is_valid, error_msg = self.validate_parameter(param_name, value, model_type)
                if not is_valid:
                    errors.append(error_msg)
            
            # 检查参数兼容性
            compatibility_errors = self._check_parameter_compatibility(config_dict, model_type)
            errors.extend(compatibility_errors)
            
            return len(errors) == 0, errors
            
        except Exception as e:
            errors.append(f"验证配置时发生错误: {str(e)}")
            return False, errors
    
    def get_parameter_suggestions(
        self, 
        param_name: str, 
        current_value: Any,
        model_type: Optional[ModelType] = None
    ) -> List[str]:
        """
        获取参数修正建议
        
        Args:
            param_name: 参数名称
            current_value: 当前值
            model_type: 模型类型（可选）
            
        Returns:
            建议列表
        """
        suggestions = []
        
        # 检查参数是否适用于模型类型
        if model_type and not self._is_parameter_compatible(param_name, model_type):
            suggestions.append(f"参数 '{param_name}' 不适用于 {model_type.value} 模型，建议移除")
            return suggestions
        
        # 范围建议
        if param_name in self.PARAMETER_RANGES:
            min_val, max_val = self.PARAMETER_RANGES[param_name]
            if isinstance(current_value, (int, float)):
                if current_value < min_val:
                    suggestions.append(f"建议将 '{param_name}' 设置为不小于 {min_val}")
                elif current_value > max_val:
                    suggestions.append(f"建议将 '{param_name}' 设置为不大于 {max_val}")
        
        # 特殊值建议
        if param_name == "torch_dtype" and current_value not in self.VALID_TORCH_DTYPES:
            suggestions.append(f"建议将 'torch_dtype' 设置为: {', '.join(self.VALID_TORCH_DTYPES)}")
        
        # 常用值建议
        common_values = self._get_common_values(param_name)
        if common_values:
            suggestions.append(f"常用值: {', '.join(map(str, common_values))}")
        
        return suggestions
    
    def _is_parameter_compatible(self, param_name: str, model_type: ModelType) -> bool:
        """检查参数是否与模型类型兼容"""
        if model_type == ModelType.PYTORCH:
            return param_name in self.PYTORCH_PARAMETERS
        elif model_type == ModelType.GGUF:
            return param_name in self.GGUF_PARAMETERS
        return True
    
    def _validate_type(self, param_name: str, value: Any) -> bool:
        """验证参数类型"""
        # 获取ModelConfig中的字段类型
        model_config_fields = {f.name: f.type for f in fields(ModelConfig)}
        
        if param_name not in model_config_fields:
            return True  # 未知参数跳过类型检查
        
        expected_type = model_config_fields[param_name]
        
        # 处理Optional类型
        if hasattr(expected_type, '__origin__') and expected_type.__origin__ is Union:
            # Optional[T] 等价于 Union[T, None]
            args = expected_type.__args__
            if len(args) == 2 and type(None) in args:
                if value is None:
                    return True
                expected_type = args[0] if args[1] is type(None) else args[1]
        
        # 基本类型检查
        if expected_type == bool:
            return isinstance(value, bool)
        elif expected_type == int:
            return isinstance(value, int)
        elif expected_type == float:
            return isinstance(value, (int, float))
        elif expected_type == str:
            return isinstance(value, str)
        
        return True
    
    def _validate_range(self, param_name: str, value: Any) -> bool:
        """验证参数范围"""
        if param_name not in self.PARAMETER_RANGES:
            return True
        
        if not isinstance(value, (int, float)):
            return True
        
        min_val, max_val = self.PARAMETER_RANGES[param_name]
        return min_val <= value <= max_val
    
    def _validate_special_values(self, param_name: str, value: Any) -> bool:
        """验证特殊值"""
        if param_name == "torch_dtype":
            return value in self.VALID_TORCH_DTYPES
        
        return True
    
    def _get_expected_type(self, param_name: str) -> str:
        """获取参数期望类型的字符串表示"""
        model_config_fields = {f.name: f.type for f in fields(ModelConfig)}
        
        if param_name not in model_config_fields:
            return "unknown"
        
        expected_type = model_config_fields[param_name]
        
        # 处理Optional类型
        if hasattr(expected_type, '__origin__') and expected_type.__origin__ is Union:
            args = expected_type.__args__
            if len(args) == 2 and type(None) in args:
                expected_type = args[0] if args[1] is type(None) else args[1]
                return f"Optional[{expected_type.__name__}]"
        
        return expected_type.__name__ if hasattr(expected_type, '__name__') else str(expected_type)
    
    def _get_range_info(self, param_name: str) -> str:
        """获取参数范围信息"""
        if param_name in self.PARAMETER_RANGES:
            min_val, max_val = self.PARAMETER_RANGES[param_name]
            return f"[{min_val}, {max_val}]"
        return "无限制"
    
    def _get_valid_values(self, param_name: str) -> str:
        """获取有效值信息"""
        if param_name == "torch_dtype":
            return ", ".join(self.VALID_TORCH_DTYPES)
        return "未定义"
    
    def _get_common_values(self, param_name: str) -> List[Any]:
        """获取常用值"""
        common_values = {
            "temperature": [0.1, 0.7, 1.0],
            "top_p": [0.9, 0.95, 1.0],
            "top_k": [20, 40, 50],
            "max_tokens": [256, 512, 1024, 2048],
            "repeat_penalty": [1.0, 1.1, 1.2],
            "n_ctx": [1024, 2048, 4096],
        }
        return common_values.get(param_name, [])
    
    def _get_relevant_parameters(self, model_type: ModelType) -> set:
        """获取指定模型类型的相关参数"""
        if model_type == ModelType.PYTORCH:
            return self.PYTORCH_PARAMETERS
        elif model_type == ModelType.GGUF:
            return self.GGUF_PARAMETERS
        else:
            # 如果模型类型未知，返回所有参数
            return self.PYTORCH_PARAMETERS.union(self.GGUF_PARAMETERS)
    
    def _check_parameter_compatibility(
        self, 
        config_dict: Dict[str, Any], 
        model_type: Optional[ModelType]
    ) -> List[str]:
        """检查参数间的兼容性"""
        errors = []
        
        # 检查temperature和top_p的组合
        temperature = config_dict.get("temperature", 0.7)
        top_p = config_dict.get("top_p", 0.9)
        
        if temperature == 0.0 and top_p < 1.0:
            errors.append("当temperature为0时，建议将top_p设置为1.0以确保确定性输出")
        
        # 检查GGUF特定的兼容性
        if model_type == ModelType.GGUF:
            n_ctx = config_dict.get("n_ctx", 2048)
            max_tokens = config_dict.get("max_tokens", 512)
            
            if max_tokens > n_ctx:
                errors.append(f"max_tokens ({max_tokens}) 不能大于 n_ctx ({n_ctx})")
        
        return errors


class RealTimeValidator:
    """实时参数验证器"""
    
    def __init__(self):
        """初始化实时验证器"""
        self.validator = ParameterValidator()
        self._validation_cache = {}
    
    def validate_on_change(
        self, 
        param_name: str, 
        value: Any, 
        model_type: Optional[ModelType] = None
    ) -> Dict[str, Any]:
        """
        参数变更时的实时验证
        
        Args:
            param_name: 参数名称
            value: 新值
            model_type: 模型类型（可选）
            
        Returns:
            验证结果字典
        """
        # 生成缓存键
        cache_key = f"{param_name}_{value}_{model_type}"
        
        # 检查缓存
        if cache_key in self._validation_cache:
            return self._validation_cache[cache_key]
        
        # 执行验证
        is_valid, error_msg = self.validator.validate_parameter(param_name, value, model_type)
        suggestions = self.validator.get_parameter_suggestions(param_name, value, model_type)
        
        result = {
            "is_valid": is_valid,
            "error_message": error_msg,
            "suggestions": suggestions,
            "parameter": param_name,
            "value": value,
            "model_type": model_type.value if model_type else None
        }
        
        # 缓存结果
        self._validation_cache[cache_key] = result
        
        return result
    
    def clear_cache(self):
        """清除验证缓存"""
        self._validation_cache.clear()