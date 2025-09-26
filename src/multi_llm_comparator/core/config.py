"""
配置管理

处理应用程序配置和参数验证。
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

from .models import ModelConfig, ModelType
from .validators import ParameterValidator
from .exceptions import ValidationError


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_dir: Optional[str] = None):
        """初始化配置管理器"""
        if config_dir is None:
            config_dir = os.path.expanduser("~/.multi_llm_comparator")
        
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.config_dir / "config.json"
        self.validator = ParameterValidator()
    
    def save_config(self, config: Dict[str, Any]) -> None:
        """保存配置到文件"""
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    def load_config(self) -> Dict[str, Any]:
        """从文件加载配置"""
        if not self.config_file.exists():
            return {}
        
        with open(self.config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_model_config(self, model_id: str) -> ModelConfig:
        """获取指定模型的配置"""
        config = self.load_config()
        model_configs = config.get("model_configs", {})
        model_config_dict = model_configs.get(model_id, {})
        
        return ModelConfig(**model_config_dict)
    
    def save_model_config(
        self, 
        model_id: str, 
        model_config: ModelConfig, 
        model_type: Optional[ModelType] = None,
        validate: bool = True
    ) -> None:
        """
        保存指定模型的配置
        
        Args:
            model_id: 模型ID
            model_config: 模型配置
            model_type: 模型类型（可选）
            validate: 是否验证配置
        """
        # 确保model_config是ModelConfig对象
        if not isinstance(model_config, ModelConfig):
            raise ValidationError(f"model_config必须是ModelConfig对象，得到: {type(model_config)}")
        
        if validate:
            is_valid, errors = self.validator.validate_config(model_config, model_type)
            if not is_valid:
                raise ValidationError(f"配置验证失败: {'; '.join(errors)}")
        
        config = self.load_config()
        if "model_configs" not in config:
            config["model_configs"] = {}
        
        # 安全地转换ModelConfig为字典
        try:
            config_dict = model_config.__dict__.copy()
            config["model_configs"][model_id] = config_dict
            self.save_config(config)
        except Exception as e:
            raise ValidationError(f"保存配置时发生错误: {str(e)}")
    
    def validate_model_config(
        self, 
        model_config: ModelConfig, 
        model_type: Optional[ModelType] = None
    ) -> Tuple[bool, List[str]]:
        """
        验证模型配置
        
        Args:
            model_config: 模型配置
            model_type: 模型类型（可选）
            
        Returns:
            (is_valid, error_messages)
        """
        return self.validator.validate_config(model_config, model_type)
    
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
        return self.validator.validate_parameter(param_name, value, model_type)
    
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
        return self.validator.get_parameter_suggestions(param_name, current_value, model_type)