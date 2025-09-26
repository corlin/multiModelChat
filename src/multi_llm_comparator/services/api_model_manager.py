"""
API模型管理器

管理OpenAI兼容的API模型，包括Doubao等云端模型。
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import asdict

from ..core.models import ModelInfo, ModelType, ModelConfig


logger = logging.getLogger(__name__)


class APIModelManager:
    """API模型管理器"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        初始化API模型管理器
        
        Args:
            config_file: API模型配置文件路径
        """
        if config_file is None:
            config_file = "api_models.json"
        self.config_file = Path(config_file)
        self.api_models: Dict[str, ModelInfo] = {}
        self._load_api_models()
    
    def add_doubao_model(
        self, 
        model_id: str, 
        model_name: str, 
        display_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> ModelInfo:
        """
        添加Doubao模型
        
        Args:
            model_id: 模型ID（如doubao-seed-1-6-250615）
            model_name: 模型名称
            display_name: 显示名称
            api_key: API密钥
            base_url: API基础URL
            
        Returns:
            ModelInfo: 创建的模型信息
        """
        # 创建默认配置
        config = ModelConfig(
            temperature=0.7,
            max_tokens=1024,
            top_p=0.9,
            api_key=api_key,
            base_url=base_url or "https://ark.cn-beijing.volces.com/api/v3",
            model_name=model_id,
            stream=True,
            presence_penalty=0.0,
            frequency_penalty=0.0
        )
        
        # 创建模型信息
        model_info = ModelInfo(
            id=f"doubao_{model_id}",
            name=display_name or f"Doubao {model_name}",
            path=model_id,  # 对于API模型，path存储模型ID
            model_type=ModelType.OPENAI_API,
            size=0,  # API模型没有本地大小
            config=asdict(config)
        )
        
        self.api_models[model_info.id] = model_info
        self._save_api_models()
        
        logger.info(f"已添加Doubao模型: {model_info.name}")
        return model_info
    
    def add_openai_model(
        self, 
        model_id: str, 
        model_name: str, 
        display_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> ModelInfo:
        """
        添加OpenAI模型
        
        Args:
            model_id: 模型ID（如gpt-3.5-turbo）
            model_name: 模型名称
            display_name: 显示名称
            api_key: API密钥
            base_url: API基础URL
            
        Returns:
            ModelInfo: 创建的模型信息
        """
        # 创建默认配置
        config = ModelConfig(
            temperature=0.7,
            max_tokens=1024,
            top_p=0.9,
            api_key=api_key,
            base_url=base_url or "https://api.openai.com/v1",
            model_name=model_id,
            stream=True,
            presence_penalty=0.0,
            frequency_penalty=0.0
        )
        
        # 创建模型信息
        model_info = ModelInfo(
            id=f"openai_{model_id}",
            name=display_name or f"OpenAI {model_name}",
            path=model_id,  # 对于API模型，path存储模型ID
            model_type=ModelType.OPENAI_API,
            size=0,  # API模型没有本地大小
            config=asdict(config)
        )
        
        self.api_models[model_info.id] = model_info
        self._save_api_models()
        
        logger.info(f"已添加OpenAI模型: {model_info.name}")
        return model_info
    
    def get_api_models(self) -> List[ModelInfo]:
        """
        获取所有API模型
        
        Returns:
            List[ModelInfo]: API模型列表
        """
        return list(self.api_models.values())
    
    def get_doubao_models(self) -> List[ModelInfo]:
        """
        获取所有Doubao模型
        
        Returns:
            List[ModelInfo]: Doubao模型列表
        """
        return [model for model in self.api_models.values() 
                if "doubao" in model.id.lower()]
    
    def remove_model(self, model_id: str) -> bool:
        """
        移除API模型
        
        Args:
            model_id: 模型ID
            
        Returns:
            bool: 是否成功移除
        """
        if model_id in self.api_models:
            del self.api_models[model_id]
            self._save_api_models()
            logger.info(f"已移除API模型: {model_id}")
            return True
        return False
    
    def _load_api_models(self) -> None:
        """从文件加载API模型配置"""
        try:
            if not self.config_file.exists():
                # 创建默认的Doubao模型配置
                self._create_default_models()
                return
            
            with open(self.config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.api_models.clear()
            for model_id, model_data in data.get('api_models', {}).items():
                # 重建ModelInfo对象
                model_data['model_type'] = ModelType(model_data['model_type'])
                model_info = ModelInfo(**model_data)
                self.api_models[model_id] = model_info
            
            logger.info(f"从配置文件加载了 {len(self.api_models)} 个API模型")
            
        except Exception as e:
            logger.error(f"加载API模型配置失败: {str(e)}")
            self._create_default_models()
    
    def _save_api_models(self) -> None:
        """保存API模型配置到文件"""
        try:
            # 转换模型数据
            api_models_data = {}
            for model_id, model in self.api_models.items():
                model_dict = asdict(model)
                model_dict['model_type'] = model.model_type.value
                api_models_data[model_id] = model_dict
            
            data = {
                'api_models': api_models_data
            }
            
            # 确保目录存在
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"API模型配置已保存到: {self.config_file}")
            
        except Exception as e:
            logger.error(f"保存API模型配置失败: {str(e)}")
    
    def _create_default_models(self) -> None:
        """创建默认的API模型"""
        try:
            # 添加默认的Doubao模型
            self.add_doubao_model(
                model_id="doubao-seed-1-6-250615",
                model_name="Seed 1.6",
                display_name="Doubao Seed 1.6"
            )
            
            self.add_doubao_model(
                model_id="doubao-pro-4k",
                model_name="Pro 4K",
                display_name="Doubao Pro 4K"
            )
            
            self.add_doubao_model(
                model_id="doubao-lite-4k",
                model_name="Lite 4K", 
                display_name="Doubao Lite 4K"
            )
            
            logger.info("已创建默认的Doubao模型配置")
            
        except Exception as e:
            logger.error(f"创建默认API模型失败: {str(e)}")