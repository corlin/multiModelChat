"""
OpenAI API推理器

支持OpenAI兼容的API接口，包括Doubao等模型。
"""

import os
import logging
import time
from typing import Iterator, Optional, Dict, Any

from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from .base import BaseInferencer
from ..core.models import ModelConfig
from ..core.exceptions import ModelLoadError, InferenceError


logger = logging.getLogger(__name__)


class OpenAIInferencer(BaseInferencer):
    """OpenAI API推理器"""
    
    def __init__(self):
        """初始化OpenAI推理器"""
        super().__init__()
        self.client: Optional[OpenAI] = None
        self.model_name: Optional[str] = None
        self.config: Optional[ModelConfig] = None
        
    def load_model(self, model_path: str, config: Dict[str, Any]) -> None:
        """
        加载OpenAI API模型
        
        Args:
            model_path: 模型路径（对于API模型，这里是模型名称）
            config: 模型配置
        """
        try:
            # 转换配置
            if isinstance(config, dict):
                self.config = ModelConfig(**config)
            else:
                self.config = config
            
            # 获取API配置
            api_key = self.config.api_key or os.environ.get("ARK_API_KEY") or os.environ.get("OPENAI_API_KEY")
            base_url = self.config.base_url or "https://ark.cn-beijing.volces.com/api/v3"
            
            if not api_key:
                raise ModelLoadError("未找到API Key，请设置ARK_API_KEY或OPENAI_API_KEY环境变量，或在配置中指定api_key")
            
            # 初始化OpenAI客户端
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
            
            # 设置模型名称
            self.model_name = self.config.model_name or model_path or "doubao-seed-1-6-250615"
            
            self.is_loaded = True
            logger.info(f"OpenAI API模型已加载: {self.model_name}")
            
        except Exception as e:
            logger.error(f"加载OpenAI API模型失败: {str(e)}")
            raise ModelLoadError(f"加载OpenAI API模型失败: {str(e)}")
    
    def generate_stream(self, prompt: str) -> Iterator[str]:
        """
        流式生成文本
        
        Args:
            prompt: 输入提示词
            
        Yields:
            str: 生成的文本片段
        """
        if not self.is_loaded or not self.client:
            raise InferenceError("模型未加载")
        
        try:
            # 构建消息
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            # 调用API进行流式生成
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                presence_penalty=self.config.presence_penalty,
                frequency_penalty=self.config.frequency_penalty,
                stream=True
            )
            
            # 处理流式响应
            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        yield delta.content
                        
        except Exception as e:
            logger.error(f"OpenAI API流式生成失败: {str(e)}")
            raise InferenceError(f"OpenAI API流式生成失败: {str(e)}")
    
    def generate_complete(self, prompt: str) -> str:
        """
        完整生成文本
        
        Args:
            prompt: 输入提示词
            
        Returns:
            str: 生成的完整文本
        """
        if not self.is_loaded or not self.client:
            raise InferenceError("模型未加载")
        
        try:
            # 构建消息
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            # 调用API进行完整生成
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                presence_penalty=self.config.presence_penalty,
                frequency_penalty=self.config.frequency_penalty,
                stream=False
            )
            
            # 提取生成的文本
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content or ""
            else:
                return ""
                
        except Exception as e:
            logger.error(f"OpenAI API完整生成失败: {str(e)}")
            raise InferenceError(f"OpenAI API完整生成失败: {str(e)}")
    
    def unload_model(self) -> None:
        """卸载模型"""
        self.client = None
        self.model_name = None
        self.config = None
        self.is_loaded = False
        logger.info("OpenAI API模型已卸载")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict[str, Any]: 模型信息
        """
        return {
            "type": "OpenAI API",
            "model_name": self.model_name,
            "base_url": self.config.base_url if self.config else None,
            "is_loaded": self.is_loaded,
            "supports_streaming": True
        }


class DoubaoInferencer(OpenAIInferencer):
    """Doubao模型推理器（基于OpenAI API）"""
    
    def __init__(self):
        """初始化Doubao推理器"""
        super().__init__()
    
    def load_model(self, model_path: str, config: Dict[str, Any]) -> None:
        """
        加载Doubao模型
        
        Args:
            model_path: 模型路径（Doubao模型名称）
            config: 模型配置
        """
        try:
            # 转换配置
            if isinstance(config, dict):
                self.config = ModelConfig(**config)
            else:
                self.config = config
            
            # 设置Doubao特定的默认值
            api_key = self.config.api_key or os.environ.get("ARK_API_KEY")
            base_url = self.config.base_url or "https://ark.cn-beijing.volces.com/api/v3"
            
            if not api_key:
                raise ModelLoadError("未找到Doubao API Key，请设置ARK_API_KEY环境变量或在配置中指定api_key")
            
            # 初始化OpenAI客户端（Doubao兼容OpenAI API）
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
            
            # 设置Doubao模型名称
            self.model_name = self.config.model_name or model_path or "doubao-seed-1-6-250615"
            
            self.is_loaded = True
            logger.info(f"Doubao模型已加载: {self.model_name}")
            
        except Exception as e:
            logger.error(f"加载Doubao模型失败: {str(e)}")
            raise ModelLoadError(f"加载Doubao模型失败: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取Doubao模型信息
        
        Returns:
            Dict[str, Any]: 模型信息
        """
        return {
            "type": "Doubao API",
            "model_name": self.model_name,
            "base_url": self.config.base_url if self.config else None,
            "is_loaded": self.is_loaded,
            "supports_streaming": True,
            "provider": "ByteDance"
        }