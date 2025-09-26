"""
PyTorch推理器实现

使用transformers库实现PyTorch模型的加载和推理功能。
"""

import gc
import logging
import time
from typing import Any, Dict, Iterator, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    GenerationConfig,
)
from threading import Thread

from ..core.models import InferenceStats
from ..core.exceptions import ModelLoadError, InferenceError
from .base import BaseInferencer

logger = logging.getLogger(__name__)


class PyTorchInferencer(BaseInferencer):
    """PyTorch模型推理器"""
    
    def __init__(self):
        """初始化PyTorch推理器"""
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.device = None
        self.model_path = None
        self.config = None
        
    def _detect_device(self) -> str:
        """自动检测可用的设备"""
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"使用CUDA设备: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
            logger.info("使用MPS设备 (Apple Silicon)")
        else:
            device = "cpu"
            logger.info("使用CPU设备")
        
        return device
    
    def _get_torch_dtype(self, dtype_str: str) -> torch.dtype:
        """获取torch数据类型"""
        dtype_mapping = {
            "auto": torch.float16 if self.device != "cpu" else torch.float32,
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        return dtype_mapping.get(dtype_str, torch.float16)
    
    def load_model(self, model_path: str, config: Dict[str, Any]) -> None:
        """
        加载PyTorch模型和tokenizer
        
        Args:
            model_path: 模型文件路径
            config: 模型配置参数
            
        Raises:
            ModelLoadError: 模型加载失败时抛出
        """
        try:
            logger.info(f"开始加载PyTorch模型: {model_path}")
            
            self.model_path = model_path
            self.config = config
            self.device = self._detect_device()
            
            # 获取torch数据类型
            torch_dtype = self._get_torch_dtype(config.get("torch_dtype", "auto"))
            
            # 加载tokenizer
            logger.info("加载tokenizer...")
            
            # 检查是否是本地路径
            import os
            if os.path.exists(model_path) and os.path.isdir(model_path):
                # 本地路径，使用绝对路径
                abs_model_path = os.path.abspath(model_path)
                logger.info(f"使用本地模型路径: {abs_model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    abs_model_path,
                    trust_remote_code=True,
                    use_fast=True,
                    local_files_only=True,  # 强制只使用本地文件
                )
            else:
                # Hugging Face 模型 ID
                logger.info(f"使用 Hugging Face 模型: {model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    use_fast=True,
                )
            
            # 设置pad_token如果不存在
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
            # 加载模型
            logger.info("加载模型...")
            model_kwargs = {
                "torch_dtype": torch_dtype,
                "trust_remote_code": True,
                "low_cpu_mem_usage": config.get("low_cpu_mem_usage", True),
            }
            
            # 如果使用GPU，添加device_map
            if self.device != "cpu":
                model_kwargs["device_map"] = "auto"
            
            # 检查是否是本地路径
            if os.path.exists(model_path) and os.path.isdir(model_path):
                # 本地路径，使用绝对路径
                abs_model_path = os.path.abspath(model_path)
                model_kwargs["local_files_only"] = True  # 强制只使用本地文件
                self.model = AutoModelForCausalLM.from_pretrained(
                    abs_model_path,
                    **model_kwargs
                )
            else:
                # Hugging Face 模型 ID
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    **model_kwargs
                )
            
            # 如果是CPU，手动移动模型到设备
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            # 设置为评估模式
            self.model.eval()
            
            self.is_loaded = True
            logger.info(f"模型加载成功，设备: {self.device}, 数据类型: {torch_dtype}")
            
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            self._cleanup_resources()
            raise ModelLoadError(f"无法加载PyTorch模型 {model_path}: {str(e)}")
    
    def generate_stream(self, prompt: str) -> Iterator[str]:
        """
        流式生成文本
        
        Args:
            prompt: 输入提示词
            
        Yields:
            str: 生成的文本片段
            
        Raises:
            InferenceError: 推理过程中出现错误时抛出
        """
        if not self.is_loaded or self.model is None or self.tokenizer is None:
            raise InferenceError("模型未加载，无法进行推理")
        
        # 初始化统计信息
        self._current_stats = InferenceStats(
            start_time=time.time(),
            end_time=None,
            token_count=0,
            tokens_per_second=None
        )
        
        try:
            logger.info("开始流式推理...")
            
            # 编码输入
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            inputs = inputs.to(self.device)
            input_length = inputs.shape[1]
            
            # 创建生成配置
            generation_config = GenerationConfig(
                temperature=self.config.get("temperature", 0.7),
                top_p=self.config.get("top_p", 0.9),
                max_new_tokens=self.config.get("max_tokens", 512),
                do_sample=self.config.get("do_sample", True),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
            
            # 创建流式输出器
            streamer = TextIteratorStreamer(
                self.tokenizer,
                timeout=60.0,
                skip_prompt=True,
                skip_special_tokens=True
            )
            
            # 准备生成参数
            generate_kwargs = {
                "input_ids": inputs,
                "generation_config": generation_config,
                "streamer": streamer,
            }
            
            # 在单独线程中运行生成
            generation_thread = Thread(
                target=self.model.generate,
                kwargs=generate_kwargs
            )
            generation_thread.start()
            
            # 流式输出生成的文本
            generated_text = ""
            generation_interrupted = False
            
            try:
                for new_text in streamer:
                    if new_text:
                        generated_text += new_text
                        self._current_stats.token_count += 1  # 简化的token计数
                        yield new_text
                        
            except KeyboardInterrupt:
                logger.info("生成过程被用户中断")
                generation_interrupted = True
                raise InferenceError("生成过程被用户中断")
            except Exception as e:
                logger.error(f"流式生成过程中出现错误: {str(e)}")
                generation_interrupted = True
                raise InferenceError(f"流式生成失败: {str(e)}")
            
            # 等待生成线程完成
            if not generation_interrupted:
                generation_thread.join(timeout=120.0)
                if generation_thread.is_alive():
                    logger.warning("生成线程超时，可能存在问题")
            
            # 完成统计信息
            self._current_stats.end_time = time.time()
            duration = self._current_stats.end_time - self._current_stats.start_time
            
            if duration > 0:
                self._current_stats.tokens_per_second = self._current_stats.token_count / duration
            
            logger.info(f"推理完成 - 用时: {duration:.2f}s, Token数: {self._current_stats.token_count}, 速度: {self._current_stats.tokens_per_second:.2f} tokens/s")
            
        except Exception as e:
            # 确保统计信息被正确设置，即使出现错误
            if hasattr(self, '_current_stats'):
                self._current_stats.end_time = time.time()
            
            logger.error(f"推理过程中出现错误: {str(e)}")
            if not isinstance(e, InferenceError):
                raise InferenceError(f"PyTorch推理失败: {str(e)}")
            raise
    
    def generate_complete(self, prompt: str) -> str:
        """
        非流式生成完整文本
        
        Args:
            prompt: 输入提示词
            
        Returns:
            str: 完整的生成文本
            
        Raises:
            InferenceError: 推理过程中出现错误时抛出
        """
        if not self.is_loaded or self.model is None or self.tokenizer is None:
            raise InferenceError("模型未加载，无法进行推理")
        
        # 初始化统计信息
        self._current_stats = InferenceStats(
            start_time=time.time(),
            end_time=None,
            token_count=0,
            tokens_per_second=None
        )
        
        try:
            logger.info("开始非流式推理...")
            
            # 编码输入
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            inputs = inputs.to(self.device)
            input_length = inputs.shape[1]
            
            # 创建生成配置
            generation_config = GenerationConfig(
                temperature=self.config.get("temperature", 0.7),
                top_p=self.config.get("top_p", 0.9),
                max_new_tokens=self.config.get("max_tokens", 512),
                do_sample=self.config.get("do_sample", True),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
            
            # 直接生成完整结果
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=False,
                )
            
            # 解码生成的文本
            generated_ids = outputs.sequences[0][input_length:]  # 只取新生成的部分
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # 计算token数量
            self._current_stats.token_count = len(generated_ids)
            
            # 完成统计信息
            self._current_stats.end_time = time.time()
            duration = self._current_stats.end_time - self._current_stats.start_time
            
            if duration > 0:
                self._current_stats.tokens_per_second = self._current_stats.token_count / duration
            
            logger.info(f"非流式推理完成 - 用时: {duration:.2f}s, Token数: {self._current_stats.token_count}, 速度: {self._current_stats.tokens_per_second:.2f} tokens/s")
            
            return generated_text
            
        except Exception as e:
            # 确保统计信息被正确设置，即使出现错误
            if hasattr(self, '_current_stats'):
                self._current_stats.end_time = time.time()
            
            logger.error(f"非流式推理过程中出现错误: {str(e)}")
            if not isinstance(e, InferenceError):
                raise InferenceError(f"PyTorch非流式推理失败: {str(e)}")
            raise
    
    def get_inference_stats(self) -> Optional[InferenceStats]:
        """
        获取最近一次推理的统计信息
        
        Returns:
            Optional[InferenceStats]: 推理统计信息，如果没有进行过推理则返回None
        """
        return getattr(self, '_current_stats', None)
    
    def unload_model(self) -> None:
        """卸载模型并释放资源"""
        try:
            logger.info("开始卸载PyTorch模型...")
            
            self._cleanup_resources()
            
            # 强制垃圾回收
            gc.collect()
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # 清理MPS缓存
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            
            self.is_loaded = False
            logger.info("模型卸载完成")
            
        except Exception as e:
            logger.error(f"模型卸载过程中出现错误: {str(e)}")
    
    def _cleanup_resources(self) -> None:
        """清理模型资源"""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        self.device = None
        self.model_path = None
        self.config = None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict[str, Any]: 模型信息字典
        """
        info = {
            "type": "pytorch",
            "is_loaded": self.is_loaded,
            "model_path": self.model_path,
            "device": self.device,
        }
        
        if self.is_loaded and self.model is not None:
            try:
                # 获取模型参数数量
                param_count = sum(p.numel() for p in self.model.parameters())
                info.update({
                    "parameter_count": param_count,
                    "model_type": type(self.model).__name__,
                })
                
                # 获取内存使用情况
                if torch.cuda.is_available() and self.device == "cuda":
                    info["gpu_memory_allocated"] = torch.cuda.memory_allocated()
                    info["gpu_memory_reserved"] = torch.cuda.memory_reserved()
                
            except Exception as e:
                logger.warning(f"获取模型详细信息时出现错误: {str(e)}")
        
        return info