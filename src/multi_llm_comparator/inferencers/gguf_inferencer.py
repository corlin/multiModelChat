"""
GGUF推理器

使用llama-cpp-python库实现GGUF模型的加载和推理。
"""

import gc
import logging
import time
from typing import Dict, Iterator, Any, Optional

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

from .base import BaseInferencer
from ..core.exceptions import ModelLoadError, InferenceError, MemoryError
from ..core.models import InferenceStats


logger = logging.getLogger(__name__)


class GGUFInferencer(BaseInferencer):
    """GGUF模型推理器"""
    
    def __init__(self):
        """初始化GGUF推理器"""
        super().__init__()
        self.llm: Optional[Llama] = None
        self.model_path: str = ""
        self.config: Dict[str, Any] = {}
        self._last_stats: Optional[InferenceStats] = None
        
        # 检查llama-cpp-python是否可用
        if Llama is None:
            raise ModelLoadError(
                "llama-cpp-python库未安装。请运行: pip install llama-cpp-python"
            )
    
    def load_model(self, model_path: str, config: Dict[str, Any]) -> None:
        """
        加载GGUF模型
        
        Args:
            model_path: 模型文件路径
            config: 模型配置参数
            
        Raises:
            ModelLoadError: 模型加载失败
        """
        try:
            logger.info(f"开始加载GGUF模型: {model_path}")
            
            # 如果已有模型加载，先卸载
            if self.is_loaded:
                self.unload_model()
            
            self.model_path = model_path
            self.config = config.copy()
            
            # 提取GGUF特定配置参数
            n_ctx = config.get('n_ctx', 2048)
            n_threads = config.get('n_threads', None)
            use_gpu = config.get('use_gpu', True)
            
            # 配置GPU层数
            n_gpu_layers = -1 if use_gpu else 0
            
            # 创建Llama实例
            self.llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
                verbose=False,  # 减少日志输出
                use_mmap=True,  # 使用内存映射提高加载速度
                use_mlock=False,  # 不锁定内存，允许系统管理
            )
            
            self.is_loaded = True
            logger.info(f"GGUF模型加载成功: {model_path}")
            
        except FileNotFoundError:
            error_msg = f"模型文件未找到: {model_path}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg)
        except Exception as e:
            # 检查是否是文件不存在的错误
            if "does not exist" in str(e) or "No such file" in str(e):
                error_msg = f"模型文件未找到: {model_path}"
                logger.error(error_msg)
                raise ModelLoadError(error_msg)
            else:
                error_msg = f"GGUF模型加载失败: {str(e)}"
                logger.error(error_msg)
                raise ModelLoadError(error_msg)
    
    def generate_stream(self, prompt: str) -> Iterator[str]:
        """
        流式生成文本
        
        Args:
            prompt: 输入提示词
            
        Yields:
            str: 生成的文本片段
            
        Raises:
            InferenceError: 推理过程出错
        """
        if not self.is_loaded or self.llm is None:
            raise InferenceError("模型未加载，无法进行推理")
        
        try:
            logger.info(f"开始GGUF流式推理，提示词长度: {len(prompt)}")
            start_time = time.time()
            token_count = 0
            generated_text = ""
            last_update_time = start_time
            
            # 提取生成参数
            temperature = self.config.get('temperature', 0.7)
            top_p = self.config.get('top_p', 0.9)
            top_k = self.config.get('top_k', 40)
            max_tokens = self.config.get('max_tokens', 512)
            repeat_penalty = self.config.get('repeat_penalty', 1.1)
            
            # 验证参数范围
            temperature = max(0.01, min(2.0, temperature))
            top_p = max(0.01, min(1.0, top_p))
            top_k = max(1, min(200, top_k))
            repeat_penalty = max(0.1, min(2.0, repeat_penalty))
            
            logger.info(f"推理参数: temperature={temperature}, top_p={top_p}, "
                       f"top_k={top_k}, max_tokens={max_tokens}, repeat_penalty={repeat_penalty}")
            
            # 使用llama-cpp-python的流式生成
            try:
                stream = self.llm(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repeat_penalty=repeat_penalty,
                    stream=True,
                    stop=["</s>", "<|endoftext|>", "<|im_end|>"],  # 常见的停止标记
                    echo=False,  # 不回显输入
                )
            except Exception as e:
                raise InferenceError(f"启动流式生成失败: {str(e)}")
            
            # 处理流式输出
            for output in stream:
                try:
                    current_time = time.time()
                    
                    # 检查输出格式
                    if not isinstance(output, dict):
                        logger.warning(f"意外的输出格式: {type(output)}")
                        continue
                    
                    if 'choices' in output and len(output['choices']) > 0:
                        choice = output['choices'][0]
                        
                        # 提取文本内容
                        if 'text' in choice:
                            text_chunk = choice['text']
                            if text_chunk:
                                generated_text += text_chunk
                                token_count += 1
                                
                                # 记录进度（每秒最多记录一次）
                                if current_time - last_update_time >= 1.0:
                                    elapsed = current_time - start_time
                                    speed = token_count / elapsed if elapsed > 0 else 0
                                    logger.debug(f"推理进度: {token_count} tokens, {speed:.2f} tokens/s")
                                    last_update_time = current_time
                                
                                yield text_chunk
                        
                        # 检查是否完成
                        finish_reason = choice.get('finish_reason')
                        if finish_reason is not None:
                            logger.info(f"推理完成，原因: {finish_reason}")
                            break
                    
                    # 检查是否有错误信息
                    if 'error' in output:
                        error_msg = f"推理过程中出现错误: {output['error']}"
                        logger.error(error_msg)
                        raise InferenceError(error_msg)
                        
                except KeyboardInterrupt:
                    logger.info("推理被用户中断")
                    break
                except InferenceError:
                    # 重新抛出推理错误，不要继续处理
                    raise
                except Exception as e:
                    logger.error(f"处理输出时出错: {str(e)}")
                    # 继续处理下一个输出，不中断整个流程
                    continue
            
            # 计算最终统计信息
            end_time = time.time()
            # 确保end_time > start_time，避免除零错误
            if end_time <= start_time:
                end_time = start_time + 0.001  # 添加1毫秒
            duration = end_time - start_time
            tokens_per_second = token_count / duration if duration > 0 else 0
            
            self._last_stats = InferenceStats(
                start_time=start_time,
                end_time=end_time,
                token_count=token_count,
                tokens_per_second=tokens_per_second
            )
            
            logger.info(f"GGUF推理完成: 生成{token_count}个token，用时{duration:.2f}秒，"
                       f"速度{tokens_per_second:.2f} tokens/s")
            
        except InferenceError:
            # 重新抛出推理错误
            raise
        except KeyboardInterrupt:
            logger.info("推理被用户中断")
            raise InferenceError("推理被用户中断")
        except Exception as e:
            error_msg = f"GGUF推理过程出错: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise InferenceError(error_msg)
    
    def generate_complete(self, prompt: str) -> str:
        """
        非流式生成完整文本
        
        Args:
            prompt: 输入提示词
            
        Returns:
            str: 完整的生成文本
            
        Raises:
            InferenceError: 推理过程出错
        """
        if not self.is_loaded or self.llm is None:
            raise InferenceError("模型未加载，无法进行推理")
        
        try:
            logger.info(f"开始GGUF非流式推理，提示词长度: {len(prompt)}")
            start_time = time.time()
            
            # 提取生成参数
            temperature = self.config.get('temperature', 0.7)
            top_p = self.config.get('top_p', 0.9)
            top_k = self.config.get('top_k', 40)
            max_tokens = self.config.get('max_tokens', 512)
            repeat_penalty = self.config.get('repeat_penalty', 1.1)
            
            # 验证参数范围
            temperature = max(0.01, min(2.0, temperature))
            top_p = max(0.01, min(1.0, top_p))
            top_k = max(1, min(200, top_k))
            repeat_penalty = max(0.1, min(2.0, repeat_penalty))
            
            logger.info(f"推理参数: temperature={temperature}, top_p={top_p}, "
                       f"top_k={top_k}, max_tokens={max_tokens}, repeat_penalty={repeat_penalty}")
            
            # 使用llama-cpp-python的非流式生成
            try:
                result = self.llm(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repeat_penalty=repeat_penalty,
                    stream=False,  # 非流式
                    stop=["</s>", "<|endoftext|>", "<|im_end|>"],  # 常见的停止标记
                    echo=False,  # 不回显输入
                )
            except Exception as e:
                raise InferenceError(f"启动非流式生成失败: {str(e)}")
            
            # 提取生成的文本
            generated_text = ""
            token_count = 0
            
            if isinstance(result, dict) and 'choices' in result and len(result['choices']) > 0:
                choice = result['choices'][0]
                if 'text' in choice:
                    generated_text = choice['text']
                    # 估算token数量（简化计算）
                    token_count = len(generated_text.split())
                
                # 检查是否有错误
                if 'error' in result:
                    error_msg = f"推理过程中出现错误: {result['error']}"
                    logger.error(error_msg)
                    raise InferenceError(error_msg)
            else:
                logger.warning(f"意外的输出格式: {type(result)}")
                if hasattr(result, 'get'):
                    generated_text = str(result.get('text', ''))
                else:
                    generated_text = str(result)
                token_count = len(generated_text.split())
            
            # 计算最终统计信息
            end_time = time.time()
            # 确保end_time > start_time，避免除零错误
            if end_time <= start_time:
                end_time = start_time + 0.001  # 添加1毫秒
            duration = end_time - start_time
            tokens_per_second = token_count / duration if duration > 0 else 0
            
            self._last_stats = InferenceStats(
                start_time=start_time,
                end_time=end_time,
                token_count=token_count,
                tokens_per_second=tokens_per_second
            )
            
            logger.info(f"GGUF非流式推理完成: 生成{token_count}个token，用时{duration:.2f}秒，"
                       f"速度{tokens_per_second:.2f} tokens/s")
            
            return generated_text
            
        except InferenceError:
            # 重新抛出推理错误
            raise
        except KeyboardInterrupt:
            logger.info("推理被用户中断")
            raise InferenceError("推理被用户中断")
        except Exception as e:
            error_msg = f"GGUF非流式推理过程出错: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise InferenceError(error_msg)
    
    def unload_model(self) -> None:
        """
        卸载模型并释放内存
        
        Raises:
            MemoryError: 内存释放失败
        """
        try:
            logger.info("开始卸载GGUF模型")
            
            if self.llm is not None:
                # llama-cpp-python没有显式的close方法，直接删除引用
                del self.llm
                self.llm = None
            
            self.is_loaded = False
            self.model_path = ""
            self.config = {}
            self._last_stats = None
            
            # 强制垃圾回收
            gc.collect()
            
            logger.info("GGUF模型卸载完成")
            
        except Exception as e:
            error_msg = f"GGUF模型卸载失败: {str(e)}"
            logger.error(error_msg)
            raise MemoryError(error_msg)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict[str, Any]: 模型信息字典
        """
        info = {
            "type": "GGUF",
            "path": self.model_path,
            "is_loaded": self.is_loaded,
            "config": self.config.copy(),
        }
        
        if self.llm is not None:
            try:
                # 获取模型的上下文长度等信息
                info.update({
                    "n_ctx": self.llm.n_ctx(),
                    "n_vocab": self.llm.n_vocab(),
                })
            except Exception as e:
                logger.warning(f"获取GGUF模型详细信息失败: {str(e)}")
        
        return info
    
    def get_inference_stats(self) -> Optional[InferenceStats]:
        """
        获取最近一次推理的统计信息
        
        Returns:
            Optional[InferenceStats]: 推理统计信息
        """
        return self._last_stats
    
    def estimate_memory_usage(self) -> Dict[str, Any]:
        """
        估算模型内存使用情况
        
        Returns:
            Dict[str, Any]: 内存使用信息
        """
        memory_info = {
            "model_loaded": self.is_loaded,
            "estimated_vram_mb": 0,
            "estimated_ram_mb": 0,
        }
        
        if self.is_loaded and self.llm is not None:
            try:
                # 尝试获取模型大小信息
                import os
                if os.path.exists(self.model_path):
                    file_size_mb = os.path.getsize(self.model_path) / (1024 * 1024)
                    memory_info["model_file_size_mb"] = file_size_mb
                    
                    # 估算内存使用（通常比文件大小稍大）
                    use_gpu = self.config.get('use_gpu', True)
                    if use_gpu:
                        memory_info["estimated_vram_mb"] = file_size_mb * 1.2
                    else:
                        memory_info["estimated_ram_mb"] = file_size_mb * 1.2
                        
            except Exception as e:
                logger.warning(f"获取内存使用信息失败: {str(e)}")
        
        return memory_info
    
    def cancel_inference(self) -> None:
        """
        取消当前推理（如果支持的话）
        
        注意：llama-cpp-python可能不支持中途取消，此方法主要用于接口一致性
        """
        logger.info("收到取消推理请求")
        # llama-cpp-python的流式生成通常通过异常或break来中断
        # 实际的取消需要在调用方通过停止迭代来实现
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, str]:
        """
        验证GGUF特定的配置参数
        
        Args:
            config: 配置参数字典
            
        Returns:
            Dict[str, str]: 验证错误信息，键为参数名，值为错误描述
        """
        errors = {}
        
        # 验证n_ctx
        n_ctx = config.get('n_ctx', 2048)
        if not isinstance(n_ctx, int) or n_ctx < 1 or n_ctx > 32768:
            errors['n_ctx'] = "上下文长度必须是1-32768之间的整数"
        
        # 验证n_threads
        n_threads = config.get('n_threads')
        if n_threads is not None and (not isinstance(n_threads, int) or n_threads < 1):
            errors['n_threads'] = "线程数必须是正整数"
        
        # 验证top_k
        top_k = config.get('top_k', 40)
        if not isinstance(top_k, int) or top_k < 1 or top_k > 200:
            errors['top_k'] = "top_k必须是1-200之间的整数"
        
        # 验证repeat_penalty
        repeat_penalty = config.get('repeat_penalty', 1.1)
        if not isinstance(repeat_penalty, (int, float)) or repeat_penalty < 0.1 or repeat_penalty > 2.0:
            errors['repeat_penalty'] = "重复惩罚必须是0.1-2.0之间的数值"
        
        return errors