"""
推理引擎协调器

协调多模型推理，实现顺序处理、结果聚合、错误恢复和资源管理。
"""

import logging
import threading
import time
import uuid
from typing import Iterator, List, Optional, Dict, Any, Callable
from concurrent.futures import ThreadPoolExecutor, Future
from contextlib import contextmanager

from ..core.models import ModelInfo, InferenceResult, InferenceStats, ModelType
from ..core.exceptions import (
    InferenceError, 
    ModelLoadError, 
    MemoryError,
    MultiLLMComparatorError
)
from ..inferencers.base import BaseInferencer
from ..inferencers.pytorch_inferencer import PyTorchInferencer
from ..inferencers.gguf_inferencer import GGUFInferencer
from ..services.memory_manager import MemoryManager
from ..services.error_handler import handle_error
from ..services.recovery_manager import attempt_recovery

logger = logging.getLogger(__name__)


class InferenceTask:
    """推理任务封装"""
    
    def __init__(self, model_info: ModelInfo, prompt: str, task_id: str):
        self.model_info = model_info
        self.prompt = prompt
        self.task_id = task_id
        self.status = "pending"  # pending, running, completed, failed, cancelled
        self.result: Optional[InferenceResult] = None
        self.error: Optional[str] = None
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.cancel_event = threading.Event()
        
    def cancel(self):
        """取消任务"""
        self.cancel_event.set()
        self.status = "cancelled"
        
    def is_cancelled(self) -> bool:
        """检查任务是否被取消"""
        return self.cancel_event.is_set()


class InferenceEngine:
    """推理引擎协调器"""
    
    def __init__(self, memory_manager: Optional[MemoryManager] = None):
        """
        初始化推理引擎
        
        Args:
            memory_manager: 内存管理器实例，如果为None则创建新实例
        """
        self.memory_manager = memory_manager or MemoryManager()
        self.current_inferencer: Optional[BaseInferencer] = None
        self.active_tasks: Dict[str, InferenceTask] = {}
        self.executor = ThreadPoolExecutor(max_workers=1)  # 顺序处理，使用单线程
        self.lock = threading.Lock()
        self.max_retries = 3
        self.retry_delay = 1.0  # 重试延迟（秒）
        
    def create_inferencer(self, model_info: ModelInfo) -> BaseInferencer:
        """
        根据模型类型创建推理器
        
        Args:
            model_info: 模型信息
            
        Returns:
            BaseInferencer: 对应类型的推理器实例
            
        Raises:
            InferenceError: 不支持的模型类型
        """
        if model_info.model_type == ModelType.PYTORCH:
            return PyTorchInferencer()
        elif model_info.model_type == ModelType.GGUF:
            return GGUFInferencer()
        else:
            raise InferenceError(f"不支持的模型类型: {model_info.model_type}")
    
    def run_inference(
        self, 
        prompt: str, 
        models: List[ModelInfo],
        progress_callback: Optional[Callable[[str, str, str], None]] = None,
        streaming: bool = True
    ) -> Iterator[InferenceResult]:
        """
        运行多模型推理
        
        Args:
            prompt: 输入提示词
            models: 模型列表
            progress_callback: 进度回调函数，参数为(task_id, model_id, status)
            streaming: 是否使用流式输出，默认为True
            
        Yields:
            InferenceResult: 推理结果
        """
        if not models:
            raise InferenceError("模型列表不能为空")
        
        # 创建推理任务
        tasks = []
        for model in models:
            task_id = str(uuid.uuid4())
            task = InferenceTask(model, prompt, task_id)
            tasks.append(task)
            self.active_tasks[task_id] = task
        
        try:
            # 顺序处理每个模型
            for task in tasks:
                if task.is_cancelled():
                    continue
                
                try:
                    # 执行推理任务
                    result = self._execute_task_with_retry(task, progress_callback, streaming)
                    if result:
                        yield result
                        
                except Exception as e:
                    logger.error(f"任务 {task.task_id} 执行失败: {e}")
                    # 创建错误结果
                    error_result = InferenceResult(
                        model_id=task.model_info.id,
                        content="",
                        is_complete=True,
                        error=str(e),
                        stats=InferenceStats(
                            start_time=task.start_time or time.time(),
                            end_time=time.time(),
                            token_count=0,
                            tokens_per_second=0.0
                        )
                    )
                    yield error_result
                
                finally:
                    # 清理资源
                    self._cleanup_task(task)
                    
        finally:
            # 清理所有任务
            for task in tasks:
                if task.task_id in self.active_tasks:
                    del self.active_tasks[task.task_id]
    
    def _execute_task_with_retry(
        self, 
        task: InferenceTask,
        progress_callback: Optional[Callable[[str, str, str], None]] = None,
        streaming: bool = True
    ) -> Optional[InferenceResult]:
        """
        执行任务并支持重试
        
        Args:
            task: 推理任务
            progress_callback: 进度回调函数
            streaming: 是否使用流式输出
            
        Returns:
            Optional[InferenceResult]: 推理结果，失败时返回None
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            if task.is_cancelled():
                return None
                
            try:
                if progress_callback:
                    progress_callback(task.task_id, task.model_info.id, f"尝试 {attempt + 1}/{self.max_retries}")
                
                return self._execute_single_task(task, progress_callback, streaming)
                
            except Exception as e:
                last_error = e
                logger.warning(f"任务 {task.task_id} 第 {attempt + 1} 次尝试失败: {e}")
                
                if attempt < self.max_retries - 1:
                    # 等待后重试
                    time.sleep(self.retry_delay * (attempt + 1))
                    # 清理当前推理器
                    self._cleanup_current_inferencer()
        
        # 所有重试都失败
        task.status = "failed"
        task.error = str(last_error)
        raise InferenceError(f"任务执行失败，已重试 {self.max_retries} 次: {last_error}")
    
    def _execute_single_task(
        self, 
        task: InferenceTask,
        progress_callback: Optional[Callable[[str, str, str], None]] = None,
        streaming: bool = True
    ) -> InferenceResult:
        """
        执行单个推理任务
        
        Args:
            task: 推理任务
            progress_callback: 进度回调函数
            streaming: 是否使用流式输出
            
        Returns:
            InferenceResult: 推理结果
        """
        task.status = "running"
        task.start_time = time.time()
        
        if progress_callback:
            progress_callback(task.task_id, task.model_info.id, "加载模型")
        
        # 创建并加载推理器
        inferencer = None
        
        try:
            inferencer = self.create_inferencer(task.model_info)
            
            # 检查内存可用性
            if not self.memory_manager.check_memory_availability(1024):  # 至少需要1GB
                self.memory_manager.force_cleanup()
                if not self.memory_manager.check_memory_availability(512):  # 最低512MB
                    error_context = {
                        'model_id': task.model_info.id,
                        'model_path': task.model_info.path,
                        'memory_manager': self.memory_manager
                    }
                    memory_error = MemoryError("内存不足，无法加载模型")
                    handle_error(memory_error, error_context)
                    raise memory_error
            
            # 加载模型 - 使用错误恢复机制
            def load_model_operation():
                inferencer.load_model(task.model_info.path, task.model_info.config)
                return inferencer
            
            context = {
                'model_id': task.model_info.id,
                'model_path': task.model_info.path,
                'inferencer': inferencer,
                'memory_manager': self.memory_manager,
                'config': task.model_info.config
            }
            
            try:
                inferencer = load_model_operation()
                self.current_inferencer = inferencer
            except Exception as load_error:
                # 尝试错误恢复
                success, recovered_inferencer, final_error = attempt_recovery(
                    load_error,
                    load_model_operation,
                    context,
                    lambda msg: progress_callback(task.task_id, task.model_info.id, f"恢复中: {msg}") if progress_callback else None
                )
                
                if success:
                    inferencer = recovered_inferencer
                    self.current_inferencer = inferencer
                else:
                    # 记录错误并重新抛出
                    error_context = context.copy()
                    error_context['recovery_attempted'] = True
                    handle_error(final_error, error_context)
                    raise final_error
            
            if progress_callback:
                progress_callback(task.task_id, task.model_info.id, "生成中")
            
            # 执行推理 - 使用错误恢复机制
            def inference_operation():
                if streaming:
                    # 流式推理
                    content_parts = []
                    token_count = 0
                    
                    for token in inferencer.generate_stream(task.prompt):
                        if task.is_cancelled():
                            break
                            
                        content_parts.append(token)
                        token_count += 1
                        
                        # 流式进度更新 - 每个token都发送更新以实现真正的实时显示
                        if progress_callback:
                            current_content = "".join(content_parts)
                            progress_callback(task.task_id, task.model_info.id, f"streaming_update:{current_content}:{token_count}")
                    
                    # 确保发送最终的流式更新
                    if progress_callback and token_count > 0:
                        final_content = "".join(content_parts)
                        progress_callback(task.task_id, task.model_info.id, f"streaming_final:{final_content}:{token_count}")
                    
                    return "".join(content_parts), token_count
                else:
                    # 非流式推理
                    complete_text = inferencer.generate_complete(task.prompt)
                    # 估算token数量（简化计算）
                    token_count = len(complete_text.split())
                    return complete_text, token_count
            
            try:
                generated_text, token_count = inference_operation()
            except Exception as inference_error:
                # 尝试推理错误恢复
                inference_context = context.copy()
                inference_context['prompt'] = task.prompt
                inference_context['inference_engine'] = self
                
                success, result, final_error = attempt_recovery(
                    inference_error,
                    inference_operation,
                    inference_context,
                    lambda msg: progress_callback(task.task_id, task.model_info.id, f"推理恢复: {msg}") if progress_callback else None
                )
                
                if success:
                    generated_text, token_count = result
                else:
                    # 记录错误并创建错误结果
                    error_context = inference_context.copy()
                    error_context['recovery_attempted'] = True
                    handle_error(final_error, error_context)
                    
                    # 返回部分结果（如果有的话）
                    task.end_time = time.time()
                    task.status = "failed"
                    task.error = str(final_error)
                    
                    stats = InferenceStats(
                        start_time=task.start_time,
                        end_time=task.end_time,
                        token_count=0,
                        tokens_per_second=0.0
                    )
                    
                    return InferenceResult(
                        model_id=task.model_info.id,
                        content="",
                        is_complete=False,
                        error=str(final_error),
                        stats=stats
                    )
            
            task.end_time = time.time()
            task.status = "completed" if not task.is_cancelled() else "cancelled"
            
            # 计算统计信息
            duration = task.end_time - task.start_time
            tokens_per_second = token_count / duration if duration > 0 else 0.0
            
            stats = InferenceStats(
                start_time=task.start_time,
                end_time=task.end_time,
                token_count=token_count,
                tokens_per_second=tokens_per_second
            )
            
            # 创建结果
            result = InferenceResult(
                model_id=task.model_info.id,
                content=generated_text,
                is_complete=not task.is_cancelled(),
                error=None,
                stats=stats
            )
            
            task.result = result
            
            if progress_callback:
                status = "完成" if not task.is_cancelled() else "已取消"
                progress_callback(task.task_id, task.model_info.id, status)
            
            return result
            
        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            task.end_time = time.time()
            
            # 记录未处理的错误
            error_context = {
                'model_id': task.model_info.id,
                'model_path': task.model_info.path,
                'task_id': task.task_id
            }
            handle_error(e, error_context)
            raise
            
        finally:
            # 卸载模型并清理资源
            if inferencer:
                try:
                    inferencer.unload_model()
                except Exception as e:
                    logger.warning(f"卸载模型时出错: {e}")
                    # 记录卸载错误但不抛出
                    handle_error(e, {'model_id': task.model_info.id, 'operation': 'unload_model'})
            
            self.current_inferencer = None
            
            try:
                self.memory_manager.force_cleanup()
            except Exception as e:
                logger.warning(f"内存清理时出错: {e}")
                handle_error(e, {'operation': 'memory_cleanup'})
    
    def _cleanup_task(self, task: InferenceTask):
        """
        清理任务资源
        
        Args:
            task: 要清理的任务
        """
        try:
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
        except Exception as e:
            logger.warning(f"清理任务 {task.task_id} 时出错: {e}")
    
    def _cleanup_current_inferencer(self):
        """清理当前推理器"""
        if self.current_inferencer:
            try:
                self.current_inferencer.unload_model()
            except Exception as e:
                logger.warning(f"清理推理器时出错: {e}")
            finally:
                self.current_inferencer = None
                self.memory_manager.force_cleanup()
    
    def cancel_all_tasks(self):
        """取消所有活动任务"""
        with self.lock:
            for task in self.active_tasks.values():
                task.cancel()
            
            # 清理当前推理器
            self._cleanup_current_inferencer()
    
    def cancel_task(self, task_id: str) -> bool:
        """
        取消指定任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            bool: 是否成功取消
        """
        with self.lock:
            if task_id in self.active_tasks:
                self.active_tasks[task_id].cancel()
                return True
            return False
    
    def get_task_status(self, task_id: str) -> Optional[str]:
        """
        获取任务状态
        
        Args:
            task_id: 任务ID
            
        Returns:
            Optional[str]: 任务状态，如果任务不存在返回None
        """
        task = self.active_tasks.get(task_id)
        return task.status if task else None
    
    def get_active_tasks(self) -> List[str]:
        """
        获取所有活动任务ID
        
        Returns:
            List[str]: 活动任务ID列表
        """
        return list(self.active_tasks.keys())
    
    def run_non_streaming_inference(
        self, 
        prompt: str, 
        models: List[ModelInfo],
        progress_callback: Optional[Callable[[str, str, str], None]] = None
    ) -> Iterator[InferenceResult]:
        """
        运行非流式多模型推理（优化版本）
        
        Args:
            prompt: 输入提示词
            models: 模型列表
            progress_callback: 进度回调函数，参数为(task_id, model_id, status)
            
        Yields:
            InferenceResult: 推理结果
        """
        return self.run_inference(prompt, models, progress_callback, streaming=False)
    
    def cleanup_resources(self):
        """清理所有资源"""
        self.cancel_all_tasks()
        
        # 关闭线程池
        try:
            self.executor.shutdown(wait=True, timeout=10)
        except Exception as e:
            logger.warning(f"关闭线程池时出错: {e}")
        
        # 最终内存清理
        self.memory_manager.force_cleanup()
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.cleanup_resources()