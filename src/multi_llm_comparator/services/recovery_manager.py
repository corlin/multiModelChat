"""
错误恢复管理器

提供自动错误恢复、重试机制和系统状态恢复功能。
"""

import time
import logging
from typing import Dict, Any, Optional, Callable, List, Tuple
from enum import Enum
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, Future
import threading

from ..core.exceptions import (
    MultiLLMComparatorError,
    ModelLoadError,
    InferenceError,
    MemoryError
)
from .error_handler import ErrorHandler, ErrorInfo, ErrorSeverity


class RecoveryStrategy(Enum):
    """恢复策略"""
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    ABORT = "abort"
    RESET = "reset"


class RecoveryAction(Enum):
    """恢复动作"""
    RELOAD_MODEL = "reload_model"
    CLEAR_MEMORY = "clear_memory"
    RESTART_INFERENCE = "restart_inference"
    USE_FALLBACK_MODEL = "use_fallback_model"
    REDUCE_PARAMETERS = "reduce_parameters"
    WAIT_AND_RETRY = "wait_and_retry"


@dataclass
class RecoveryPlan:
    """恢复计划"""
    strategy: RecoveryStrategy
    actions: List[RecoveryAction]
    max_attempts: int
    delay_seconds: float
    fallback_options: Optional[Dict[str, Any]] = None
    success_callback: Optional[Callable] = None
    failure_callback: Optional[Callable] = None


class RecoveryManager:
    """错误恢复管理器"""
    
    def __init__(self, error_handler: Optional[ErrorHandler] = None):
        """
        初始化恢复管理器
        
        Args:
            error_handler: 错误处理器实例
        """
        self.error_handler = error_handler or ErrorHandler()
        self.logger = logging.getLogger(__name__)
        self.recovery_plans: Dict[type, RecoveryPlan] = {}
        self.recovery_history: List[Dict[str, Any]] = []
        self.active_recoveries: Dict[str, Future] = {}
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.lock = threading.Lock()
        
        # 设置默认恢复计划
        self._setup_default_recovery_plans()
    
    def _setup_default_recovery_plans(self):
        """设置默认恢复计划"""
        # 模型加载错误恢复计划
        self.recovery_plans[ModelLoadError] = RecoveryPlan(
            strategy=RecoveryStrategy.RETRY,
            actions=[
                RecoveryAction.CLEAR_MEMORY,
                RecoveryAction.WAIT_AND_RETRY,
                RecoveryAction.RELOAD_MODEL
            ],
            max_attempts=3,
            delay_seconds=2.0
        )
        
        # 推理错误恢复计划
        self.recovery_plans[InferenceError] = RecoveryPlan(
            strategy=RecoveryStrategy.RETRY,
            actions=[
                RecoveryAction.WAIT_AND_RETRY,
                RecoveryAction.RESTART_INFERENCE,
                RecoveryAction.REDUCE_PARAMETERS
            ],
            max_attempts=2,
            delay_seconds=1.0
        )
        
        # 内存错误恢复计划
        self.recovery_plans[MemoryError] = RecoveryPlan(
            strategy=RecoveryStrategy.FALLBACK,
            actions=[
                RecoveryAction.CLEAR_MEMORY,
                RecoveryAction.REDUCE_PARAMETERS,
                RecoveryAction.USE_FALLBACK_MODEL
            ],
            max_attempts=2,
            delay_seconds=3.0
        )
        
        # 通用错误恢复计划
        self.recovery_plans[Exception] = RecoveryPlan(
            strategy=RecoveryStrategy.RETRY,
            actions=[
                RecoveryAction.WAIT_AND_RETRY
            ],
            max_attempts=1,
            delay_seconds=1.0
        )
    
    def register_recovery_plan(self, error_type: type, plan: RecoveryPlan):
        """
        注册恢复计划
        
        Args:
            error_type: 错误类型
            plan: 恢复计划
        """
        self.recovery_plans[error_type] = plan
        self.logger.info(f"已注册 {error_type.__name__} 的恢复计划")
    
    def attempt_recovery(
        self,
        error: Exception,
        operation: Callable,
        context: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> Tuple[bool, Any, Optional[Exception]]:
        """
        尝试错误恢复
        
        Args:
            error: 发生的错误
            operation: 要重试的操作
            context: 操作上下文
            progress_callback: 进度回调函数
            
        Returns:
            Tuple[bool, Any, Optional[Exception]]: (是否成功, 结果, 最后的错误)
        """
        # 查找恢复计划
        recovery_plan = self._find_recovery_plan(error)
        if not recovery_plan:
            self.logger.warning(f"未找到 {type(error).__name__} 的恢复计划")
            return False, None, error
        
        recovery_id = f"recovery_{int(time.time() * 1000)}"
        
        if progress_callback:
            progress_callback(f"开始错误恢复: {type(error).__name__}")
        
        # 记录恢复开始
        recovery_record = {
            'recovery_id': recovery_id,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'strategy': recovery_plan.strategy.value,
            'start_time': time.time(),
            'attempts': 0,
            'success': False,
            'context': context or {}
        }
        
        last_error = error
        
        try:
            for attempt in range(recovery_plan.max_attempts):
                recovery_record['attempts'] = attempt + 1
                
                if progress_callback:
                    progress_callback(f"恢复尝试 {attempt + 1}/{recovery_plan.max_attempts}")
                
                try:
                    # 执行恢复动作
                    self._execute_recovery_actions(
                        recovery_plan.actions,
                        context,
                        progress_callback
                    )
                    
                    # 等待延迟
                    if recovery_plan.delay_seconds > 0:
                        if progress_callback:
                            progress_callback(f"等待 {recovery_plan.delay_seconds} 秒后重试...")
                        time.sleep(recovery_plan.delay_seconds)
                    
                    # 重试操作
                    if progress_callback:
                        progress_callback("重试操作...")
                    
                    result = operation()
                    
                    # 成功恢复
                    recovery_record['success'] = True
                    recovery_record['end_time'] = time.time()
                    
                    if progress_callback:
                        progress_callback("恢复成功!")
                    
                    if recovery_plan.success_callback:
                        recovery_plan.success_callback(recovery_record)
                    
                    self.logger.info(f"错误恢复成功: {recovery_id}")
                    return True, result, None
                    
                except Exception as retry_error:
                    last_error = retry_error
                    self.logger.warning(f"恢复尝试 {attempt + 1} 失败: {retry_error}")
                    
                    if attempt < recovery_plan.max_attempts - 1:
                        if progress_callback:
                            progress_callback(f"尝试 {attempt + 1} 失败，准备下次尝试...")
            
            # 所有尝试都失败
            recovery_record['success'] = False
            recovery_record['end_time'] = time.time()
            recovery_record['final_error'] = str(last_error)
            
            if progress_callback:
                progress_callback("恢复失败")
            
            if recovery_plan.failure_callback:
                recovery_plan.failure_callback(recovery_record)
            
            self.logger.error(f"错误恢复失败: {recovery_id}")
            return False, None, last_error
            
        finally:
            # 记录恢复历史
            with self.lock:
                self.recovery_history.append(recovery_record)
                # 限制历史记录大小
                if len(self.recovery_history) > 100:
                    self.recovery_history = self.recovery_history[-100:]
    
    def _find_recovery_plan(self, error: Exception) -> Optional[RecoveryPlan]:
        """查找适合的恢复计划"""
        # 首先查找精确匹配
        for error_type, plan in self.recovery_plans.items():
            if type(error) == error_type:
                return plan
        
        # 然后查找继承匹配
        for error_type, plan in self.recovery_plans.items():
            if isinstance(error, error_type):
                return plan
        
        return None
    
    def _execute_recovery_actions(
        self,
        actions: List[RecoveryAction],
        context: Optional[Dict[str, Any]],
        progress_callback: Optional[Callable[[str], None]]
    ):
        """执行恢复动作"""
        for action in actions:
            if progress_callback:
                progress_callback(f"执行恢复动作: {action.value}")
            
            try:
                if action == RecoveryAction.CLEAR_MEMORY:
                    self._clear_memory(context)
                elif action == RecoveryAction.RELOAD_MODEL:
                    self._reload_model(context)
                elif action == RecoveryAction.RESTART_INFERENCE:
                    self._restart_inference(context)
                elif action == RecoveryAction.USE_FALLBACK_MODEL:
                    self._use_fallback_model(context)
                elif action == RecoveryAction.REDUCE_PARAMETERS:
                    self._reduce_parameters(context)
                elif action == RecoveryAction.WAIT_AND_RETRY:
                    # 这个动作在主循环中处理
                    pass
                else:
                    self.logger.warning(f"未知的恢复动作: {action}")
                    
            except Exception as action_error:
                self.logger.warning(f"执行恢复动作 {action.value} 失败: {action_error}")
    
    def _clear_memory(self, context: Optional[Dict[str, Any]]):
        """清理内存"""
        try:
            # 获取内存管理器
            memory_manager = context.get('memory_manager') if context else None
            if memory_manager and hasattr(memory_manager, 'force_cleanup'):
                memory_manager.force_cleanup()
                self.logger.info("已执行内存清理")
            else:
                # 执行Python垃圾回收
                import gc
                gc.collect()
                self.logger.info("已执行垃圾回收")
        except Exception as e:
            self.logger.warning(f"内存清理失败: {e}")
    
    def _reload_model(self, context: Optional[Dict[str, Any]]):
        """重新加载模型"""
        try:
            inferencer = context.get('inferencer') if context else None
            if inferencer and hasattr(inferencer, 'unload_model'):
                inferencer.unload_model()
                self.logger.info("已卸载模型，准备重新加载")
        except Exception as e:
            self.logger.warning(f"模型重新加载失败: {e}")
    
    def _restart_inference(self, context: Optional[Dict[str, Any]]):
        """重启推理"""
        try:
            inference_engine = context.get('inference_engine') if context else None
            if inference_engine and hasattr(inference_engine, 'cancel_all_tasks'):
                inference_engine.cancel_all_tasks()
                self.logger.info("已取消所有推理任务")
        except Exception as e:
            self.logger.warning(f"重启推理失败: {e}")
    
    def _use_fallback_model(self, context: Optional[Dict[str, Any]]):
        """使用备用模型"""
        try:
            # 这里可以实现切换到更小或更稳定的模型
            self.logger.info("准备使用备用模型")
            # 具体实现需要根据上下文中的模型管理器来处理
        except Exception as e:
            self.logger.warning(f"使用备用模型失败: {e}")
    
    def _reduce_parameters(self, context: Optional[Dict[str, Any]]):
        """减少参数以降低资源使用"""
        try:
            # 减少max_tokens, batch_size等参数
            if context and 'config' in context:
                config = context['config']
                if hasattr(config, 'max_tokens'):
                    config.max_tokens = min(config.max_tokens, 256)
                if hasattr(config, 'n_ctx'):
                    config.n_ctx = min(config.n_ctx, 1024)
                self.logger.info("已减少模型参数")
        except Exception as e:
            self.logger.warning(f"减少参数失败: {e}")
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """获取恢复统计信息"""
        if not self.recovery_history:
            return {
                'total_recoveries': 0,
                'successful_recoveries': 0,
                'failed_recoveries': 0,
                'success_rate': 0.0,
                'by_error_type': {},
                'by_strategy': {}
            }
        
        total = len(self.recovery_history)
        successful = sum(1 for r in self.recovery_history if r['success'])
        failed = total - successful
        success_rate = successful / total if total > 0 else 0.0
        
        # 按错误类型统计
        by_error_type = {}
        for record in self.recovery_history:
            error_type = record['error_type']
            if error_type not in by_error_type:
                by_error_type[error_type] = {'total': 0, 'successful': 0}
            by_error_type[error_type]['total'] += 1
            if record['success']:
                by_error_type[error_type]['successful'] += 1
        
        # 按策略统计
        by_strategy = {}
        for record in self.recovery_history:
            strategy = record['strategy']
            if strategy not in by_strategy:
                by_strategy[strategy] = {'total': 0, 'successful': 0}
            by_strategy[strategy]['total'] += 1
            if record['success']:
                by_strategy[strategy]['successful'] += 1
        
        return {
            'total_recoveries': total,
            'successful_recoveries': successful,
            'failed_recoveries': failed,
            'success_rate': success_rate,
            'by_error_type': by_error_type,
            'by_strategy': by_strategy
        }
    
    def clear_recovery_history(self):
        """清空恢复历史"""
        with self.lock:
            self.recovery_history.clear()
        self.logger.info("恢复历史已清空")
    
    def shutdown(self):
        """关闭恢复管理器"""
        try:
            self.executor.shutdown(wait=True, timeout=10)
        except Exception as e:
            self.logger.warning(f"关闭恢复管理器时出错: {e}")


# 全局恢复管理器实例
_global_recovery_manager: Optional[RecoveryManager] = None


def get_recovery_manager() -> RecoveryManager:
    """获取全局恢复管理器实例"""
    global _global_recovery_manager
    if _global_recovery_manager is None:
        _global_recovery_manager = RecoveryManager()
    return _global_recovery_manager


def attempt_recovery(
    error: Exception,
    operation: Callable,
    context: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[[str], None]] = None
) -> Tuple[bool, Any, Optional[Exception]]:
    """
    尝试错误恢复的便捷函数
    
    Args:
        error: 发生的错误
        operation: 要重试的操作
        context: 操作上下文
        progress_callback: 进度回调函数
        
    Returns:
        Tuple[bool, Any, Optional[Exception]]: (是否成功, 结果, 最后的错误)
    """
    return get_recovery_manager().attempt_recovery(error, operation, context, progress_callback)