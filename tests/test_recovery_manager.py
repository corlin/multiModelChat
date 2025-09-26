"""
错误恢复管理器测试

测试自动错误恢复、重试机制和系统状态恢复功能。
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock

from src.multi_llm_comparator.services.recovery_manager import (
    RecoveryManager,
    RecoveryStrategy,
    RecoveryAction,
    RecoveryPlan,
    get_recovery_manager,
    attempt_recovery
)
from src.multi_llm_comparator.services.error_handler import ErrorHandler
from src.multi_llm_comparator.core.exceptions import (
    ModelLoadError,
    InferenceError,
    MemoryError
)


class TestRecoveryPlan:
    """恢复计划测试类"""
    
    def test_recovery_plan_creation(self):
        """测试恢复计划创建"""
        plan = RecoveryPlan(
            strategy=RecoveryStrategy.RETRY,
            actions=[RecoveryAction.CLEAR_MEMORY, RecoveryAction.RELOAD_MODEL],
            max_attempts=3,
            delay_seconds=2.0,
            fallback_options={"test": "option"}
        )
        
        assert plan.strategy == RecoveryStrategy.RETRY
        assert len(plan.actions) == 2
        assert RecoveryAction.CLEAR_MEMORY in plan.actions
        assert RecoveryAction.RELOAD_MODEL in plan.actions
        assert plan.max_attempts == 3
        assert plan.delay_seconds == 2.0
        assert plan.fallback_options == {"test": "option"}


class TestRecoveryManager:
    """恢复管理器测试类"""
    
    def setup_method(self):
        """测试前设置"""
        self.error_handler = Mock(spec=ErrorHandler)
        self.recovery_manager = RecoveryManager(self.error_handler)
    
    def teardown_method(self):
        """测试后清理"""
        self.recovery_manager.shutdown()
    
    def test_recovery_manager_initialization(self):
        """测试恢复管理器初始化"""
        assert self.recovery_manager.error_handler == self.error_handler
        assert len(self.recovery_manager.recovery_plans) > 0
        assert len(self.recovery_manager.recovery_history) == 0
        
        # 检查默认恢复计划
        assert ModelLoadError in self.recovery_manager.recovery_plans
        assert InferenceError in self.recovery_manager.recovery_plans
        assert MemoryError in self.recovery_manager.recovery_plans
        assert Exception in self.recovery_manager.recovery_plans
    
    def test_register_recovery_plan(self):
        """测试注册恢复计划"""
        custom_plan = RecoveryPlan(
            strategy=RecoveryStrategy.FALLBACK,
            actions=[RecoveryAction.USE_FALLBACK_MODEL],
            max_attempts=1,
            delay_seconds=0.5
        )
        
        class CustomError(Exception):
            pass
        
        self.recovery_manager.register_recovery_plan(CustomError, custom_plan)
        
        assert CustomError in self.recovery_manager.recovery_plans
        assert self.recovery_manager.recovery_plans[CustomError] == custom_plan
    
    def test_find_recovery_plan_exact_match(self):
        """测试查找恢复计划 - 精确匹配"""
        error = ModelLoadError("测试错误")
        plan = self.recovery_manager._find_recovery_plan(error)
        
        assert plan is not None
        assert plan == self.recovery_manager.recovery_plans[ModelLoadError]
    
    def test_find_recovery_plan_inheritance_match(self):
        """测试查找恢复计划 - 继承匹配"""
        class CustomModelLoadError(ModelLoadError):
            pass
        
        error = CustomModelLoadError("测试错误")
        plan = self.recovery_manager._find_recovery_plan(error)
        
        assert plan is not None
        assert plan == self.recovery_manager.recovery_plans[ModelLoadError]
    
    def test_find_recovery_plan_generic_match(self):
        """测试查找恢复计划 - 通用匹配"""
        class UnknownError(Exception):
            pass
        
        error = UnknownError("未知错误")
        plan = self.recovery_manager._find_recovery_plan(error)
        
        assert plan is not None
        assert plan == self.recovery_manager.recovery_plans[Exception]
    
    def test_successful_recovery(self):
        """测试成功恢复"""
        # 模拟一个会成功的操作
        operation = Mock(return_value="成功结果")
        error = ModelLoadError("测试错误")
        
        success, result, final_error = self.recovery_manager.attempt_recovery(
            error, operation
        )
        
        assert success is True
        assert result == "成功结果"
        assert final_error is None
        assert len(self.recovery_manager.recovery_history) == 1
        assert self.recovery_manager.recovery_history[0]['success'] is True
    
    def test_failed_recovery_all_attempts(self):
        """测试恢复失败 - 所有尝试都失败"""
        # 模拟一个总是失败的操作
        operation = Mock(side_effect=RuntimeError("操作失败"))
        error = ModelLoadError("测试错误")
        
        success, result, final_error = self.recovery_manager.attempt_recovery(
            error, operation
        )
        
        assert success is False
        assert result is None
        assert final_error is not None
        assert isinstance(final_error, RuntimeError)
        assert len(self.recovery_manager.recovery_history) == 1
        assert self.recovery_manager.recovery_history[0]['success'] is False
        
        # 检查尝试次数
        plan = self.recovery_manager.recovery_plans[ModelLoadError]
        assert self.recovery_manager.recovery_history[0]['attempts'] == plan.max_attempts
    
    def test_recovery_with_progress_callback(self):
        """测试带进度回调的恢复"""
        progress_messages = []
        
        def progress_callback(message):
            progress_messages.append(message)
        
        operation = Mock(return_value="成功")
        error = InferenceError("测试错误")
        
        success, result, final_error = self.recovery_manager.attempt_recovery(
            error, operation, progress_callback=progress_callback
        )
        
        assert success is True
        assert len(progress_messages) > 0
        assert any("开始错误恢复" in msg for msg in progress_messages)
        assert any("恢复成功" in msg for msg in progress_messages)
    
    def test_recovery_with_context(self):
        """测试带上下文的恢复"""
        context = {
            'model_id': 'test_model',
            'memory_manager': Mock(),
            'inferencer': Mock()
        }
        
        operation = Mock(return_value="成功")
        error = MemoryError("内存不足")
        
        success, result, final_error = self.recovery_manager.attempt_recovery(
            error, operation, context
        )
        
        assert success is True
        assert len(self.recovery_manager.recovery_history) == 1
        assert self.recovery_manager.recovery_history[0]['context'] == context
    
    @patch('time.sleep')
    def test_recovery_delay(self, mock_sleep):
        """测试恢复延迟"""
        operation = Mock(return_value="成功")
        error = ModelLoadError("测试错误")
        
        # 获取恢复计划的延迟时间
        plan = self.recovery_manager.recovery_plans[ModelLoadError]
        expected_delay = plan.delay_seconds
        
        success, result, final_error = self.recovery_manager.attempt_recovery(
            error, operation
        )
        
        assert success is True
        # 验证sleep被调用了正确的次数和时间
        mock_sleep.assert_called_with(expected_delay)
    
    def test_clear_memory_action(self):
        """测试清理内存动作"""
        memory_manager = Mock()
        context = {'memory_manager': memory_manager}
        
        self.recovery_manager._clear_memory(context)
        
        memory_manager.force_cleanup.assert_called_once()
    
    def test_clear_memory_action_fallback(self):
        """测试清理内存动作 - 回退到垃圾回收"""
        with patch('gc.collect') as mock_gc:
            self.recovery_manager._clear_memory(None)
            mock_gc.assert_called_once()
    
    def test_reload_model_action(self):
        """测试重新加载模型动作"""
        inferencer = Mock()
        context = {'inferencer': inferencer}
        
        self.recovery_manager._reload_model(context)
        
        inferencer.unload_model.assert_called_once()
    
    def test_restart_inference_action(self):
        """测试重启推理动作"""
        inference_engine = Mock()
        context = {'inference_engine': inference_engine}
        
        self.recovery_manager._restart_inference(context)
        
        inference_engine.cancel_all_tasks.assert_called_once()
    
    def test_reduce_parameters_action(self):
        """测试减少参数动作"""
        config = Mock()
        config.max_tokens = 1000
        config.n_ctx = 2048
        context = {'config': config}
        
        self.recovery_manager._reduce_parameters(context)
        
        assert config.max_tokens <= 256
        assert config.n_ctx <= 1024
    
    def test_recovery_statistics_empty(self):
        """测试空恢复统计"""
        stats = self.recovery_manager.get_recovery_statistics()
        
        assert stats['total_recoveries'] == 0
        assert stats['successful_recoveries'] == 0
        assert stats['failed_recoveries'] == 0
        assert stats['success_rate'] == 0.0
        assert stats['by_error_type'] == {}
        assert stats['by_strategy'] == {}
    
    def test_recovery_statistics_with_data(self):
        """测试有数据的恢复统计"""
        # 添加一些恢复记录
        self.recovery_manager.recovery_history = [
            {
                'error_type': 'ModelLoadError',
                'strategy': 'retry',
                'success': True
            },
            {
                'error_type': 'ModelLoadError',
                'strategy': 'retry',
                'success': False
            },
            {
                'error_type': 'InferenceError',
                'strategy': 'fallback',
                'success': True
            }
        ]
        
        stats = self.recovery_manager.get_recovery_statistics()
        
        assert stats['total_recoveries'] == 3
        assert stats['successful_recoveries'] == 2
        assert stats['failed_recoveries'] == 1
        assert stats['success_rate'] == 2/3
        
        assert stats['by_error_type']['ModelLoadError']['total'] == 2
        assert stats['by_error_type']['ModelLoadError']['successful'] == 1
        assert stats['by_error_type']['InferenceError']['total'] == 1
        assert stats['by_error_type']['InferenceError']['successful'] == 1
        
        assert stats['by_strategy']['retry']['total'] == 2
        assert stats['by_strategy']['retry']['successful'] == 1
        assert stats['by_strategy']['fallback']['total'] == 1
        assert stats['by_strategy']['fallback']['successful'] == 1
    
    def test_clear_recovery_history(self):
        """测试清空恢复历史"""
        # 添加一些历史记录
        self.recovery_manager.recovery_history = [{'test': 'data'}]
        assert len(self.recovery_manager.recovery_history) == 1
        
        self.recovery_manager.clear_recovery_history()
        assert len(self.recovery_manager.recovery_history) == 0
    
    def test_recovery_plan_callbacks(self):
        """测试恢复计划回调"""
        success_callback = Mock()
        failure_callback = Mock()
        
        custom_plan = RecoveryPlan(
            strategy=RecoveryStrategy.RETRY,
            actions=[RecoveryAction.WAIT_AND_RETRY],
            max_attempts=1,
            delay_seconds=0.1,
            success_callback=success_callback,
            failure_callback=failure_callback
        )
        
        class TestError(Exception):
            pass
        
        self.recovery_manager.register_recovery_plan(TestError, custom_plan)
        
        # 测试成功回调
        operation = Mock(return_value="成功")
        error = TestError("测试")
        
        success, result, final_error = self.recovery_manager.attempt_recovery(
            error, operation
        )
        
        assert success is True
        success_callback.assert_called_once()
        failure_callback.assert_not_called()
        
        # 重置mock
        success_callback.reset_mock()
        failure_callback.reset_mock()
        
        # 测试失败回调
        operation = Mock(side_effect=RuntimeError("失败"))
        error = TestError("测试")
        
        success, result, final_error = self.recovery_manager.attempt_recovery(
            error, operation
        )
        
        assert success is False
        success_callback.assert_not_called()
        failure_callback.assert_called_once()


class TestGlobalRecoveryManager:
    """全局恢复管理器测试类"""
    
    def test_get_recovery_manager(self):
        """测试获取全局恢复管理器"""
        manager1 = get_recovery_manager()
        manager2 = get_recovery_manager()
        
        # 应该返回同一个实例
        assert manager1 is manager2
    
    def test_attempt_recovery_function(self):
        """测试全局恢复函数"""
        operation = Mock(return_value="成功")
        error = ValidationError("测试错误")
        
        success, result, final_error = attempt_recovery(error, operation)
        
        # ValidationError应该使用通用恢复计划
        assert success is True
        assert result == "成功"
        assert final_error is None


if __name__ == "__main__":
    pytest.main([__file__])