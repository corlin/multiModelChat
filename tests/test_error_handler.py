"""
错误处理器测试

测试错误处理、日志记录和用户友好错误提示功能。
"""

import pytest
import tempfile
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.multi_llm_comparator.services.error_handler import (
    ErrorHandler,
    ErrorInfo,
    ErrorSeverity,
    ErrorCategory,
    get_error_handler,
    handle_error,
    setup_error_handling
)
from src.multi_llm_comparator.core.exceptions import (
    ModelLoadError,
    ModelNotFoundError,
    InferenceError,
    ConfigurationError,
    MemoryError,
    ValidationError,
    ModelSelectionError
)


class TestErrorHandler:
    """错误处理器测试类"""
    
    def setup_method(self):
        """测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.temp_dir, "test_error.log")
        self.error_handler = ErrorHandler(self.log_file)
    
    def teardown_method(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_error_handler_initialization(self):
        """测试错误处理器初始化"""
        assert self.error_handler.log_file == self.log_file
        assert len(self.error_handler.error_history) == 0
        assert self.error_handler.max_history_size == 100
        assert Path(self.log_file).parent.exists()
    
    def test_model_load_error_handling(self):
        """测试模型加载错误处理"""
        error = ModelLoadError("模型文件损坏")
        context = {"model_path": "/path/to/model.bin"}
        
        error_info = self.error_handler.handle_error(error, context)
        
        assert error_info.category == ErrorCategory.MODEL_LOADING
        assert error_info.severity == ErrorSeverity.HIGH
        assert "模型加载失败" in error_info.user_message
        assert "/path/to/model.bin" in error_info.user_message
        assert "检查模型文件是否存在且完整" in error_info.suggestions
        assert error_info.recoverable is True
        assert error_info.context == context
    
    def test_model_not_found_error_handling(self):
        """测试模型未找到错误处理"""
        error = ModelNotFoundError("找不到模型文件")
        
        error_info = self.error_handler.handle_error(error)
        
        assert error_info.category == ErrorCategory.MODEL_LOADING
        assert error_info.severity == ErrorSeverity.MEDIUM
        assert "找不到指定的模型文件" in error_info.user_message
        assert "检查模型目录路径是否正确" in error_info.suggestions
        assert error_info.recoverable is True
    
    def test_inference_error_handling(self):
        """测试推理错误处理"""
        error = InferenceError("推理过程中断")
        context = {"model_id": "test_model"}
        
        error_info = self.error_handler.handle_error(error, context)
        
        assert error_info.category == ErrorCategory.INFERENCE
        assert error_info.severity == ErrorSeverity.HIGH
        assert "模型推理失败: test_model" in error_info.user_message
        assert "检查输入提示词是否有效" in error_info.suggestions
        assert error_info.recoverable is True
    
    def test_configuration_error_handling(self):
        """测试配置错误处理"""
        error = ConfigurationError("配置文件格式错误")
        
        error_info = self.error_handler.handle_error(error)
        
        assert error_info.category == ErrorCategory.CONFIGURATION
        assert error_info.severity == ErrorSeverity.MEDIUM
        assert "配置错误" in error_info.user_message
        assert "检查配置文件格式是否正确" in error_info.suggestions
        assert error_info.recoverable is True
    
    def test_memory_error_handling(self):
        """测试内存错误处理"""
        error = MemoryError("内存不足")
        
        error_info = self.error_handler.handle_error(error)
        
        assert error_info.category == ErrorCategory.MEMORY
        assert error_info.severity == ErrorSeverity.HIGH
        assert "内存不足" in error_info.user_message
        assert "关闭其他占用内存的应用程序" in error_info.suggestions
        assert error_info.recoverable is True
    
    def test_validation_error_handling(self):
        """测试验证错误处理"""
        error = ValidationError("参数超出范围")
        
        error_info = self.error_handler.handle_error(error)
        
        assert error_info.category == ErrorCategory.VALIDATION
        assert error_info.severity == ErrorSeverity.LOW
        assert "参数验证失败" in error_info.user_message
        assert "检查输入参数的格式和范围" in error_info.suggestions
        assert error_info.recoverable is True
    
    def test_model_selection_error_handling(self):
        """测试模型选择错误处理"""
        error = ModelSelectionError("选择的模型数量超过限制")
        
        error_info = self.error_handler.handle_error(error)
        
        assert error_info.category == ErrorCategory.VALIDATION
        assert error_info.severity == ErrorSeverity.MEDIUM
        assert "模型选择错误" in error_info.user_message
        assert "检查选择的模型数量是否超过限制" in error_info.suggestions
        assert error_info.recoverable is True
    
    def test_file_not_found_error_handling(self):
        """测试文件未找到错误处理"""
        error = FileNotFoundError("No such file or directory: 'test.txt'")
        error.filename = "test.txt"
        
        error_info = self.error_handler.handle_error(error)
        
        assert error_info.category == ErrorCategory.FILE_ACCESS
        assert error_info.severity == ErrorSeverity.MEDIUM
        assert "找不到文件: test.txt" in error_info.user_message
        assert "检查文件路径是否正确" in error_info.suggestions
        assert error_info.recoverable is True
    
    def test_permission_error_handling(self):
        """测试权限错误处理"""
        error = PermissionError("Permission denied: 'protected.txt'")
        error.filename = "protected.txt"
        
        error_info = self.error_handler.handle_error(error)
        
        assert error_info.category == ErrorCategory.FILE_ACCESS
        assert error_info.severity == ErrorSeverity.HIGH
        assert "权限不足: protected.txt" in error_info.user_message
        assert "检查文件或目录的访问权限" in error_info.suggestions
        assert error_info.recoverable is True
    
    def test_import_error_handling(self):
        """测试导入错误处理"""
        error = ImportError("No module named 'missing_module'")
        error.name = "missing_module"
        
        error_info = self.error_handler.handle_error(error)
        
        assert error_info.category == ErrorCategory.SYSTEM
        assert error_info.severity == ErrorSeverity.CRITICAL
        assert "缺少依赖模块: missing_module" in error_info.user_message
        assert "使用 'uv sync' 安装项目依赖" in error_info.suggestions
        assert error_info.recoverable is False
    
    def test_generic_error_handling(self):
        """测试通用错误处理"""
        error = RuntimeError("未知运行时错误")
        
        error_info = self.error_handler.handle_error(error)
        
        assert error_info.category == ErrorCategory.SYSTEM
        assert error_info.severity == ErrorSeverity.MEDIUM
        assert "发生未知错误" in error_info.user_message
        assert "重试操作" in error_info.suggestions
        assert error_info.recoverable is True
    
    def test_user_callback(self):
        """测试用户回调功能"""
        callback_called = False
        callback_error_info = None
        
        def user_callback(error_info):
            nonlocal callback_called, callback_error_info
            callback_called = True
            callback_error_info = error_info
        
        error = ValidationError("测试错误")
        error_info = self.error_handler.handle_error(error, user_callback=user_callback)
        
        assert callback_called is True
        assert callback_error_info == error_info
    
    def test_error_history(self):
        """测试错误历史记录"""
        # 添加多个错误
        errors = [
            ModelLoadError("错误1"),
            InferenceError("错误2"),
            ValidationError("错误3")
        ]
        
        for error in errors:
            self.error_handler.handle_error(error)
        
        history = self.error_handler.get_error_history()
        assert len(history) == 3
        
        # 测试限制数量
        limited_history = self.error_handler.get_error_history(limit=2)
        assert len(limited_history) == 2
        
        # 测试历史记录内容
        assert history[0].error == errors[0]
        assert history[1].error == errors[1]
        assert history[2].error == errors[2]
    
    def test_error_statistics(self):
        """测试错误统计信息"""
        # 添加不同类型的错误
        errors = [
            ModelLoadError("错误1"),
            ModelLoadError("错误2"),
            InferenceError("错误3"),
            ValidationError("错误4")
        ]
        
        for error in errors:
            self.error_handler.handle_error(error)
        
        stats = self.error_handler.get_error_statistics()
        
        assert stats['total_errors'] == 4
        assert stats['by_category']['model_loading'] == 2
        assert stats['by_category']['inference'] == 1
        assert stats['by_category']['validation'] == 1
        assert stats['by_severity']['high'] == 3  # 2 ModelLoadError + 1 InferenceError
        assert stats['by_severity']['low'] == 1   # 1 ValidationError
        assert stats['recoverable_count'] == 4
        assert stats['non_recoverable_count'] == 0
    
    def test_clear_history(self):
        """测试清空历史记录"""
        # 添加错误
        self.error_handler.handle_error(ValidationError("测试错误"))
        assert len(self.error_handler.get_error_history()) == 1
        
        # 清空历史
        self.error_handler.clear_history()
        assert len(self.error_handler.get_error_history()) == 0
    
    def test_export_error_log(self):
        """测试导出错误日志"""
        # 添加错误
        error = ModelLoadError("测试错误")
        context = {"model_path": "/test/path"}
        self.error_handler.handle_error(error, context)
        
        # 导出日志
        export_file = os.path.join(self.temp_dir, "exported_log.json")
        success = self.error_handler.export_error_log(export_file)
        
        assert success is True
        assert os.path.exists(export_file)
        
        # 验证导出内容
        with open(export_file, 'r', encoding='utf-8') as f:
            log_data = json.load(f)
        
        assert len(log_data) == 1
        assert log_data[0]['category'] == 'model_loading'
        assert log_data[0]['user_message'] == '模型加载失败: /test/path'
        assert log_data[0]['context'] == context
    
    def test_logging_functionality(self):
        """测试日志记录功能"""
        error = ModelLoadError("测试日志记录")
        self.error_handler.handle_error(error)
        
        # 检查日志文件是否创建
        assert os.path.exists(self.log_file)
        
        # 检查日志内容
        with open(self.log_file, 'r', encoding='utf-8') as f:
            log_content = f.read()
        
        assert "HIGH SEVERITY ERROR" in log_content
        assert "model_loading" in log_content
        assert "测试日志记录" in log_content


class TestGlobalErrorHandler:
    """全局错误处理器测试类"""
    
    def test_get_error_handler(self):
        """测试获取全局错误处理器"""
        handler1 = get_error_handler()
        handler2 = get_error_handler()
        
        # 应该返回同一个实例
        assert handler1 is handler2
    
    def test_handle_error_function(self):
        """测试全局错误处理函数"""
        error = ValidationError("测试全局处理")
        error_info = handle_error(error)
        
        assert error_info.category == ErrorCategory.VALIDATION
        assert error_info.user_message == "参数验证失败"
    
    def test_setup_error_handling(self):
        """测试设置全局错误处理"""
        temp_dir = tempfile.mkdtemp()
        log_file = os.path.join(temp_dir, "custom_error.log")
        
        try:
            setup_error_handling(log_file)
            handler = get_error_handler()
            
            assert handler.log_file == log_file
            
            # 测试日志记录
            handle_error(ValidationError("测试自定义日志"))
            assert os.path.exists(log_file)
            
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestErrorInfo:
    """错误信息测试类"""
    
    def test_error_info_creation(self):
        """测试错误信息创建"""
        error = ValidationError("测试错误")
        context = {"test": "context"}
        suggestions = ["建议1", "建议2"]
        
        error_info = ErrorInfo(
            error=error,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            user_message="用户消息",
            technical_details="技术详情",
            suggestions=suggestions,
            context=context,
            recoverable=True
        )
        
        assert error_info.error == error
        assert error_info.category == ErrorCategory.VALIDATION
        assert error_info.severity == ErrorSeverity.LOW
        assert error_info.user_message == "用户消息"
        assert error_info.technical_details == "技术详情"
        assert error_info.suggestions == suggestions
        assert error_info.context == context
        assert error_info.recoverable is True
        assert error_info.timestamp is not None
        assert error_info.error_id.startswith("validation_")


if __name__ == "__main__":
    pytest.main([__file__])