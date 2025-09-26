"""
错误处理服务

提供全面的错误处理、日志记录和用户友好的错误提示功能。
"""

import logging
import traceback
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Callable
from enum import Enum
import json

from ..core.exceptions import (
    MultiLLMComparatorError,
    ModelLoadError,
    ModelNotFoundError,
    InferenceError,
    ConfigurationError,
    MemoryError,
    ValidationError,
    ModelSelectionError
)


class ErrorSeverity(Enum):
    """错误严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """错误类别"""
    MODEL_LOADING = "model_loading"
    INFERENCE = "inference"
    CONFIGURATION = "configuration"
    MEMORY = "memory"
    FILE_ACCESS = "file_access"
    VALIDATION = "validation"
    NETWORK = "network"
    SYSTEM = "system"
    USER_INPUT = "user_input"


class ErrorInfo:
    """错误信息封装"""
    
    def __init__(
        self,
        error: Exception,
        category: ErrorCategory,
        severity: ErrorSeverity,
        user_message: str,
        technical_details: str,
        suggestions: List[str],
        context: Optional[Dict[str, Any]] = None,
        recoverable: bool = True
    ):
        self.error = error
        self.category = category
        self.severity = severity
        self.user_message = user_message
        self.technical_details = technical_details
        self.suggestions = suggestions
        self.context = context or {}
        self.recoverable = recoverable
        self.timestamp = datetime.now()
        self.error_id = f"{category.value}_{int(self.timestamp.timestamp())}"


class ErrorHandler:
    """错误处理器"""
    
    def __init__(self, log_file: Optional[str] = None):
        """
        初始化错误处理器
        
        Args:
            log_file: 日志文件路径，如果为None则使用默认路径
        """
        self.log_file = log_file or "logs/error.log"
        self.setup_logging()
        self.error_history: List[ErrorInfo] = []
        self.max_history_size = 100
        
        # 错误处理映射
        self.error_handlers = {
            ModelLoadError: self._handle_model_load_error,
            ModelNotFoundError: self._handle_model_not_found_error,
            InferenceError: self._handle_inference_error,
            ConfigurationError: self._handle_configuration_error,
            MemoryError: self._handle_memory_error,
            ValidationError: self._handle_validation_error,
            ModelSelectionError: self._handle_model_selection_error,
            FileNotFoundError: self._handle_file_not_found_error,
            PermissionError: self._handle_permission_error,
            OSError: self._handle_os_error,
            ImportError: self._handle_import_error,
            Exception: self._handle_generic_error
        }
    
    def setup_logging(self):
        """设置日志记录"""
        # 创建日志目录
        log_path = Path(self.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 配置日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 文件处理器
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # 配置根日志器
        self.logger = logging.getLogger('multi_llm_comparator.error_handler')
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        user_callback: Optional[Callable[[ErrorInfo], None]] = None
    ) -> ErrorInfo:
        """
        处理错误
        
        Args:
            error: 异常对象
            context: 错误上下文信息
            user_callback: 用户回调函数，用于显示错误信息
            
        Returns:
            ErrorInfo: 错误信息对象
        """
        # 查找合适的错误处理器
        handler = None
        for error_type, error_handler in self.error_handlers.items():
            if isinstance(error, error_type):
                handler = error_handler
                break
        
        if not handler:
            handler = self._handle_generic_error
        
        # 处理错误
        error_info = handler(error, context)
        
        # 记录错误
        self._log_error(error_info)
        
        # 添加到历史记录
        self._add_to_history(error_info)
        
        # 调用用户回调
        if user_callback:
            try:
                user_callback(error_info)
            except Exception as callback_error:
                self.logger.error(f"用户回调函数执行失败: {callback_error}")
        
        return error_info
    
    def _handle_model_load_error(self, error: ModelLoadError, context: Optional[Dict] = None) -> ErrorInfo:
        """处理模型加载错误"""
        suggestions = [
            "检查模型文件是否存在且完整",
            "确认模型格式是否受支持（PyTorch: .bin, .pt, .pth, .safetensors; GGUF: .gguf）",
            "检查可用内存是否足够加载模型",
            "尝试使用较小的模型或调整内存设置",
            "检查模型文件权限是否正确"
        ]
        
        model_path = context.get('model_path', '未知') if context else '未知'
        user_message = f"模型加载失败: {model_path}"
        
        return ErrorInfo(
            error=error,
            category=ErrorCategory.MODEL_LOADING,
            severity=ErrorSeverity.HIGH,
            user_message=user_message,
            technical_details=str(error),
            suggestions=suggestions,
            context=context,
            recoverable=True
        )
    
    def _handle_model_not_found_error(self, error: ModelNotFoundError, context: Optional[Dict] = None) -> ErrorInfo:
        """处理模型未找到错误"""
        suggestions = [
            "检查模型目录路径是否正确",
            "确认模型文件是否存在",
            "重新扫描模型目录",
            "检查文件名是否正确",
            "确认模型文件未被移动或删除"
        ]
        
        return ErrorInfo(
            error=error,
            category=ErrorCategory.MODEL_LOADING,
            severity=ErrorSeverity.MEDIUM,
            user_message="找不到指定的模型文件",
            technical_details=str(error),
            suggestions=suggestions,
            context=context,
            recoverable=True
        )
    
    def _handle_inference_error(self, error: InferenceError, context: Optional[Dict] = None) -> ErrorInfo:
        """处理推理错误"""
        suggestions = [
            "检查输入提示词是否有效",
            "尝试调整模型参数（temperature, max_tokens等）",
            "检查模型是否正确加载",
            "确认系统资源是否充足",
            "尝试重新启动推理过程"
        ]
        
        model_id = context.get('model_id', '未知') if context else '未知'
        user_message = f"模型推理失败: {model_id}"
        
        return ErrorInfo(
            error=error,
            category=ErrorCategory.INFERENCE,
            severity=ErrorSeverity.HIGH,
            user_message=user_message,
            technical_details=str(error),
            suggestions=suggestions,
            context=context,
            recoverable=True
        )
    
    def _handle_configuration_error(self, error: ConfigurationError, context: Optional[Dict] = None) -> ErrorInfo:
        """处理配置错误"""
        suggestions = [
            "检查配置文件格式是否正确",
            "确认所有必需的配置项都已设置",
            "重置为默认配置",
            "检查配置文件权限",
            "参考文档中的配置示例"
        ]
        
        return ErrorInfo(
            error=error,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.MEDIUM,
            user_message="配置错误",
            technical_details=str(error),
            suggestions=suggestions,
            context=context,
            recoverable=True
        )
    
    def _handle_memory_error(self, error: MemoryError, context: Optional[Dict] = None) -> ErrorInfo:
        """处理内存错误"""
        suggestions = [
            "关闭其他占用内存的应用程序",
            "减少同时比较的模型数量",
            "使用较小的模型",
            "调整模型参数以减少内存使用",
            "重启应用程序以释放内存",
            "考虑升级系统内存"
        ]
        
        return ErrorInfo(
            error=error,
            category=ErrorCategory.MEMORY,
            severity=ErrorSeverity.HIGH,
            user_message="内存不足",
            technical_details=str(error),
            suggestions=suggestions,
            context=context,
            recoverable=True
        )
    
    def _handle_validation_error(self, error: ValidationError, context: Optional[Dict] = None) -> ErrorInfo:
        """处理验证错误"""
        suggestions = [
            "检查输入参数的格式和范围",
            "参考参数说明调整设置",
            "使用推荐的默认值",
            "确认输入数据的有效性"
        ]
        
        return ErrorInfo(
            error=error,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            user_message="参数验证失败",
            technical_details=str(error),
            suggestions=suggestions,
            context=context,
            recoverable=True
        )
    
    def _handle_model_selection_error(self, error: ModelSelectionError, context: Optional[Dict] = None) -> ErrorInfo:
        """处理模型选择错误"""
        suggestions = [
            "检查选择的模型数量是否超过限制",
            "确认所选模型都是有效的",
            "重新扫描模型目录",
            "检查模型文件是否完整"
        ]
        
        return ErrorInfo(
            error=error,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            user_message="模型选择错误",
            technical_details=str(error),
            suggestions=suggestions,
            context=context,
            recoverable=True
        )
    
    def _handle_file_not_found_error(self, error: FileNotFoundError, context: Optional[Dict] = None) -> ErrorInfo:
        """处理文件未找到错误"""
        suggestions = [
            "检查文件路径是否正确",
            "确认文件是否存在",
            "检查文件权限",
            "尝试使用绝对路径",
            "确认文件未被移动或删除"
        ]
        
        filename = getattr(error, 'filename', '未知文件')
        user_message = f"找不到文件: {filename}"
        
        return ErrorInfo(
            error=error,
            category=ErrorCategory.FILE_ACCESS,
            severity=ErrorSeverity.MEDIUM,
            user_message=user_message,
            technical_details=str(error),
            suggestions=suggestions,
            context=context,
            recoverable=True
        )
    
    def _handle_permission_error(self, error: PermissionError, context: Optional[Dict] = None) -> ErrorInfo:
        """处理权限错误"""
        suggestions = [
            "检查文件或目录的访问权限",
            "以管理员身份运行程序",
            "更改文件或目录的权限设置",
            "确认当前用户有足够的权限",
            "检查文件是否被其他程序占用"
        ]
        
        filename = getattr(error, 'filename', '未知文件')
        user_message = f"权限不足: {filename}"
        
        return ErrorInfo(
            error=error,
            category=ErrorCategory.FILE_ACCESS,
            severity=ErrorSeverity.HIGH,
            user_message=user_message,
            technical_details=str(error),
            suggestions=suggestions,
            context=context,
            recoverable=True
        )
    
    def _handle_os_error(self, error: OSError, context: Optional[Dict] = None) -> ErrorInfo:
        """处理操作系统错误"""
        suggestions = [
            "检查系统资源是否充足",
            "确认文件系统状态正常",
            "重启应用程序",
            "检查磁盘空间",
            "确认网络连接正常（如果涉及网络操作）"
        ]
        
        return ErrorInfo(
            error=error,
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.HIGH,
            user_message="系统操作失败",
            technical_details=str(error),
            suggestions=suggestions,
            context=context,
            recoverable=True
        )
    
    def _handle_import_error(self, error: ImportError, context: Optional[Dict] = None) -> ErrorInfo:
        """处理导入错误"""
        suggestions = [
            "检查所需的Python包是否已安装",
            "使用 'uv sync' 安装项目依赖",
            "检查Python环境是否正确",
            "确认包版本兼容性",
            "重新安装相关依赖包"
        ]
        
        module_name = getattr(error, 'name', '未知模块')
        user_message = f"缺少依赖模块: {module_name}"
        
        return ErrorInfo(
            error=error,
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.CRITICAL,
            user_message=user_message,
            technical_details=str(error),
            suggestions=suggestions,
            context=context,
            recoverable=False
        )
    
    def _handle_generic_error(self, error: Exception, context: Optional[Dict] = None) -> ErrorInfo:
        """处理通用错误"""
        suggestions = [
            "重试操作",
            "重启应用程序",
            "检查系统日志获取更多信息",
            "联系技术支持",
            "查看详细错误信息进行诊断"
        ]
        
        return ErrorInfo(
            error=error,
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.MEDIUM,
            user_message="发生未知错误",
            technical_details=str(error),
            suggestions=suggestions,
            context=context,
            recoverable=True
        )
    
    def _log_error(self, error_info: ErrorInfo):
        """记录错误到日志"""
        # 清理 context 中不能序列化的对象
        safe_context = {}
        if error_info.context:
            for key, value in error_info.context.items():
                try:
                    json.dumps(value)  # 测试是否可以序列化
                    safe_context[key] = value
                except (TypeError, ValueError):
                    # 如果不能序列化，转换为字符串表示
                    safe_context[key] = str(value)
        
        log_data = {
            'error_id': error_info.error_id,
            'timestamp': error_info.timestamp.isoformat(),
            'category': error_info.category.value,
            'severity': error_info.severity.value,
            'user_message': error_info.user_message,
            'technical_details': error_info.technical_details,
            'context': safe_context,
            'recoverable': error_info.recoverable,
            'traceback': traceback.format_exc()
        }
        
        # 根据严重程度选择日志级别
        if error_info.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(f"CRITICAL ERROR: {json.dumps(log_data, ensure_ascii=False, indent=2)}")
        elif error_info.severity == ErrorSeverity.HIGH:
            self.logger.error(f"HIGH SEVERITY ERROR: {json.dumps(log_data, ensure_ascii=False, indent=2)}")
        elif error_info.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(f"MEDIUM SEVERITY ERROR: {json.dumps(log_data, ensure_ascii=False, indent=2)}")
        else:
            self.logger.info(f"LOW SEVERITY ERROR: {json.dumps(log_data, ensure_ascii=False, indent=2)}")
    
    def _add_to_history(self, error_info: ErrorInfo):
        """添加错误到历史记录"""
        self.error_history.append(error_info)
        
        # 限制历史记录大小
        if len(self.error_history) > self.max_history_size:
            self.error_history = self.error_history[-self.max_history_size:]
    
    def get_error_history(self, limit: Optional[int] = None) -> List[ErrorInfo]:
        """
        获取错误历史记录
        
        Args:
            limit: 返回的最大记录数
            
        Returns:
            List[ErrorInfo]: 错误历史记录列表
        """
        if limit:
            return self.error_history[-limit:]
        return self.error_history.copy()
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """
        获取错误统计信息
        
        Returns:
            Dict[str, Any]: 错误统计信息
        """
        if not self.error_history:
            return {
                'total_errors': 0,
                'by_category': {},
                'by_severity': {},
                'recoverable_count': 0,
                'non_recoverable_count': 0
            }
        
        # 按类别统计
        by_category = {}
        for error_info in self.error_history:
            category = error_info.category.value
            by_category[category] = by_category.get(category, 0) + 1
        
        # 按严重程度统计
        by_severity = {}
        for error_info in self.error_history:
            severity = error_info.severity.value
            by_severity[severity] = by_severity.get(severity, 0) + 1
        
        # 可恢复性统计
        recoverable_count = sum(1 for e in self.error_history if e.recoverable)
        non_recoverable_count = len(self.error_history) - recoverable_count
        
        return {
            'total_errors': len(self.error_history),
            'by_category': by_category,
            'by_severity': by_severity,
            'recoverable_count': recoverable_count,
            'non_recoverable_count': non_recoverable_count
        }
    
    def clear_history(self):
        """清空错误历史记录"""
        self.error_history.clear()
        self.logger.info("错误历史记录已清空")
    
    def export_error_log(self, output_file: str) -> bool:
        """
        导出错误日志
        
        Args:
            output_file: 输出文件路径
            
        Returns:
            bool: 是否成功导出
        """
        try:
            log_data = []
            for error_info in self.error_history:
                log_data.append({
                    'error_id': error_info.error_id,
                    'timestamp': error_info.timestamp.isoformat(),
                    'category': error_info.category.value,
                    'severity': error_info.severity.value,
                    'user_message': error_info.user_message,
                    'technical_details': error_info.technical_details,
                    'suggestions': error_info.suggestions,
                    'context': error_info.context,
                    'recoverable': error_info.recoverable
                })
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"错误日志已导出到: {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"导出错误日志失败: {e}")
            return False


# 全局错误处理器实例
_global_error_handler: Optional[ErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """获取全局错误处理器实例"""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler()
    return _global_error_handler


def handle_error(
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    user_callback: Optional[Callable[[ErrorInfo], None]] = None
) -> ErrorInfo:
    """
    处理错误的便捷函数
    
    Args:
        error: 异常对象
        context: 错误上下文信息
        user_callback: 用户回调函数
        
    Returns:
        ErrorInfo: 错误信息对象
    """
    return get_error_handler().handle_error(error, context, user_callback)


def setup_error_handling(log_file: Optional[str] = None):
    """
    设置全局错误处理
    
    Args:
        log_file: 日志文件路径
    """
    global _global_error_handler
    _global_error_handler = ErrorHandler(log_file)