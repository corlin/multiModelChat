"""
UI组件测试

测试Streamlit界面组件的功能。
"""

import pytest
from unittest.mock import Mock, patch

from src.multi_llm_comparator.ui.utils import (
    format_memory_usage,
    show_error,
    show_success,
    show_info,
    show_warning
)


class TestUIUtils:
    """UI工具函数测试类"""
    
    def test_format_memory_usage_bytes(self):
        """测试字节格式化"""
        result = format_memory_usage(512)
        assert result == "512 B"
    
    def test_format_memory_usage_kb(self):
        """测试KB格式化"""
        result = format_memory_usage(1536)  # 1.5 KB
        assert result == "1.5 KB"
    
    def test_format_memory_usage_mb(self):
        """测试MB格式化"""
        result = format_memory_usage(1572864)  # 1.5 MB
        assert result == "1.5 MB"
    
    def test_format_memory_usage_gb(self):
        """测试GB格式化"""
        result = format_memory_usage(1610612736)  # 1.5 GB
        assert result == "1.5 GB"
    
    @patch('streamlit.error')
    def test_show_error(self, mock_error):
        """测试显示错误消息"""
        show_error("Test error")
        mock_error.assert_called_once_with("❌ Test error")
    
    @patch('streamlit.success')
    def test_show_success(self, mock_success):
        """测试显示成功消息"""
        show_success("Test success")
        mock_success.assert_called_once_with("✅ Test success")
    
    @patch('streamlit.info')
    def test_show_info(self, mock_info):
        """测试显示信息消息"""
        show_info("Test info")
        mock_info.assert_called_once_with("ℹ️ Test info")
    
    @patch('streamlit.warning')
    def test_show_warning(self, mock_warning):
        """测试显示警告消息"""
        show_warning("Test warning")
        mock_warning.assert_called_once_with("⚠️ Test warning")