"""
UI组件单元测试
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.multi_llm_comparator.core.models import ModelInfo, ModelType
from src.multi_llm_comparator.ui.utils import (
    format_file_size,
    format_duration,
    format_tokens_per_second,
    safe_get_nested_value,
    format_model_display_name,
    truncate_text
)


class TestUIUtils:
    """UI工具函数测试"""
    
    def test_format_file_size(self):
        """测试文件大小格式化"""
        assert format_file_size(512) == "512 B"
        assert format_file_size(1024) == "1.0 KB"
        assert format_file_size(1024 * 1024) == "1.0 MB"
        assert format_file_size(1024 * 1024 * 1024) == "1.0 GB"
        assert format_file_size(1536) == "1.5 KB"
        assert format_file_size(2.5 * 1024 * 1024 * 1024) == "2.5 GB"
    
    def test_format_duration(self):
        """测试时间长度格式化"""
        assert format_duration(0.5) == "500ms"
        assert format_duration(1.5) == "1.5s"
        assert format_duration(65) == "1m 5s"
        assert format_duration(3661) == "1h 1m"
        assert format_duration(0.001) == "1ms"
    
    def test_format_tokens_per_second(self):
        """测试token速度格式化"""
        assert format_tokens_per_second(100, 10) == "10.0 tokens/s"
        assert format_tokens_per_second(50, 2.5) == "20.0 tokens/s"
        assert format_tokens_per_second(100, 0) == "N/A"
        assert format_tokens_per_second(0, 10) == "0.0 tokens/s"
    
    def test_safe_get_nested_value(self):
        """测试安全获取嵌套值"""
        data = {
            'level1': {
                'level2': {
                    'value': 'test'
                }
            }
        }
        
        assert safe_get_nested_value(data, ['level1', 'level2', 'value']) == 'test'
        assert safe_get_nested_value(data, ['level1', 'missing'], 'default') == 'default'
        assert safe_get_nested_value(data, ['missing'], 'default') == 'default'
        assert safe_get_nested_value({}, ['key'], None) is None
    
    def test_format_model_display_name(self):
        """测试模型显示名称格式化"""
        result = format_model_display_name("test-model", "pytorch", 1024 * 1024)
        assert result == "test-model (PYTORCH) - 1.0 MB"
        
        result = format_model_display_name("gguf-model", "gguf", 2.5 * 1024 * 1024 * 1024)
        assert result == "gguf-model (GGUF) - 2.5 GB"
    
    def test_truncate_text(self):
        """测试文本截断"""
        long_text = "这是一个很长的文本，需要被截断"
        
        assert truncate_text(long_text, 10) == "这是一个很长的..."
        assert truncate_text("短文本", 10) == "短文本"
        assert truncate_text(long_text, 10, ">>") == "这是一个很长的文>>"
        assert truncate_text("", 10) == ""


class TestUIComponentsIntegration:
    """UI组件集成测试"""
    
    @patch('streamlit.markdown')
    def test_render_model_card_integration(self, mock_markdown):
        """测试模型卡片渲染集成"""
        from src.multi_llm_comparator.ui.components import render_model_card
        
        model = ModelInfo(
            id="test-model",
            name="Test Model",
            path="/path/to/model",
            model_type=ModelType.PYTORCH,
            size=1024 * 1024,  # 1MB
            config={}
        )
        
        render_model_card(model, is_selected=True)
        
        # 验证markdown被调用
        mock_markdown.assert_called_once()
        call_args = mock_markdown.call_args[0][0]
        
        assert "✅ Test Model" in call_args
        assert "🔥 PYTORCH" in call_args
        assert "1.0 MB" in call_args
        assert "/path/to/model" in call_args
    
    @patch('streamlit.slider')
    @patch('streamlit.number_input')
    @patch('streamlit.checkbox')
    @patch('streamlit.selectbox')
    def test_render_parameter_input_types(self, mock_selectbox, mock_checkbox, 
                                        mock_number_input, mock_slider):
        """测试参数输入控件类型"""
        from src.multi_llm_comparator.ui.components import render_parameter_input
        
        # 测试slider
        mock_slider.return_value = 0.7
        result = render_parameter_input("temperature", "slider", 0.5, 0.0, 1.0, 0.1)
        assert result == 0.7
        mock_slider.assert_called_once()
        
        # 测试number_input
        mock_number_input.return_value = 512
        result = render_parameter_input("max_tokens", "number", 256, 1, 4096)
        assert result == 512
        mock_number_input.assert_called_once()
        
        # 测试checkbox
        mock_checkbox.return_value = True
        result = render_parameter_input("do_sample", "checkbox", False)
        assert result is True
        mock_checkbox.assert_called_once()
        
        # 测试selectbox
        mock_selectbox.return_value = "float16"
        result = render_parameter_input("dtype", "selectbox", "auto", 
                                      options=["auto", "float16", "float32"])
        assert result == "float16"
        mock_selectbox.assert_called_once()
    
    @patch('streamlit.progress')
    @patch('streamlit.text')
    def test_render_progress_indicator(self, mock_text, mock_progress):
        """测试进度指示器渲染"""
        from src.multi_llm_comparator.ui.components import render_progress_indicator
        
        render_progress_indicator(3, 5, "处理中", True)
        
        mock_progress.assert_called_once_with(0.6)  # 3/5 = 0.6
        mock_text.assert_called_once_with("处理中 - 60% (3/5)")
    
    @patch('streamlit.success')
    @patch('streamlit.error')
    @patch('streamlit.warning')
    @patch('streamlit.info')
    def test_render_status_badge(self, mock_info, mock_warning, mock_error, mock_success):
        """测试状态徽章渲染"""
        from src.multi_llm_comparator.ui.components import render_status_badge
        
        render_status_badge('success', '操作成功')
        mock_success.assert_called_once_with("✅ 操作成功")
        
        render_status_badge('error', '操作失败')
        mock_error.assert_called_once_with("❌ 操作失败")
        
        render_status_badge('warning', '注意事项')
        mock_warning.assert_called_once_with("⚠️ 注意事项")
        
        render_status_badge('info', '信息提示')
        mock_info.assert_called_once_with("ℹ️ 信息提示")
    
    @patch('streamlit.columns')
    @patch('streamlit.metric')
    def test_render_model_statistics(self, mock_metric, mock_columns):
        """测试模型统计信息渲染"""
        from src.multi_llm_comparator.ui.components import render_model_statistics
        
        # 模拟columns返回，创建支持上下文管理器的Mock
        mock_col1 = MagicMock()
        mock_col2 = MagicMock()
        mock_col3 = MagicMock()
        mock_col4 = MagicMock()
        mock_columns.return_value = [mock_col1, mock_col2, mock_col3, mock_col4]
        
        stats = {
            'total_models': 10,
            'pytorch_models': 6,
            'gguf_models': 4,
            'total_size_gb': 25.5
        }
        
        render_model_statistics(stats)
        
        # 验证columns被调用
        mock_columns.assert_called_once_with(4)
        
        # 验证每个列的上下文管理器被使用
        mock_col1.__enter__.assert_called_once()
        mock_col2.__enter__.assert_called_once()
        mock_col3.__enter__.assert_called_once()
        mock_col4.__enter__.assert_called_once()
    
    @patch('streamlit.columns')
    @patch('streamlit.metric')
    @patch('streamlit.progress')
    @patch('streamlit.caption')
    def test_render_comparison_summary(self, mock_caption, mock_progress, 
                                     mock_metric, mock_columns):
        """测试比较结果摘要渲染"""
        from src.multi_llm_comparator.ui.components import render_comparison_summary
        
        # 模拟columns返回，创建支持上下文管理器的Mock
        mock_columns.return_value = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        
        results = {
            'model1': {'status': 'completed'},
            'model2': {'status': 'completed'},
            'model3': {'status': 'running'},
            'model4': {'error': 'some error'}
        }
        
        render_comparison_summary(results)
        
        # 验证进度条被调用 (2/4 = 0.5)
        mock_progress.assert_called_once_with(0.5)
        mock_caption.assert_called_once_with("整体进度: 2/4 (50%)")
    
    @patch('streamlit.info')
    def test_render_comparison_summary_empty(self, mock_info):
        """测试空比较结果摘要"""
        from src.multi_llm_comparator.ui.components import render_comparison_summary
        
        render_comparison_summary({})
        mock_info.assert_called_once_with("暂无比较结果")


class TestUIComponentsErrorHandling:
    """UI组件错误处理测试"""
    
    @patch('streamlit.error')
    @patch('streamlit.expander')
    @patch('streamlit.code')
    def test_render_error_message(self, mock_code, mock_expander, mock_error):
        """测试错误消息渲染"""
        from src.multi_llm_comparator.ui.components import render_error_message
        from src.multi_llm_comparator.core.exceptions import ValidationError
        
        # 测试普通异常
        error = ValueError("测试错误")
        render_error_message(error, "操作失败")
        
        mock_error.assert_called_with("❌ **ValueError**: 操作失败: 测试错误")
        
        # 测试验证异常
        mock_expander_context = MagicMock()
        mock_expander.return_value = mock_expander_context
        
        validation_error = ValidationError("参数验证失败")
        render_error_message(validation_error)
        
        # 验证展开器被创建
        mock_expander.assert_called_with("查看详细错误信息")
    
    def test_safe_get_nested_value_edge_cases(self):
        """测试安全获取嵌套值的边界情况"""
        # 测试None数据
        assert safe_get_nested_value(None, ['key'], 'default') == 'default'
        
        # 测试空键列表
        data = {'key': 'value'}
        assert safe_get_nested_value(data, [], 'default') == data
        
        # 测试非字典类型
        assert safe_get_nested_value("string", ['key'], 'default') == 'default'
        assert safe_get_nested_value(123, ['key'], 'default') == 'default'
        assert safe_get_nested_value([], ['key'], 'default') == 'default'


if __name__ == "__main__":
    pytest.main([__file__])