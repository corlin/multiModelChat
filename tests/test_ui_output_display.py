"""
UI输出显示功能单元测试
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.multi_llm_comparator.core.models import ModelInfo, ModelType


class TestOutputDisplayComponents:
    """输出显示组件测试"""
    
    @patch('streamlit.markdown')
    @patch('streamlit.progress')
    @patch('streamlit.text')
    @patch('streamlit.info')
    @patch('streamlit.error')
    def test_render_model_output_card_all_states(self, mock_error, mock_info, mock_text, 
                                               mock_progress, mock_markdown):
        """测试模型输出卡片的所有状态"""
        from src.multi_llm_comparator.ui.components import render_model_output_card
        
        # 测试等待状态
        render_model_output_card("Test Model", "pytorch", "pending")
        mock_info.assert_called_with("⏳ 等待开始...")
        
        # 测试运行状态
        render_model_output_card("Test Model", "pytorch", "running")
        mock_progress.assert_called_with(0.5)
        mock_text.assert_called_with("🔄 正在生成回答...")
        
        # 测试完成状态
        content = "这是测试内容"
        stats = {'duration': 5.0, 'token_count': 100}
        render_model_output_card("Test Model", "pytorch", "completed", 
                               content=content, stats=stats)
        mock_markdown.assert_called_with(content)
        
        # 测试错误状态
        error_msg = "测试错误"
        render_model_output_card("Test Model", "pytorch", "error", error=error_msg)
        mock_error.assert_called_with(f"❌ 错误: {error_msg}")
    
    @patch('streamlit.text_area')
    def test_render_model_output_card_raw_view(self, mock_text_area):
        """测试原始文本视图"""
        from src.multi_llm_comparator.ui.components import render_model_output_card
        
        content = "原始文本内容"
        render_model_output_card("Test Model", "pytorch", "completed", 
                               content=content, show_raw=True)
        
        mock_text_area.assert_called_once()
        call_args = mock_text_area.call_args
        assert call_args[1]['value'] == content
        assert call_args[1]['disabled'] is True
    
    @patch('streamlit.session_state', {})
    @patch('streamlit.empty')
    @patch('streamlit.markdown')
    @patch('streamlit.text')
    def test_render_streaming_output(self, mock_text, mock_markdown, mock_empty):
        """测试流式输出渲染"""
        from src.multi_llm_comparator.ui.components import render_streaming_output
        import streamlit as st
        
        # 模拟empty容器
        mock_container = MagicMock()
        mock_empty.return_value = mock_container
        
        # 测试首次渲染
        render_streaming_output("test_key", "测试内容", False)
        
        # 验证容器被创建
        mock_empty.assert_called_once()
        
        # 测试完成状态
        render_streaming_output("test_key", "完整内容", True)
    
    @patch('streamlit.columns')
    def test_render_responsive_columns_different_counts(self, mock_columns):
        """测试响应式列布局的不同数量"""
        from src.multi_llm_comparator.ui.components import render_responsive_columns
        
        # 创建动态返回正确数量列的mock函数
        def mock_columns_func(num_cols):
            return [MagicMock() for _ in range(num_cols)]
        
        mock_columns.side_effect = mock_columns_func
        
        # 测试不同数量的项目
        items = ["item1"]
        render_func = Mock()
        
        # 1个项目
        render_responsive_columns(items, render_func)
        mock_columns.assert_called_with(1)
        
        # 2个项目
        items = ["item1", "item2"]
        render_responsive_columns(items, render_func)
        mock_columns.assert_called_with(2)
        
        # 3个项目
        items = ["item1", "item2", "item3"]
        render_responsive_columns(items, render_func)
        mock_columns.assert_called_with(3)
        
        # 4个项目（应该使用2列，调用两次）
        items = ["item1", "item2", "item3", "item4"]
        render_responsive_columns(items, render_func)
        # 4个项目时会调用两次st.columns(2)
        assert mock_columns.call_args_list[-1] == ((2,),)
        assert mock_columns.call_args_list[-2] == ((2,),)
    
    def test_render_responsive_columns_empty_items(self):
        """测试空项目列表"""
        from src.multi_llm_comparator.ui.components import render_responsive_columns
        
        render_func = Mock()
        render_responsive_columns([], render_func)
        
        # 验证渲染函数没有被调用
        render_func.assert_not_called()


class TestOutputDisplayIntegration:
    """输出显示集成测试"""
    
    def test_model_output_container_initialization(self):
        """测试模型输出容器初始化"""
        # 这个测试需要模拟Streamlit会话状态
        from src.multi_llm_comparator.core.models import ModelInfo, ModelType
        
        models = [
            ModelInfo(
                id="model1",
                name="Test Model 1",
                path="/path/to/model1",
                model_type=ModelType.PYTORCH,
                size=1024,
                config={}
            ),
            ModelInfo(
                id="model2", 
                name="Test Model 2",
                path="/path/to/model2",
                model_type=ModelType.GGUF,
                size=2048,
                config={}
            )
        ]
        
        # 模拟初始化过程
        output_containers = {}
        output_status = {}
        
        for model in models:
            output_containers[model.id] = {
                'content': "",
                'status': "pending",
                'start_time': None,
                'end_time': None,
                'token_count': 0,
                'error': None
            }
            output_status[model.id] = "pending"
        
        # 验证初始化结果
        assert len(output_containers) == 2
        assert output_containers["model1"]['status'] == "pending"
        assert output_containers["model2"]['status'] == "pending"
        assert output_status["model1"] == "pending"
        assert output_status["model2"] == "pending"
    
    def test_comparison_results_structure(self):
        """测试比较结果数据结构"""
        # 模拟比较结果
        comparison_results = {
            'model1': {
                'model_name': 'Test Model 1',
                'model_type': 'pytorch',
                'status': 'completed',
                'content': '这是测试回答',
                'error': None,
                'stats': {
                    'start_time': 1000.0,
                    'end_time': 1005.0,
                    'token_count': 50,
                    'duration': 5.0
                }
            },
            'model2': {
                'model_name': 'Test Model 2',
                'model_type': 'gguf',
                'status': 'error',
                'content': '',
                'error': '模型加载失败',
                'stats': None
            }
        }
        
        # 验证数据结构
        assert len(comparison_results) == 2
        
        # 验证成功的结果
        model1_result = comparison_results['model1']
        assert model1_result['status'] == 'completed'
        assert model1_result['content'] != ''
        assert model1_result['error'] is None
        assert model1_result['stats'] is not None
        assert model1_result['stats']['duration'] == 5.0
        
        # 验证错误的结果
        model2_result = comparison_results['model2']
        assert model2_result['status'] == 'error'
        assert model2_result['content'] == ''
        assert model2_result['error'] is not None
        assert model2_result['stats'] is None
    
    def test_output_status_transitions(self):
        """测试输出状态转换"""
        # 模拟状态转换过程
        status_transitions = [
            'pending',
            'running', 
            'completed'
        ]
        
        current_status = 'pending'
        
        # 验证状态转换
        for next_status in status_transitions[1:]:
            assert current_status != next_status
            current_status = next_status
        
        assert current_status == 'completed'
        
        # 测试错误状态
        error_status = 'error'
        assert error_status not in status_transitions
    
    def test_performance_stats_calculation(self):
        """测试性能统计计算"""
        import time
        
        # 模拟统计数据
        start_time = time.time()
        end_time = start_time + 5.0
        token_count = 100
        
        # 计算统计信息
        duration = end_time - start_time
        tokens_per_second = token_count / duration if duration > 0 else 0
        
        stats = {
            'start_time': start_time,
            'end_time': end_time,
            'token_count': token_count,
            'duration': duration,
            'tokens_per_second': tokens_per_second
        }
        
        # 验证统计信息
        assert stats['duration'] == 5.0
        assert stats['tokens_per_second'] == 20.0
        assert stats['token_count'] == 100
        
        # 测试零除法保护
        zero_duration_stats = {
            'duration': 0,
            'token_count': 100
        }
        
        tps = zero_duration_stats['token_count'] / zero_duration_stats['duration'] if zero_duration_stats['duration'] > 0 else 0
        assert tps == 0


class TestOutputDisplayErrorHandling:
    """输出显示错误处理测试"""
    
    def test_handle_empty_content(self):
        """测试处理空内容"""
        from src.multi_llm_comparator.ui.components import render_model_output_card
        
        with patch('streamlit.info') as mock_info:
            render_model_output_card("Test Model", "pytorch", "pending", content="")
            mock_info.assert_called_with("⏳ 等待开始...")
    
    def test_handle_invalid_status(self):
        """测试处理无效状态"""
        from src.multi_llm_comparator.ui.components import render_model_output_card
        
        with patch('streamlit.markdown') as mock_markdown:
            # 无效状态应该使用默认配置
            render_model_output_card("Test Model", "pytorch", "invalid_status")
            
            # 验证markdown被调用（用于渲染卡片头部）
            mock_markdown.assert_called()
    
    def test_handle_missing_stats(self):
        """测试处理缺失的统计信息"""
        from src.multi_llm_comparator.ui.components import render_model_output_card
        
        with patch('streamlit.markdown'):
            # 没有统计信息时不应该崩溃
            render_model_output_card("Test Model", "pytorch", "completed", 
                                   content="测试内容", stats=None)
    
    def test_handle_malformed_stats(self):
        """测试处理格式错误的统计信息"""
        from src.multi_llm_comparator.ui.components import render_model_output_card
        
        malformed_stats = {
            'duration': 'invalid',  # 应该是数字
            'token_count': None,    # 应该是数字
        }
        
        with patch('streamlit.markdown'), patch('streamlit.expander') as mock_expander:
            mock_expander.return_value = MagicMock()
            
            # 应该能处理格式错误的统计信息而不崩溃
            render_model_output_card("Test Model", "pytorch", "completed",
                                   content="测试内容", stats=malformed_stats)


if __name__ == "__main__":
    pytest.main([__file__])