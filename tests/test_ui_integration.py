"""
UI增强功能集成测试

测试用户界面增强功能的集成场景和用户体验。
"""

import pytest
import streamlit as st
from unittest.mock import Mock, patch, MagicMock
import time
from typing import Dict, Any

from src.multi_llm_comparator.ui.enhancements import (
    initialize_ui_enhancements, show_enhanced_progress, show_operation_confirmation,
    render_undo_redo_controls, save_operation_state, render_loading_overlay,
    show_toast_notification, render_advanced_text_viewer, render_status_indicator,
    create_interactive_tutorial, NotificationType, ProgressType,
    progress_manager, notification_manager, undo_manager, keyboard_manager
)


class TestUIIntegration:
    """测试UI集成功能"""
    
    @patch('streamlit.session_state', {})
    def test_complete_ui_initialization(self):
        """测试完整的UI初始化流程"""
        with patch('streamlit.markdown') as mock_markdown:
            initialize_ui_enhancements()
            
            # 验证CSS和JavaScript被注入
            assert mock_markdown.call_count >= 3  # 至少包含无障碍、响应式、键盘处理
            
            # 验证快捷键被注册
            assert len(keyboard_manager.shortcuts) >= 5
    
    @patch('streamlit.progress')
    @patch('streamlit.columns')
    @patch('streamlit.metric')
    @patch('streamlit.caption')
    def test_enhanced_progress_with_eta(self, mock_caption, mock_metric, mock_columns, mock_progress):
        """测试带预估时间的增强进度指示器"""
        # 创建支持上下文管理器的Mock对象
        mock_col1 = Mock()
        mock_col1.__enter__ = Mock(return_value=mock_col1)
        mock_col1.__exit__ = Mock(return_value=None)
        mock_col2 = Mock()
        mock_col2.__enter__ = Mock(return_value=mock_col2)
        mock_col2.__exit__ = Mock(return_value=None)
        mock_col3 = Mock()
        mock_col3.__enter__ = Mock(return_value=mock_col3)
        mock_col3.__exit__ = Mock(return_value=None)
        
        mock_columns.return_value = [mock_col1, mock_col2, mock_col3]
        
        # 测试进度更新
        show_enhanced_progress("test_progress", "测试进度", 25, 100, "正在处理...", show_eta=True)
        
        mock_progress.assert_called_once_with(0.25)
        mock_caption.assert_called_once_with("📋 正在处理...")
        assert mock_metric.call_count == 3  # 进度、完成度、状态
    
    @patch('streamlit.container')
    @patch('streamlit.markdown')
    @patch('streamlit.columns')
    @patch('streamlit.button')
    @patch('streamlit.expander')
    @patch('streamlit.session_state', {})
    def test_operation_confirmation_with_details(self, mock_expander, mock_button, mock_columns, 
                                               mock_markdown, mock_container):
        """测试带详细信息的操作确认"""
        mock_container.return_value.__enter__ = Mock()
        mock_container.return_value.__exit__ = Mock()
        mock_columns.return_value = [Mock(), Mock()]
        mock_button.return_value = False
        mock_expander.return_value.__enter__ = Mock()
        mock_expander.return_value.__exit__ = Mock()
        
        details = ["将删除所有比较结果", "此操作不可撤销", "请确认您的选择"]
        result = show_operation_confirmation("删除结果", "确定要删除所有结果吗？", 
                                           danger=True, details=details)
        
        assert result is False
        mock_expander.assert_called_once_with("📋 详细信息")
    
    @patch('streamlit.session_state', {
        'comparison_results': {'model1': 'result1'},
        'selected_models': ['model1'],
        'current_prompt': 'test prompt'
    })
    @patch('src.multi_llm_comparator.ui.enhancements.undo_manager')
    def test_save_and_restore_operation_state(self, mock_undo_manager):
        """测试保存和恢复操作状态"""
        # 保存状态
        save_operation_state("测试操作")
        
        # 验证状态被保存
        mock_undo_manager.save_state.assert_called_once()
        saved_state = mock_undo_manager.save_state.call_args[0][0]
        
        assert 'comparison_results' in saved_state
        assert 'selected_models' in saved_state
        assert 'current_prompt' in saved_state
        assert saved_state['comparison_results'] == {'model1': 'result1'}
    
    @patch('streamlit.spinner')
    @patch('streamlit.progress')
    @patch('streamlit.caption')
    def test_loading_overlay_with_progress(self, mock_caption, mock_progress, mock_spinner):
        """测试带进度的加载遮罩层"""
        mock_spinner.return_value.__enter__ = Mock()
        mock_spinner.return_value.__exit__ = Mock()
        
        render_loading_overlay("正在加载模型...", show_progress=True, progress=0.7)
        
        mock_spinner.assert_called_once_with("正在加载模型...")
        mock_progress.assert_called_once_with(0.7)
        mock_caption.assert_called_once_with("进度: 70%")
    
    @patch('src.multi_llm_comparator.ui.enhancements.notification_manager')
    def test_toast_notification_types(self, mock_notification_manager):
        """测试不同类型的Toast通知"""
        # 测试成功通知
        show_toast_notification("操作成功", NotificationType.SUCCESS)
        mock_notification_manager.show_notification.assert_called_with("操作成功", NotificationType.SUCCESS)
        
        # 测试错误通知
        show_toast_notification("操作失败", NotificationType.ERROR)
        mock_notification_manager.show_notification.assert_called_with("操作失败", NotificationType.ERROR)
    
    @patch('streamlit.container')
    @patch('streamlit.subheader')
    @patch('streamlit.columns')
    @patch('streamlit.checkbox')
    @patch('streamlit.button')
    @patch('streamlit.text_area')
    @patch('streamlit.markdown')
    def test_advanced_text_viewer(self, mock_markdown, mock_text_area, mock_button, 
                                 mock_checkbox, mock_columns, mock_subheader, mock_container):
        """测试高级文本查看器"""
        mock_container.return_value.__enter__ = Mock()
        mock_container.return_value.__exit__ = Mock()
        
        # 创建支持上下文管理器的Mock对象
        mock_col1 = Mock()
        mock_col1.__enter__ = Mock(return_value=mock_col1)
        mock_col1.__exit__ = Mock(return_value=None)
        mock_col2 = Mock()
        mock_col2.__enter__ = Mock(return_value=mock_col2)
        mock_col2.__exit__ = Mock(return_value=None)
        mock_col3 = Mock()
        mock_col3.__enter__ = Mock(return_value=mock_col3)
        mock_col3.__exit__ = Mock(return_value=None)
        
        mock_columns.return_value = [mock_col1, mock_col2, mock_col3]
        mock_checkbox.side_effect = [False, True]  # show_raw=False, syntax_highlight=True
        mock_button.return_value = False
        
        content = "# 测试内容\n\n这是一个测试文档。"
        render_advanced_text_viewer(
            content=content,
            title="测试文档",
            container_key="test_viewer",
            enable_syntax_highlighting=True,
            language="markdown"
        )
        
        mock_subheader.assert_called_once_with("测试文档")
        # 验证markdown被调用（可能被调用多次）
        assert mock_markdown.called
    
    @patch('streamlit.markdown')
    def test_status_indicator_rendering(self, mock_markdown):
        """测试状态指示器渲染"""
        render_status_indicator("success", "操作完成", show_spinner=False)
        
        mock_markdown.assert_called_once()
        args = mock_markdown.call_args[0][0]
        assert "✅" in args  # 成功图标
        assert "操作完成" in args
        assert "green" in args  # 成功颜色
    
    def test_interactive_tutorial(self):
        """测试交互式教程"""
        with patch('streamlit.session_state') as mock_session_state, \
             patch('streamlit.expander') as mock_expander, \
             patch('streamlit.markdown') as mock_markdown, \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.button') as mock_button:
            
            # 设置session_state mock
            mock_session_state.get.return_value = 0  # tutorial_step = 0
            mock_session_state.__contains__ = Mock(return_value=False)  # 'tutorial_step' not in session_state
            mock_session_state.tutorial_step = 0
            
            mock_expander.return_value.__enter__ = Mock()
            mock_expander.return_value.__exit__ = Mock()
            
            # 创建支持上下文管理器的Mock对象
            mock_col1 = Mock()
            mock_col1.__enter__ = Mock(return_value=mock_col1)
            mock_col1.__exit__ = Mock(return_value=None)
            mock_col2 = Mock()
            mock_col2.__enter__ = Mock(return_value=mock_col2)
            mock_col2.__exit__ = Mock(return_value=None)
            mock_col3 = Mock()
            mock_col3.__enter__ = Mock(return_value=mock_col3)
            mock_col3.__exit__ = Mock(return_value=None)
            
            mock_columns.return_value = [mock_col1, mock_col2, mock_col3]
            mock_button.return_value = False
            
            create_interactive_tutorial()
            
            mock_expander.assert_called_once_with("📚 使用教程", expanded=True)
            # 验证markdown被调用
            assert mock_markdown.called


class TestProgressManager:
    """测试进度管理器集成"""
    
    def test_progress_context_manager(self):
        """测试进度上下文管理器"""
        manager = progress_manager
        
        with patch('streamlit.spinner') as mock_spinner:
            mock_spinner.return_value = Mock()
            
            with manager.progress_context("test_id", "测试进度", ProgressType.INDETERMINATE):
                assert "test_id" in manager.active_progress
                assert manager.active_progress["test_id"].title == "测试进度"
            
            # 上下文结束后应该清理
            assert "test_id" not in manager.active_progress
    
    def test_progress_update_with_eta(self):
        """测试带预估时间的进度更新"""
        manager = progress_manager
        
        with patch('streamlit.empty') as mock_empty, \
             patch('streamlit.text') as mock_text, \
             patch('streamlit.progress') as mock_progress:
            
            mock_container = Mock()
            mock_empty.return_value = mock_container
            mock_container.container.return_value.__enter__ = Mock()
            mock_container.container.return_value.__exit__ = Mock()
            mock_progress.return_value = Mock()
            
            # 开始确定进度
            manager.start_progress("eta_test", "测试ETA", ProgressType.DETERMINATE, 100)
            
            # 模拟进度更新
            time.sleep(0.1)  # 确保有时间差
            manager.update_progress("eta_test", 25, "进行中", "处理数据")
            
            state = manager.active_progress["eta_test"]
            assert state.current == 25
            assert state.message == "进行中"
            assert state.details == "处理数据"
            assert state.estimated_remaining is not None
            
            manager.finish_progress("eta_test")


class TestNotificationManager:
    """测试通知管理器集成"""
    
    def test_notification_history_management(self):
        """测试通知历史管理"""
        manager = notification_manager
        manager.clear_notifications()  # 清空历史
        
        with patch('streamlit.success'), patch('streamlit.error'), patch('streamlit.warning'):
            # 添加多个通知
            for i in range(7):  # 超过默认限制5个
                manager.show_notification(f"通知 {i}", NotificationType.INFO)
            
            # 应该只保留最新的5个
            assert len(manager.notifications) == 5
            assert manager.notifications[-1]['message'] == "通知 6"
            
            # 测试获取最近通知
            recent = manager.get_recent_notifications(3)
            assert len(recent) == 3
            assert recent[-1]['message'] == "通知 6"


class TestKeyboardShortcutManager:
    """测试键盘快捷键管理器集成"""
    
    def test_shortcut_registration_and_help(self):
        """测试��捷键注册和帮助显示"""
        manager = keyboard_manager
        
        # 注册自定义快捷键
        callback = Mock()
        manager.register_shortcut("Ctrl+T", callback, "测试快捷键")
        
        assert "Ctrl+T" in manager.shortcuts
        assert manager.shortcuts["Ctrl+T"]["description"] == "测试快捷键"
        
        # 测试帮助渲染
        with patch('streamlit.expander') as mock_expander, \
             patch('streamlit.markdown') as mock_markdown:
            
            mock_expander.return_value.__enter__ = Mock()
            mock_expander.return_value.__exit__ = Mock()
            
            manager.render_shortcuts_help()
            
            mock_expander.assert_called_once_with("⌨️ 键盘快捷键", expanded=False)
            # 应该显示全局快捷键和自定义快捷键
            markdown_calls = [call[0][0] for call in mock_markdown.call_args_list]
            assert any("全局快捷键" in call for call in markdown_calls)
            assert any("自定义快捷键" in call for call in markdown_calls)
    
    @patch('streamlit.markdown')
    def test_keyboard_handler_injection(self, mock_markdown):
        """测试键盘处理器注入"""
        manager = keyboard_manager
        manager.inject_keyboard_handler()
        
        mock_markdown.assert_called_once()
        js_code = mock_markdown.call_args[0][0]
        
        # 验证JavaScript代码包含必要的功能
        assert '<script>' in js_code
        assert 'keydown' in js_code
        assert 'Ctrl+Enter' in js_code
        assert 'preventDefault' in js_code
        assert 'focus' in js_code  # 焦点管理


class TestAccessibilityFeatures:
    """测试无障碍功能"""
    
    @patch('streamlit.markdown')
    def test_accessibility_css_injection(self, mock_markdown):
        """测试无障碍CSS注入"""
        from src.multi_llm_comparator.ui.enhancements import AccessibilityManager
        
        AccessibilityManager.add_aria_labels()
        
        mock_markdown.assert_called_once()
        css_code = mock_markdown.call_args[0][0]
        
        # 验证无障碍CSS包含必要的样式
        assert 'focus' in css_code
        assert 'outline' in css_code
        assert 'prefers-contrast' in css_code
        assert 'prefers-reduced-motion' in css_code
        assert 'sr-only' in css_code
    
    @patch('streamlit.markdown')
    def test_skip_links_addition(self, mock_markdown):
        """测试跳转链接添加"""
        from src.multi_llm_comparator.ui.enhancements import AccessibilityManager
        
        AccessibilityManager.add_skip_links()
        
        mock_markdown.assert_called_once()
        html_code = mock_markdown.call_args[0][0]
        
        assert '跳转到主要内容' in html_code
        assert '#main-content' in html_code
    
    @patch('streamlit.markdown')
    def test_screen_reader_support(self, mock_markdown):
        """测试屏幕阅读器支持"""
        from src.multi_llm_comparator.ui.enhancements import AccessibilityManager
        
        AccessibilityManager.add_screen_reader_support()
        
        mock_markdown.assert_called_once()
        html_code = mock_markdown.call_args[0][0]
        
        assert 'aria-live' in html_code
        assert 'live-region' in html_code
        assert 'status-updates' in html_code


class TestResponsiveDesign:
    """测试响应式设计"""
    
    @patch('streamlit.markdown')
    def test_responsive_css_injection(self, mock_markdown):
        """测试响应式CSS注入"""
        from src.multi_llm_comparator.ui.enhancements import ResponsivenessManager
        
        ResponsivenessManager.add_responsive_css()
        
        mock_markdown.assert_called_once()
        css_code = mock_markdown.call_args[0][0]
        
        # 验证响应式CSS包含必要的媒体查询
        assert '@media (max-width: 768px)' in css_code  # 移动端
        assert '@media (min-width: 769px) and (max-width: 1024px)' in css_code  # 平板端
        assert '@media (min-width: 1025px)' in css_code  # 桌面端
        assert 'transition' in css_code  # 动画效果
        assert 'hover' in css_code  # 悬停效果
    
    @patch('streamlit.cache_data')
    @patch('streamlit.markdown')
    def test_performance_optimization(self, mock_markdown, mock_cache_data):
        """测试性能优化"""
        from src.multi_llm_comparator.ui.enhancements import ResponsivenessManager
        
        mock_cache_data.clear = Mock()
        
        ResponsivenessManager.optimize_performance()
        
        # 验证缓存被清理
        mock_cache_data.clear.assert_called_once()
        
        # 验证性能监控JavaScript被注入
        mock_markdown.assert_called_once()
        js_code = mock_markdown.call_args[0][0]
        
        assert 'performance' in js_code
        assert 'memory' in js_code
        assert 'console.log' in js_code


if __name__ == "__main__":
    pytest.main([__file__])