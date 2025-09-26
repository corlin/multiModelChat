"""
UI增强功能测试

测试用户界面增强功能的各个组件。
"""

import pytest
import streamlit as st
from unittest.mock import Mock, patch, MagicMock
import time
from typing import Dict, Any

from src.multi_llm_comparator.ui.enhancements import (
    ProgressManager, NotificationManager, ConfirmationDialog, UndoManager,
    KeyboardShortcutManager, AccessibilityManager, ResponsivenessManager,
    ProgressType, NotificationType, initialize_ui_enhancements,
    show_enhanced_progress, show_operation_confirmation, render_undo_redo_controls,
    save_operation_state, render_loading_overlay, show_toast_notification,
    render_advanced_text_viewer, render_status_indicator, create_interactive_tutorial
)


class TestProgressManager:
    """测试进度管理器"""
    
    def test_progress_manager_initialization(self):
        """测试进度管理器初始化"""
        manager = ProgressManager()
        assert manager.active_progress == {}
        assert manager.progress_containers == {}
        assert manager.lock is not None
    
    @patch('streamlit.progress')
    @patch('streamlit.spinner')
    def test_progress_context_indeterminate(self, mock_spinner, mock_progress):
        """测试不确定进度上下文"""
        manager = ProgressManager()
        mock_spinner.return_value = Mock()
        
        with manager.progress_context("test_id", "测试进度", ProgressType.INDETERMINATE):
            assert "test_id" in manager.active_progress
            assert manager.active_progress["test_id"].message == "测试进度"
            mock_spinner.assert_called_once_with("测试进度")
        
        # 上下文结束后应该清理
        assert "test_id" not in manager.active_progress
    
    @patch('streamlit.progress')
    def test_progress_context_determinate(self, mock_progress):
        """测试确定进度上下文"""
        manager = ProgressManager()
        mock_progress.return_value = Mock()
        
        with manager.progress_context("test_id", "测试进度", ProgressType.DETERMINATE, 50):
            assert "test_id" in manager.active_progress
            assert manager.active_progress["test_id"].total == 50
            mock_progress.assert_called_once_with(0)
        
        assert "test_id" not in manager.active_progress
    
    @patch('streamlit.caption')
    def test_update_progress(self, mock_caption):
        """测试进度更新"""
        manager = ProgressManager()
        
        # 先创建进度状态
        with manager.progress_context("test_id", "测试", ProgressType.DETERMINATE, 100):
            manager.update_progress("test_id", 25, "进行中", "详细信息")
            
            state = manager.active_progress["test_id"]
            assert state.current == 25
            assert state.message == "进行中"
            assert state.details == "详细信息"
            assert state.estimated_remaining is not None
    
    def test_update_nonexistent_progress(self):
        """测试更新不存在的进度"""
        manager = ProgressManager()
        # 不应该抛出异常
        manager.update_progress("nonexistent", 50)


class TestNotificationManager:
    """测试通知管理器"""
    
    def test_notification_manager_initialization(self):
        """测试通知管理器初始化"""
        manager = NotificationManager()
        assert manager.notifications == []
        assert manager.max_notifications == 5
    
    @patch('streamlit.success')
    def test_show_success_notification(self, mock_success):
        """测试显示成功通知"""
        manager = NotificationManager()
        manager.show_notification("测试成功", NotificationType.SUCCESS)
        
        assert len(manager.notifications) == 1
        notification = manager.notifications[0]
        assert notification['message'] == "测试成功"
        assert notification['type'] == NotificationType.SUCCESS
        mock_success.assert_called_once_with("✅ 测试成功")
    
    @patch('streamlit.error')
    def test_show_error_notification(self, mock_error):
        """测试显示错误通知"""
        manager = NotificationManager()
        manager.show_notification("测试错误", NotificationType.ERROR)
        
        mock_error.assert_called_once_with("❌ 测试错误")
    
    @patch('streamlit.warning')
    def test_show_warning_notification(self, mock_warning):
        """测试显示警告通知"""
        manager = NotificationManager()
        manager.show_notification("测试警告", NotificationType.WARNING)
        
        mock_warning.assert_called_once_with("⚠️ 测试警告")
    
    @patch('streamlit.info')
    def test_show_info_notification(self, mock_info):
        """测试显示信息通知"""
        manager = NotificationManager()
        manager.show_notification("测试信息", NotificationType.INFO)
        
        mock_info.assert_called_once_with("ℹ️ 测试信息")
    
    def test_notification_limit(self):
        """测试通知数量限制"""
        manager = NotificationManager()
        
        # 添加超过限制的通知
        for i in range(10):
            manager.show_notification(f"通知 {i}")
        
        assert len(manager.notifications) == manager.max_notifications
        # 应该保留最新的通知
        assert manager.notifications[-1]['message'] == "通知 9"
    
    def test_clear_notifications(self):
        """测试清除通知"""
        manager = NotificationManager()
        manager.show_notification("测试通知")
        assert len(manager.notifications) == 1
        
        manager.clear_notifications()
        assert len(manager.notifications) == 0


class TestUndoManager:
    """测试撤销管理器"""
    
    def test_undo_manager_initialization(self):
        """测试撤销管理器初始化"""
        manager = UndoManager(max_history=5)
        assert manager.max_history == 5
        assert manager.history == []
        assert manager.current_index == -1
    
    def test_save_state(self):
        """测试保存状态"""
        manager = UndoManager()
        state = {'key': 'value'}
        
        manager.save_state(state, "测试状态")
        
        assert len(manager.history) == 1
        assert manager.current_index == 0
        assert manager.history[0]['state'] == state
        assert manager.history[0]['description'] == "测试状态"
    
    def test_undo_redo_operations(self):
        """测试撤销和重做操作"""
        manager = UndoManager()
        
        # 保存两个状态
        state1 = {'step': 1}
        state2 = {'step': 2}
        
        manager.save_state(state1, "状态1")
        manager.save_state(state2, "状态2")
        
        assert manager.current_index == 1
        assert manager.can_undo()
        assert not manager.can_redo()
        
        # 撤销到状态1
        undone_state = manager.undo()
        assert undone_state == state1
        assert manager.current_index == 0
        assert manager.can_redo()
        
        # 重做到状态2
        redone_state = manager.redo()
        assert redone_state == state2
        assert manager.current_index == 1
    
    def test_history_limit(self):
        """测试历史记录限制"""
        manager = UndoManager(max_history=3)
        
        # 添加超过限制的状态
        for i in range(5):
            manager.save_state({'step': i}, f"状态{i}")
        
        assert len(manager.history) == 3
        assert manager.current_index == 2
        # 应该保留最新的状态
        assert manager.history[-1]['state']['step'] == 4
    
    def test_get_history_info(self):
        """测试获取历史信息"""
        manager = UndoManager()
        
        manager.save_state({'step': 1}, "状态1")
        manager.save_state({'step': 2}, "状态2")
        
        history_info = manager.get_history_info()
        assert len(history_info) == 2
        assert history_info[0]['description'] == "状态1"
        assert history_info[1]['description'] == "状态2"
        assert history_info[1]['is_current']  # 当前状态


class TestKeyboardShortcutManager:
    """测试键盘快捷键管理器"""
    
    def test_keyboard_manager_initialization(self):
        """测试键盘管理器初始化"""
        manager = KeyboardShortcutManager()
        assert manager.shortcuts == {}
        assert manager.enabled is True
    
    def test_register_shortcut(self):
        """测试注册快捷键"""
        manager = KeyboardShortcutManager()
        callback = Mock()
        
        manager.register_shortcut("Ctrl+S", callback, "保存")
        
        assert "Ctrl+S" in manager.shortcuts
        assert manager.shortcuts["Ctrl+S"]['callback'] == callback
        assert manager.shortcuts["Ctrl+S"]['description'] == "保存"
    
    @patch('streamlit.expander')
    @patch('streamlit.markdown')
    def test_render_shortcuts_help(self, mock_markdown, mock_expander):
        """测试渲染快捷键帮助"""
        manager = KeyboardShortcutManager()
        manager.register_shortcut("Ctrl+S", Mock(), "保存")
        
        mock_expander.return_value.__enter__ = Mock()
        mock_expander.return_value.__exit__ = Mock()
        
        manager.render_shortcuts_help()
        
        mock_expander.assert_called_once()
    
    @patch('streamlit.markdown')
    def test_inject_keyboard_handler(self, mock_markdown):
        """测试注入键盘处理器"""
        manager = KeyboardShortcutManager()
        manager.register_shortcut("Ctrl+S", Mock(), "保存")
        
        manager.inject_keyboard_handler()
        
        mock_markdown.assert_called_once()
        # 检查是否包含JavaScript代码
        args = mock_markdown.call_args[0]
        assert '<script>' in args[0]
        assert 'keydown' in args[0]


class TestAccessibilityManager:
    """测试无障碍功能管理器"""
    
    @patch('streamlit.markdown')
    def test_add_aria_labels(self, mock_markdown):
        """测试添加ARIA标签"""
        AccessibilityManager.add_aria_labels()
        
        mock_markdown.assert_called_once()
        args = mock_markdown.call_args[0]
        assert '<style>' in args[0]
        assert 'focus' in args[0]
        assert 'outline' in args[0]
    
    @patch('streamlit.markdown')
    def test_add_skip_links(self, mock_markdown):
        """测试添加跳转链接"""
        AccessibilityManager.add_skip_links()
        
        mock_markdown.assert_called_once()
        args = mock_markdown.call_args[0]
        assert '跳转到主要内容' in args[0]
    
    @patch('streamlit.markdown')
    def test_add_screen_reader_support(self, mock_markdown):
        """测试添加屏幕阅读器支持"""
        AccessibilityManager.add_screen_reader_support()
        
        mock_markdown.assert_called_once()
        args = mock_markdown.call_args[0]
        assert 'sr-only' in args[0]


class TestResponsivenessManager:
    """测试响应性管理器"""
    
    @patch('streamlit.markdown')
    def test_add_responsive_css(self, mock_markdown):
        """测试添加响应式CSS"""
        ResponsivenessManager.add_responsive_css()
        
        mock_markdown.assert_called_once()
        args = mock_markdown.call_args[0]
        assert '@media' in args[0]
        assert 'max-width' in args[0]
        assert 'transition' in args[0]
    
    @patch('streamlit.cache_data')
    @patch('streamlit.markdown')
    def test_optimize_performance(self, mock_markdown, mock_cache):
        """测试性能优化"""
        with patch('streamlit.session_state', {}):
            ResponsivenessManager.optimize_performance()
            
            mock_cache.clear.assert_called_once()
            mock_markdown.assert_called_once()


class TestUIEnhancementFunctions:
    """测试UI增强功能函数"""
    
    @patch('streamlit.progress')
    @patch('streamlit.columns')
    @patch('streamlit.metric')
    @patch('streamlit.caption')
    def test_show_enhanced_progress(self, mock_caption, mock_metric, mock_columns, mock_progress):
        """测试显示增强进度指示器"""
        mock_columns.return_value = [Mock(), Mock(), Mock()]
        
        show_enhanced_progress("test_id", "测试", 25, 100, "进行中")
        
        mock_progress.assert_called_once_with(0.25)
        mock_caption.assert_called_once_with("📋 进行中")
    
    @patch('streamlit.container')
    @patch('streamlit.markdown')
    @patch('streamlit.columns')
    @patch('streamlit.button')
    def test_show_operation_confirmation(self, mock_button, mock_columns, mock_markdown, mock_container):
        """测试显示操作确认对话框"""
        mock_container.return_value.__enter__ = Mock()
        mock_container.return_value.__exit__ = Mock()
        mock_columns.return_value = [Mock(), Mock()]
        mock_button.return_value = False
        
        result = show_operation_confirmation("删除", "确定要删除吗？", danger=True)
        
        assert result is False
        mock_markdown.assert_called()
    
    @patch('src.multi_llm_comparator.ui.enhancements.undo_manager')
    @patch('streamlit.columns')
    @patch('streamlit.button')
    @patch('streamlit.caption')
    def test_render_undo_redo_controls(self, mock_caption, mock_button, mock_columns, mock_undo_manager):
        """测试渲染撤销/重做控制"""
        mock_columns.return_value = [Mock(), Mock(), Mock()]
        mock_button.return_value = False
        mock_undo_manager.can_undo.return_value = True
        mock_undo_manager.can_redo.return_value = False
        mock_undo_manager.get_history_info.return_value = [
            {'description': '测试状态', 'is_current': True}
        ]
        
        render_undo_redo_controls()
        
        # 应该调用两次button（撤销和重做）
        assert mock_button.call_count == 2
    
    @patch('streamlit.session_state', {})
    @patch('src.multi_llm_comparator.ui.enhancements.undo_manager')
    def test_save_operation_state(self, mock_undo_manager):
        """测试保存操作状态"""
        save_operation_state("测试操作")
        
        mock_undo_manager.save_state.assert_called_once()
        args = mock_undo_manager.save_state.call_args
        assert args[1] == "测试操作"  # description参数
    
    @patch('streamlit.spinner')
    @patch('streamlit.progress')
    @patch('streamlit.caption')
    def test_render_loading_overlay(self, mock_caption, mock_progress, mock_spinner):
        """测试渲染加载遮罩层"""
        mock_spinner.return_value.__enter__ = Mock()
        mock_spinner.return_value.__exit__ = Mock()
        
        render_loading_overlay("加载中", show_progress=True, progress=0.5)
        
        mock_spinner.assert_called_once_with("加载中")
        mock_progress.assert_called_once_with(0.5)
        mock_caption.assert_called_once_with("进度: 50%")
    
    @patch('src.multi_llm_comparator.ui.enhancements.notification_manager')
    def test_show_toast_notification(self, mock_notification_manager):
        """测试显示Toast通知"""
        show_toast_notification("测试消息", NotificationType.SUCCESS)
        
        mock_notification_manager.show_notification.assert_called_once_with(
            "测试消息", NotificationType.SUCCESS
        )


class TestInitializeUIEnhancements:
    """测试UI增强功能初始化"""
    
    @patch('src.multi_llm_comparator.ui.enhancements.AccessibilityManager')
    @patch('src.multi_llm_comparator.ui.enhancements.ResponsivenessManager')
    @patch('src.multi_llm_comparator.ui.enhancements.keyboard_manager')
    def test_initialize_ui_enhancements(self, mock_keyboard_manager, mock_responsiveness, mock_accessibility):
        """测试初始化UI增强功能"""
        initialize_ui_enhancements()
        
        # 验证各个管理器的方法被调用
        mock_accessibility.add_aria_labels.assert_called_once()
        mock_accessibility.add_skip_links.assert_called_once()
        mock_accessibility.add_screen_reader_support.assert_called_once()
        
        mock_responsiveness.add_responsive_css.assert_called_once()
        mock_responsiveness.optimize_performance.assert_called_once()
        
        # 验证快捷键注册
        assert mock_keyboard_manager.register_shortcut.call_count >= 5
        mock_keyboard_manager.inject_keyboard_handler.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])