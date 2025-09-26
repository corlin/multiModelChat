"""
UI增强功能简化测试

测试用户界面增强功能的核心组件。
"""

import pytest
from unittest.mock import Mock, patch
import time

from src.multi_llm_comparator.ui.enhancements import (
    NotificationManager, UndoManager, KeyboardShortcutManager,
    NotificationType, initialize_ui_enhancements,
    show_enhanced_progress, show_operation_confirmation,
    save_operation_state, show_toast_notification
)


class TestNotificationManager:
    """测试通知管理器"""
    
    def test_notification_manager_initialization(self):
        """测试通知管理器初始化"""
        manager = NotificationManager()
        assert manager.notifications == []
    
    @patch('streamlit.success')
    def test_show_success_notification(self, mock_success):
        """测试显示成功通知"""
        manager = NotificationManager()
        manager.show_notification("测试成功", NotificationType.SUCCESS)
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


class TestUIEnhancementFunctions:
    """测试UI增强功能函数"""
    
    @patch('streamlit.progress')
    @patch('streamlit.columns')
    @patch('streamlit.metric')
    @patch('streamlit.caption')
    def test_show_enhanced_progress(self, mock_caption, mock_metric, mock_columns, mock_progress):
        """测试显示增强进度指示器"""
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
        
        show_enhanced_progress("test_id", "测试", 25, 100, "进行中")
        
        mock_progress.assert_called_once_with(0.25)
        mock_caption.assert_called_once_with("📋 进行中")
    
    @patch('streamlit.container')
    @patch('streamlit.markdown')
    @patch('streamlit.columns')
    @patch('streamlit.button')
    @patch('streamlit.session_state', {})
    def test_show_operation_confirmation(self, mock_button, mock_columns, mock_markdown, mock_container):
        """测试显示操作确认对话框"""
        mock_container.return_value.__enter__ = Mock()
        mock_container.return_value.__exit__ = Mock()
        mock_columns.return_value = [Mock(), Mock()]
        mock_button.return_value = False
        
        result = show_operation_confirmation("删除", "确定要删除吗？", danger=True)
        
        assert result is False
        mock_markdown.assert_called()
    
    @patch('streamlit.session_state', {})
    @patch('src.multi_llm_comparator.ui.enhancements.undo_manager')
    def test_save_operation_state(self, mock_undo_manager):
        """测试保存操作状态"""
        save_operation_state("测试操作")
        
        mock_undo_manager.save_state.assert_called_once()
        args = mock_undo_manager.save_state.call_args
        # 检查第二个参数（description）
        assert args[0][1] == "测试操作"  # args是(positional_args, keyword_args)的元组
    
    @patch('src.multi_llm_comparator.ui.enhancements.notification_manager')
    def test_show_toast_notification(self, mock_notification_manager):
        """测试显示Toast通知"""
        show_toast_notification("测试消息", NotificationType.SUCCESS)
        
        mock_notification_manager.show_notification.assert_called_once_with(
            "测试消息", NotificationType.SUCCESS
        )


class TestInitializeUIEnhancements:
    """测试UI增强功能初始化"""
    
    @patch('streamlit.markdown')
    @patch('src.multi_llm_comparator.ui.enhancements.keyboard_manager')
    def test_initialize_ui_enhancements(self, mock_keyboard_manager, mock_markdown):
        """测试初始化UI增强功能"""
        initialize_ui_enhancements()
        
        # 验证CSS被添加
        mock_markdown.assert_called()
        
        # 验证快捷键注册
        assert mock_keyboard_manager.register_shortcut.call_count >= 5
        mock_keyboard_manager.inject_keyboard_handler.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])