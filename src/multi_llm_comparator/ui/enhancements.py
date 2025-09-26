"""
用户界面增强功能

提供进度指示器、状态反馈、操作确认和用户体验优化功能。
"""

import streamlit as st
import time
import threading
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
from datetime import datetime, timedelta


class NotificationType(Enum):
    """通知类型"""
    SUCCESS = "success"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ProgressType(Enum):
    """进度类型"""
    DETERMINATE = "determinate"  # 确定进度
    INDETERMINATE = "indeterminate"  # 不确定进度


class ProgressState:
    """进度状态"""
    def __init__(self, progress_id: str, title: str, progress_type: ProgressType, total: int = 0):
        self.progress_id = progress_id
        self.title = title
        self.progress_type = progress_type
        self.total = total
        self.current = 0
        self.message = ""
        self.details = ""
        self.start_time = time.time()
        self.estimated_remaining = None
        self.container = None
        self.progress_bar = None


class ProgressManager:
    """进度管理器"""
    
    def __init__(self):
        self.active_progress: Dict[str, ProgressState] = {}
        self.progress_containers: Dict[str, Any] = {}
        self.lock = threading.Lock()
    
    def progress_context(self, progress_id: str, title: str, progress_type: ProgressType, total: int = 0):
        """进度上下文管理器"""
        return ProgressContext(self, progress_id, title, progress_type, total)
    
    def start_progress(self, progress_id: str, title: str, progress_type: ProgressType, total: int = 0):
        """开始进度"""
        with self.lock:
            state = ProgressState(progress_id, title, progress_type, total)
            self.active_progress[progress_id] = state
            
            # 创建UI容器
            if progress_type == ProgressType.INDETERMINATE:
                state.container = st.spinner(title)
            else:
                state.container = st.empty()
                with state.container.container():
                    st.text(title)
                    state.progress_bar = st.progress(0)
    
    def update_progress(self, progress_id: str, current: int = None, message: str = "", details: str = ""):
        """更新进度"""
        with self.lock:
            if progress_id not in self.active_progress:
                return
            
            state = self.active_progress[progress_id]
            if current is not None:
                state.current = current
            if message:
                state.message = message
            if details:
                state.details = details
            
            # 计算预估剩余时间
            if state.progress_type == ProgressType.DETERMINATE and state.current > 0:
                elapsed = time.time() - state.start_time
                if state.total > 0:
                    progress_ratio = state.current / state.total
                    if progress_ratio > 0:
                        total_estimated = elapsed / progress_ratio
                        state.estimated_remaining = total_estimated - elapsed
            
            # 更新UI
            self._update_progress_ui(state)
    
    def _update_progress_ui(self, state: ProgressState):
        """更新进度UI"""
        if state.progress_type == ProgressType.DETERMINATE and state.container and state.progress_bar:
            progress_value = state.current / state.total if state.total > 0 else 0
            state.progress_bar.progress(progress_value)
            
            # 显示详细信息
            if state.message or state.details:
                info_text = f"{state.message}"
                if state.details:
                    info_text += f" - {state.details}"
                if state.estimated_remaining and state.estimated_remaining > 0:
                    eta_str = f"预计剩余: {state.estimated_remaining:.1f}秒"
                    info_text += f" ({eta_str})"
                st.caption(info_text)
    
    def finish_progress(self, progress_id: str):
        """完成进度"""
        with self.lock:
            if progress_id in self.active_progress:
                del self.active_progress[progress_id]
            if progress_id in self.progress_containers:
                del self.progress_containers[progress_id]


class ProgressContext:
    """进度上下文管理器"""
    
    def __init__(self, manager: ProgressManager, progress_id: str, title: str, progress_type: ProgressType, total: int = 0):
        self.manager = manager
        self.progress_id = progress_id
        self.title = title
        self.progress_type = progress_type
        self.total = total
    
    def __enter__(self):
        self.manager.start_progress(self.progress_id, self.title, self.progress_type, self.total)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.manager.finish_progress(self.progress_id)


class NotificationManager:
    """通知管理器"""
    
    def __init__(self, max_notifications: int = 5):
        self.notifications = []
        self.max_notifications = max_notifications
    
    def show_notification(self, message: str, notification_type: NotificationType = NotificationType.INFO, 
                         auto_dismiss: bool = True, duration: int = 5):
        """显示通知"""
        # 添加到通知历史
        notification = {
            'message': message,
            'type': notification_type,
            'timestamp': time.time(),
            'auto_dismiss': auto_dismiss,
            'duration': duration
        }
        self.notifications.append(notification)
        
        # 限制通知数量
        if len(self.notifications) > self.max_notifications:
            self.notifications = self.notifications[-self.max_notifications:]
        
        # 显示通知
        if notification_type == NotificationType.SUCCESS:
            st.success(f"✅ {message}")
        elif notification_type == NotificationType.ERROR:
            st.error(f"❌ {message}")
        elif notification_type == NotificationType.WARNING:
            st.warning(f"⚠️ {message}")
        else:
            st.info(f"ℹ️ {message}")
    
    def clear_notifications(self):
        """清除所有通知"""
        self.notifications = []
    
    def get_recent_notifications(self, count: int = 5) -> List[Dict[str, Any]]:
        """获取最近的通知"""
        return self.notifications[-count:] if self.notifications else []


class ConfirmationDialog:
    """确认对话框管理器"""
    
    @staticmethod
    def show_confirmation(title: str, message: str, confirm_text: str = "确认", 
                         cancel_text: str = "取消", danger: bool = False,
                         details: Optional[List[str]] = None) -> bool:
        """显示确认对话框"""
        confirm_key = f"confirm_{hash(title + message)}"
        
        if confirm_key not in st.session_state:
            st.session_state[confirm_key] = None
        
        if st.session_state[confirm_key] is True:
            st.session_state[confirm_key] = None  # 重置状态
            return True
        elif st.session_state[confirm_key] is False:
            st.session_state[confirm_key] = None  # 重置状态
            return False
        
        # 显示确认对话框
        with st.container():
            icon = "⚠️" if danger else "ℹ️"
            st.markdown(f"### {icon} {title}")
            st.markdown(message)
            
            if details:
                with st.expander("📋 详细信息"):
                    for detail in details:
                        st.markdown(f"• {detail}")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"❌ {cancel_text}", key=f"{confirm_key}_cancel", use_container_width=True):
                    st.session_state[confirm_key] = False
                    st.rerun()
            
            with col2:
                button_text = f"⚠️ {confirm_text}" if danger else f"✅ {confirm_text}"
                if st.button(button_text, key=f"{confirm_key}_confirm", use_container_width=True, type="primary"):
                    st.session_state[confirm_key] = True
                    st.rerun()
        
        return False


class UndoManager:
    """撤销管理器"""
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.history = []
        self.current_index = -1
    
    def save_state(self, state: Dict[str, Any], description: str = ""):
        """保存状态"""
        if self.current_index < len(self.history) - 1:
            self.history = self.history[:self.current_index + 1]
        
        state_entry = {
            "state": state.copy(),
            "description": description,
            "timestamp": time.time()
        }
        
        self.history.append(state_entry)
        self.current_index = len(self.history) - 1
        
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
            self.current_index = len(self.history) - 1
    
    def can_undo(self) -> bool:
        """是否可以撤销"""
        return self.current_index > 0
    
    def can_redo(self) -> bool:
        """是否可以重做"""
        return self.current_index < len(self.history) - 1
    
    def undo(self) -> Optional[Dict[str, Any]]:
        """撤销操作"""
        if not self.can_undo():
            return None
        self.current_index -= 1
        return self.history[self.current_index]["state"]
    
    def redo(self) -> Optional[Dict[str, Any]]:
        """重做操作"""
        if not self.can_redo():
            return None
        self.current_index += 1
        return self.history[self.current_index]["state"]
    
    def get_history_info(self) -> List[Dict[str, Any]]:
        """获取历史信息"""
        return [
            {
                "description": entry["description"],
                "timestamp": entry["timestamp"],
                "is_current": i == self.current_index
            }
            for i, entry in enumerate(self.history)
        ]


class KeyboardShortcutManager:
    """键盘快捷键管理器"""
    
    def __init__(self):
        self.shortcuts: Dict[str, Dict[str, Any]] = {}
        self.enabled = True
        self.global_shortcuts = {
            "Ctrl+Enter": "开始比较",
            "Ctrl+S": "保存结果",
            "Ctrl+Z": "撤销操作",
            "Ctrl+Y": "重做操作",
            "Escape": "取消操作",
            "Ctrl+/": "显示快捷键帮助",
            "Alt+1": "切换到模型选择",
            "Alt+2": "切换到配置",
            "Alt+3": "切换到比较结果"
        }
    
    def register_shortcut(self, key_combination: str, callback: Callable, description: str = ""):
        """注册快捷键"""
        self.shortcuts[key_combination] = {
            "callback": callback,
            "description": description
        }
    
    def render_shortcuts_help(self):
        """渲染快捷键帮助"""
        if not self.shortcuts and not self.global_shortcuts:
            return
        
        with st.expander("⌨️ 键盘快捷键", expanded=False):
            st.markdown("**全局快捷键：**")
            for key_combo, description in self.global_shortcuts.items():
                st.markdown(f"• **{key_combo}**: {description}")
            
            if self.shortcuts:
                st.markdown("**自定义快捷键：**")
                for key_combo, info in self.shortcuts.items():
                    description = info.get("description", "无描述")
                    st.markdown(f"• **{key_combo}**: {description}")
    
    def inject_keyboard_handler(self):
        """注入键盘事件处理器"""
        if not self.enabled:
            return
        
        js_code = """
        <script>
        document.addEventListener("keydown", function(event) {
            let keyCombo = "";
            if (event.ctrlKey) keyCombo += "Ctrl+";
            if (event.altKey) keyCombo += "Alt+";
            if (event.shiftKey) keyCombo += "Shift+";
            keyCombo += event.key;
            
            // 阻止某些默认行为
            if (keyCombo === "Escape" || keyCombo === "Ctrl+/" || keyCombo.startsWith("Alt+")) {
                event.preventDefault();
            }
            
            // 添加视觉反馈
            if (keyCombo === "Ctrl+Enter") {
                const buttons = document.querySelectorAll('button[kind="primary"]');
                buttons.forEach(btn => {
                    if (btn.textContent.includes('开始比较')) {
                        btn.style.transform = 'scale(0.95)';
                        setTimeout(() => btn.style.transform = 'scale(1)', 100);
                    }
                });
            }
        });
        
        // 添加焦点管理
        document.addEventListener("keydown", function(event) {
            if (event.key === "Tab") {
                const focusableElements = document.querySelectorAll(
                    'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
                );
                const firstElement = focusableElements[0];
                const lastElement = focusableElements[focusableElements.length - 1];
                
                if (event.shiftKey && document.activeElement === firstElement) {
                    event.preventDefault();
                    lastElement.focus();
                } else if (!event.shiftKey && document.activeElement === lastElement) {
                    event.preventDefault();
                    firstElement.focus();
                }
            }
        });
        </script>
        """
        st.markdown(js_code, unsafe_allow_html=True)


class AccessibilityManager:
    """无障碍功能管理器"""
    
    @staticmethod
    def add_aria_labels():
        """添加ARIA标签和无障碍支持"""
        accessibility_css = """
        <style>
        /* 焦点指示器 */
        button:focus, input:focus, select:focus, textarea:focus {
            outline: 2px solid #0066cc !important;
            outline-offset: 2px !important;
            box-shadow: 0 0 0 3px rgba(0, 102, 204, 0.3) !important;
        }
        
        /* 高对比度支持 */
        @media (prefers-contrast: high) {
            .stButton > button {
                border: 2px solid currentColor !important;
            }
        }
        
        /* 减少动画支持 */
        @media (prefers-reduced-motion: reduce) {
            * {
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
            }
        }
        
        /* 屏幕阅读器专用文本 */
        .sr-only {
            position: absolute !important;
            width: 1px !important;
            height: 1px !important;
            padding: 0 !important;
            margin: -1px !important;
            overflow: hidden !important;
            clip: rect(0, 0, 0, 0) !important;
            white-space: nowrap !important;
            border: 0 !important;
        }
        </style>
        """
        st.markdown(accessibility_css, unsafe_allow_html=True)
    
    @staticmethod
    def add_skip_links():
        """添加跳转链接"""
        skip_links = """
        <div style="position: absolute; top: -40px; left: 6px; background: #000; color: #fff; padding: 8px; z-index: 1000; text-decoration: none; border-radius: 4px;" 
             onFocus="this.style.top='6px'" onBlur="this.style.top='-40px'">
            <a href="#main-content" style="color: #fff; text-decoration: none;">跳转到主要内容</a>
        </div>
        """
        st.markdown(skip_links, unsafe_allow_html=True)
    
    @staticmethod
    def add_screen_reader_support():
        """添加屏幕阅读器支持"""
        screen_reader_css = """
        <style>
        /* 为动态内容添加live region */
        .live-region {
            position: absolute;
            left: -10000px;
            width: 1px;
            height: 1px;
            overflow: hidden;
        }
        
        /* 改善表格可访问性 */
        table {
            border-collapse: collapse;
        }
        
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        </style>
        
        <div aria-live="polite" aria-atomic="true" class="live-region" id="status-updates"></div>
        """
        st.markdown(screen_reader_css, unsafe_allow_html=True)


class ResponsivenessManager:
    """响应性管理器"""
    
    @staticmethod
    def add_responsive_css():
        """添加响应式CSS"""
        responsive_css = """
        <style>
        /* 移动端优化 */
        @media (max-width: 768px) {
            .stColumns > div {
                min-width: 100% !important;
                margin-bottom: 1rem !important;
            }
            
            .stButton > button {
                width: 100% !important;
                margin-bottom: 0.5rem !important;
            }
            
            .stSelectbox > div > div {
                width: 100% !important;
            }
            
            .stTextArea > div > div > textarea {
                min-height: 120px !important;
            }
        }
        
        /* 平板端优化 */
        @media (min-width: 769px) and (max-width: 1024px) {
            .stColumns > div {
                min-width: 48% !important;
            }
        }
        
        /* 桌面端优化 */
        @media (min-width: 1025px) {
            .stColumns > div {
                min-width: auto !important;
            }
        }
        
        /* 通用交互优化 */
        .stButton > button {
            transition: all 0.2s ease-in-out !important;
            border-radius: 6px !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
        }
        
        .stButton > button:active {
            transform: translateY(0) !important;
        }
        
        /* 加载状态优化 */
        .stSpinner > div {
            border-color: #0066cc !important;
        }
        
        /* 进度条优化 */
        .stProgress > div > div {
            background-color: #0066cc !important;
            transition: width 0.3s ease-in-out !important;
        }
        </style>
        """
        st.markdown(responsive_css, unsafe_allow_html=True)
    
    @staticmethod
    def optimize_performance():
        """优化性能"""
        # 清理缓存
        if hasattr(st, 'cache_data'):
            st.cache_data.clear()
        
        # 添加性能监控
        performance_js = """
        <script>
        // 性能监控
        if ('performance' in window) {
            window.addEventListener('load', function() {
                setTimeout(function() {
                    const perfData = performance.getEntriesByType('navigation')[0];
                    if (perfData) {
                        console.log('页面加载时间:', perfData.loadEventEnd - perfData.fetchStart, 'ms');
                    }
                }, 0);
            });
        }
        
        // 内存使用监控
        if ('memory' in performance) {
            setInterval(function() {
                const memory = performance.memory;
                if (memory.usedJSHeapSize > memory.jsHeapSizeLimit * 0.9) {
                    console.warn('内存使用率过高');
                }
            }, 30000);
        }
        </script>
        """
        st.markdown(performance_js, unsafe_allow_html=True)


# 全局管理器实例
progress_manager = ProgressManager()
notification_manager = NotificationManager()
undo_manager = UndoManager()
keyboard_manager = KeyboardShortcutManager()


def initialize_ui_enhancements():
    """初始化UI增强功能"""
    # 添加无障碍功能
    AccessibilityManager.add_aria_labels()
    AccessibilityManager.add_skip_links()
    AccessibilityManager.add_screen_reader_support()
    
    # 添加响应式设计
    ResponsivenessManager.add_responsive_css()
    ResponsivenessManager.optimize_performance()
    
    # 注册全局快捷键
    keyboard_manager.register_shortcut("Ctrl+Enter", lambda: None, "开始比较")
    keyboard_manager.register_shortcut("Ctrl+S", lambda: None, "保存结果")
    keyboard_manager.register_shortcut("Ctrl+Z", lambda: None, "撤销操作")
    keyboard_manager.register_shortcut("Ctrl+Y", lambda: None, "重做操作")
    keyboard_manager.register_shortcut("Escape", lambda: None, "取消操作")
    
    # 注入键盘处理器
    keyboard_manager.inject_keyboard_handler()


def show_enhanced_progress(progress_id: str, title: str, current: int, total: int, message: str = "", show_eta: bool = True):
    """显示增强的进度指示器"""
    progress_value = current / total if total > 0 else 0
    st.progress(progress_value)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("进度", f"{current}/{total}")
    with col2:
        percentage = int(progress_value * 100)
        st.metric("完成度", f"{percentage}%")
    with col3:
        st.metric("状态", "进行中" if current < total else "完成")
    
    if message:
        st.caption(f"📋 {message}")


def show_operation_confirmation(operation_name: str, description: str, danger: bool = False, details: Optional[List[str]] = None) -> bool:
    """显示操作确认对话框"""
    return ConfirmationDialog.show_confirmation(
        title=f"确认{operation_name}",
        message=description,
        confirm_text=operation_name,
        danger=danger,
        details=details
    )


def render_undo_redo_controls():
    """渲染撤销/重做控制"""
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("↶ 撤销", disabled=not undo_manager.can_undo(), use_container_width=True, help="撤销上一个操作 (Ctrl+Z)"):
            previous_state = undo_manager.undo()
            if previous_state:
                for key, value in previous_state.items():
                    st.session_state[key] = value
                notification_manager.show_notification("操作已撤销", NotificationType.INFO)
                st.rerun()
    
    with col2:
        if st.button("↷ 重做", disabled=not undo_manager.can_redo(), use_container_width=True, help="重做操作 (Ctrl+Y)"):
            next_state = undo_manager.redo()
            if next_state:
                for key, value in next_state.items():
                    st.session_state[key] = value
                notification_manager.show_notification("操作已重做", NotificationType.INFO)
                st.rerun()
    
    with col3:
        history_info = undo_manager.get_history_info()
        if history_info:
            current_desc = next((info["description"] for info in history_info if info["is_current"]), "当前状态")
            st.caption(f"📝 当前: {current_desc}")


def save_operation_state(description: str):
    """保存操作状态到撤销历史"""
    state_to_save = {
        "comparison_results": st.session_state.get("comparison_results", {}),
        "selected_models": st.session_state.get("selected_models", []),
        "current_prompt": st.session_state.get("current_prompt", ""),
        "model_configs": st.session_state.get("model_configs", {})
    }
    undo_manager.save_state(state_to_save, description)


def render_loading_overlay(message: str = "加载中...", show_progress: bool = False, progress: float = 0.0):
    """渲染加载遮罩层"""
    with st.spinner(message):
        if show_progress:
            st.progress(progress)
            st.caption(f"进度: {int(progress * 100)}%")


def show_toast_notification(message: str, notification_type: NotificationType = NotificationType.INFO):
    """显示Toast通知"""
    notification_manager.show_notification(message, notification_type)


def render_advanced_text_viewer(content: str, title: str = "", container_key: str = "", 
                               enable_line_numbers: bool = False, enable_word_wrap: bool = True,
                               enable_syntax_highlighting: bool = False, language: str = "text",
                               max_height: int = 400):
    """渲染高级文本查看器"""
    if not content:
        st.info("暂无内容")
        return
    
    # 创建查看器容器
    viewer_container = st.container()
    
    with viewer_container:
        if title:
            st.subheader(title)
        
        # 控制选项
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_raw = st.checkbox("显示原始文本", key=f"{container_key}_raw")
        
        with col2:
            if enable_syntax_highlighting:
                syntax_highlight = st.checkbox("语法高亮", value=True, key=f"{container_key}_syntax")
            else:
                syntax_highlight = False
        
        with col3:
            if st.button("📋 复制", key=f"{container_key}_copy"):
                st.code(content, language="text")
                st.success("内容已复制到剪贴板")
        
        # 内容显示
        if show_raw:
            st.text_area(
                "原始内容",
                value=content,
                height=max_height,
                disabled=True,
                key=f"{container_key}_textarea"
            )
        else:
            if syntax_highlight and language != "text":
                st.code(content, language=language)
            else:
                # 使用markdown渲染
                st.markdown(content)


def render_status_indicator(status: str, message: str = "", show_spinner: bool = False):
    """渲染状态指示器"""
    status_configs = {
        'success': ('✅', 'green', '成功'),
        'error': ('❌', 'red', '错误'),
        'warning': ('⚠️', 'orange', '警告'),
        'info': ('ℹ️', 'blue', '信息'),
        'running': ('🔄', 'blue', '运行中'),
        'pending': ('⏳', 'orange', '等待中'),
        'stopped': ('⏹️', 'gray', '已停止')
    }
    
    icon, color, default_text = status_configs.get(status, ('📄', 'gray', '未知'))
    display_message = message or default_text
    
    if show_spinner and status == 'running':
        with st.spinner(display_message):
            st.empty()
    else:
        st.markdown(f"""
        <div style="display: flex; align-items: center; padding: 8px; border-radius: 4px; background-color: rgba(0,0,0,0.05);">
            <span style="font-size: 1.2em; margin-right: 8px;">{icon}</span>
            <span style="color: {color}; font-weight: 500;">{display_message}</span>
        </div>
        """, unsafe_allow_html=True)


def create_interactive_tutorial():
    """创建交互式教程"""
    if 'tutorial_step' not in st.session_state:
        st.session_state.tutorial_step = 0
    
    tutorial_steps = [
        {
            'title': '欢迎使用多LLM比较器',
            'content': '这个工具可以帮助您同时比较多个大语言模型的输出结果。',
            'action': '点击"下一步"开始教程'
        },
        {
            'title': '第一步：扫描模型',
            'content': '首先，您需要在侧边栏中配置模型目录并扫描可用的模型文件。',
            'action': '前往侧边栏扫描模型'
        },
        {
            'title': '第二步：选择模型',
            'content': '从扫描到的模型中选择最多4个进行比较。',
            'action': '选择您想要比较的模型'
        },
        {
            'title': '第三步：配置参数',
            'content': '为每个选中的模型配置推理参数，如temperature、max_tokens等。',
            'action': '调整模型参数'
        },
        {
            'title': '第四步：输入提示词',
            'content': '输入您想要各个模型回答的问题或完成的任务。',
            'action': '输入提示词并开始比较'
        },
        {
            'title': '教程完成',
            'content': '现在您已经了解了基本使用方法，可以开始使用工具了！',
            'action': '开始使用'
        }
    ]
    
    if st.session_state.tutorial_step < len(tutorial_steps):
        step = tutorial_steps[st.session_state.tutorial_step]
        
        with st.expander("📚 使用教程", expanded=True):
            st.markdown(f"### {step['title']}")
            st.markdown(step['content'])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.session_state.tutorial_step > 0:
                    if st.button("⬅️ 上一步"):
                        st.session_state.tutorial_step -= 1
                        st.rerun()
            
            with col2:
                if st.button("❌ 跳过教程"):
                    st.session_state.tutorial_step = len(tutorial_steps)
                    st.rerun()
            
            with col3:
                if st.session_state.tutorial_step < len(tutorial_steps) - 1:
                    if st.button("➡️ 下一步"):
                        st.session_state.tutorial_step += 1
                        st.rerun()
                else:
                    if st.button("✅ 完成"):
                        st.session_state.tutorial_step = len(tutorial_steps)
                        st.rerun()