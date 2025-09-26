"""
UI工具函数

Streamlit界面的辅助工具函数。
"""

import streamlit as st
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import json


def format_file_size(size_bytes: int) -> str:
    """
    格式化文件大小
    
    Args:
        size_bytes: 文件大小（字节）
        
    Returns:
        格式化的文件大小字符串
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def format_duration(seconds: float) -> str:
    """
    格式化时间长度
    
    Args:
        seconds: 秒数
        
    Returns:
        格式化的时间字符串
    """
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.0f}s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        return f"{hours}h {remaining_minutes}m"


def format_tokens_per_second(tokens: int, seconds: float) -> str:
    """
    格式化token生成速度
    
    Args:
        tokens: token数量
        seconds: 时间（秒）
        
    Returns:
        格式化的速度字符串
    """
    if seconds <= 0:
        return "N/A"
    
    tps = tokens / seconds
    return f"{tps:.1f} tokens/s"


def safe_get_nested_value(data: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    """
    安全地获取嵌套字典的值
    
    Args:
        data: 数据字典
        keys: 键路径列表
        default: 默认值
        
    Returns:
        获取到的值或默认值
    """
    current = data
    try:
        for key in keys:
            current = current[key]
        return current
    except (KeyError, TypeError):
        return default


def create_download_link(data: str, filename: str, mime_type: str = "text/plain") -> None:
    """
    创建下载链接
    
    Args:
        data: 要下载的数据
        filename: 文件名
        mime_type: MIME类型
    """
    st.download_button(
        label=f"📥 下载 {filename}",
        data=data,
        file_name=filename,
        mime=mime_type,
        use_container_width=True
    )


def show_json_viewer(data: Dict[str, Any], title: str = "JSON数据") -> None:
    """
    显示JSON查看器
    
    Args:
        data: JSON数据
        title: 标题
    """
    with st.expander(f"🔍 {title}"):
        st.json(data)


def show_code_block(code: str, language: str = "text", title: str = "") -> None:
    """
    显示代码块
    
    Args:
        code: 代码内容
        language: 语言类型
        title: 标题
    """
    if title:
        st.subheader(title)
    
    st.code(code, language=language)


def create_two_column_layout(left_content: callable, right_content: callable, 
                           left_ratio: float = 0.5) -> None:
    """
    创建两列布局
    
    Args:
        left_content: 左列内容函数
        right_content: 右列内容函数
        left_ratio: 左列宽度比例
    """
    right_ratio = 1.0 - left_ratio
    col1, col2 = st.columns([left_ratio, right_ratio])
    
    with col1:
        left_content()
    
    with col2:
        right_content()


def create_tabs_layout(tab_configs: List[Tuple[str, callable]]) -> None:
    """
    创建标签页布局
    
    Args:
        tab_configs: 标签页配置列表，每个元素为(标签名, 内容函数)
    """
    if not tab_configs:
        return
    
    tab_names = [config[0] for config in tab_configs]
    tabs = st.tabs(tab_names)
    
    for tab, (_, content_func) in zip(tabs, tab_configs):
        with tab:
            content_func()


def show_loading_spinner(message: str = "加载中...") -> Any:
    """
    显示加载旋转器
    
    Args:
        message: 加载消息
        
    Returns:
        Streamlit spinner上下文管理器
    """
    return st.spinner(message)


def show_success_message(message: str, icon: str = "✅") -> None:
    """
    显示成功消息
    
    Args:
        message: 消息内容
        icon: 图标
    """
    st.success(f"{icon} {message}")


def show_error_message(message: str, icon: str = "❌") -> None:
    """
    显示错误消息
    
    Args:
        message: 消息内容
        icon: 图标
    """
    st.error(f"{icon} {message}")


def show_warning_message(message: str, icon: str = "⚠️") -> None:
    """
    显示警告消息
    
    Args:
        message: 消息内容
        icon: 图标
    """
    st.warning(f"{icon} {message}")


def show_info_message(message: str, icon: str = "ℹ️") -> None:
    """
    显示信息消息
    
    Args:
        message: 消息内容
        icon: 图标
    """
    st.info(f"{icon} {message}")


def create_metric_card(title: str, value: Any, delta: Optional[Any] = None, 
                      help_text: Optional[str] = None) -> None:
    """
    创建指标卡片
    
    Args:
        title: 标题
        value: 值
        delta: 变化量
        help_text: 帮助文本
    """
    st.metric(
        label=title,
        value=value,
        delta=delta,
        help=help_text
    )


def create_progress_bar(progress: float, text: Optional[str] = None) -> None:
    """
    创建进度条
    
    Args:
        progress: 进度值 (0.0 - 1.0)
        text: 进度文本
    """
    st.progress(progress)
    if text:
        st.caption(text)


def validate_session_state_key(key: str) -> bool:
    """
    验证会话状态键是否存在
    
    Args:
        key: 会话状态键
        
    Returns:
        是否存在
    """
    return key in st.session_state


def get_session_state_value(key: str, default: Any = None) -> Any:
    """
    获取会话状态值
    
    Args:
        key: 会话状态键
        default: 默认值
        
    Returns:
        会话状态值或默认值
    """
    return st.session_state.get(key, default)


def set_session_state_value(key: str, value: Any) -> None:
    """
    设置会话状态值
    
    Args:
        key: 会话状态键
        value: 值
    """
    st.session_state[key] = value


def clear_session_state_key(key: str) -> None:
    """
    清除会话状态键
    
    Args:
        key: 会话状态键
    """
    if key in st.session_state:
        del st.session_state[key]


def get_current_timestamp() -> str:
    """
    获取当前时间戳字符串
    
    Returns:
        格式化的时间戳
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def create_collapsible_section(title: str, content_func: callable, 
                              expanded: bool = False) -> None:
    """
    创建可折叠区域
    
    Args:
        title: 标题
        content_func: 内容函数
        expanded: 是否默认展开
    """
    with st.expander(title, expanded=expanded):
        content_func()


def create_sidebar_section(title: str, content_func: callable) -> None:
    """
    创建侧边栏区域
    
    Args:
        title: 标题
        content_func: 内容函数
    """
    with st.sidebar:
        st.subheader(title)
        content_func()


def format_model_display_name(model_name: str, model_type: str, size_bytes: int) -> str:
    """
    格式化模型显示名称
    
    Args:
        model_name: 模型名称
        model_type: 模型类型
        size_bytes: 文件大小
        
    Returns:
        格式化的显示名称
    """
    size_str = format_file_size(size_bytes)
    return f"{model_name} ({model_type.upper()}) - {size_str}"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    截断文本
    
    Args:
        text: 原始文本
        max_length: 最大长度
        suffix: 后缀
        
    Returns:
        截断后的文本
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix