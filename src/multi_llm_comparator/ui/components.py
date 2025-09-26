"""
UI组件

Streamlit界面的可重用组件。
"""

import streamlit as st
import re
from typing import List, Dict, Any, Optional
from ..core.models import ModelInfo, ModelType, ModelConfig
from ..core.exceptions import ValidationError


def render_model_card(model: ModelInfo, is_selected: bool = False) -> None:
    """
    渲染模型卡片
    
    Args:
        model: 模型信息
        is_selected: 是否已选中
    """
    # 模型类型标签颜色
    type_colors = {
        ModelType.PYTORCH: "🔥",
        ModelType.GGUF: "⚡"
    }
    
    # 文件大小格式化
    size_mb = model.size / (1024 * 1024)
    if size_mb < 1024:
        size_str = f"{size_mb:.1f} MB"
    else:
        size_str = f"{size_mb / 1024:.1f} GB"
    
    # 选中状态指示器
    status_icon = "✅" if is_selected else "⭕"
    
    st.markdown(f"""
    **{status_icon} {model.name}**
    
    {type_colors.get(model.model_type, "📄")} {model.model_type.value.upper()} | 📁 {size_str}
    
    `{model.path}`
    """)


def render_parameter_input(
    param_name: str,
    param_type: str,
    current_value: Any,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    step: Optional[float] = None,
    options: Optional[List[str]] = None,
    help_text: Optional[str] = None
) -> Any:
    """
    渲染参数输入控件
    
    Args:
        param_name: 参数名称
        param_type: 参数类型 ('slider', 'number', 'checkbox', 'selectbox')
        current_value: 当前值
        min_value: 最小值（数值类型）
        max_value: 最大值（数值类型）
        step: 步长（数值类型）
        options: 选项列表（选择框类型）
        help_text: 帮助文本
        
    Returns:
        用户输入的新值
    """
    if param_type == 'slider':
        return st.slider(
            param_name,
            min_value=min_value or 0.0,
            max_value=max_value or 1.0,
            value=float(current_value),
            step=step or 0.1,
            help=help_text
        )
    
    elif param_type == 'number':
        return st.number_input(
            param_name,
            min_value=min_value,
            max_value=max_value,
            value=current_value,
            step=step,
            help=help_text
        )
    
    elif param_type == 'checkbox':
        return st.checkbox(
            param_name,
            value=bool(current_value),
            help=help_text
        )
    
    elif param_type == 'selectbox':
        if options and current_value in options:
            index = options.index(current_value)
        else:
            index = 0
        
        return st.selectbox(
            param_name,
            options=options or [],
            index=index,
            help=help_text
        )
    
    else:
        return st.text_input(
            param_name,
            value=str(current_value),
            help=help_text
        )


def render_progress_indicator(
    current: int,
    total: int,
    status_text: str = "",
    show_percentage: bool = True
) -> None:
    """
    渲染进度指示器
    
    Args:
        current: 当前进度
        total: 总数
        status_text: 状态文本
        show_percentage: 是否显示百分比
    """
    if total > 0:
        progress = current / total
        st.progress(progress)
        
        if show_percentage:
            percentage = int(progress * 100)
            progress_text = f"{percentage}% ({current}/{total})"
        else:
            progress_text = f"{current}/{total}"
        
        if status_text:
            st.text(f"{status_text} - {progress_text}")
        else:
            st.text(progress_text)
    else:
        st.text("准备中...")


def render_status_badge(status: str, message: str = "") -> None:
    """
    渲染状态徽章
    
    Args:
        status: 状态类型 ('success', 'error', 'warning', 'info', 'running')
        message: 状态消息
    """
    status_configs = {
        'success': ('✅', 'success', '成功'),
        'error': ('❌', 'error', '错误'),
        'warning': ('⚠️', 'warning', '警告'),
        'info': ('ℹ️', 'info', '信息'),
        'running': ('🔄', 'info', '运行中'),
        'pending': ('⏳', 'warning', '等待中'),
        'stopped': ('⏹️', 'error', '已停止')
    }
    
    icon, st_type, default_text = status_configs.get(status, ('📄', 'info', '未知'))
    display_message = message or default_text
    
    if st_type == 'success':
        st.success(f"{icon} {display_message}")
    elif st_type == 'error':
        st.error(f"{icon} {display_message}")
    elif st_type == 'warning':
        st.warning(f"{icon} {display_message}")
    else:
        st.info(f"{icon} {display_message}")


def render_model_statistics(stats: Dict[str, Any]) -> None:
    """
    渲染模型统计信息
    
    Args:
        stats: 统计信息字典
    """
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "总模型数",
            stats.get('total_models', 0),
            help="发现的模型总数"
        )
    
    with col2:
        st.metric(
            "PyTorch",
            stats.get('pytorch_models', 0),
            help="PyTorch格式模型数量"
        )
    
    with col3:
        st.metric(
            "GGUF",
            stats.get('gguf_models', 0),
            help="GGUF格式模型数量"
        )
    
    with col4:
        total_size_gb = stats.get('total_size_gb', 0)
        st.metric(
            "总大小",
            f"{total_size_gb:.1f} GB",
            help="所有模型文件的总大小"
        )


def render_error_message(error: Exception, context: str = "") -> None:
    """
    渲染错误消息
    
    Args:
        error: 异常对象
        context: 错误上下文
    """
    error_type = type(error).__name__
    error_message = str(error)
    
    if context:
        full_message = f"{context}: {error_message}"
    else:
        full_message = error_message
    
    st.error(f"❌ **{error_type}**: {full_message}")
    
    # 对于验证错误，提供更详细的信息
    if isinstance(error, ValidationError):
        with st.expander("查看详细错误信息"):
            st.code(error_message, language="text")


def render_comparison_summary(results: Dict[str, Dict[str, Any]]) -> None:
    """
    渲染比较结果摘要
    
    Args:
        results: 比较结果字典
    """
    if not results:
        st.info("暂无比较结果")
        return
    
    # 统计信息
    total_models = len(results)
    completed_models = len([r for r in results.values() if r.get('status') == 'completed'])
    error_models = len([r for r in results.values() if r.get('error')])
    running_models = len([r for r in results.values() if r.get('status') == 'running'])
    
    # 显示统计
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("总模型数", total_models)
    
    with col2:
        st.metric("已完成", completed_models, delta=completed_models - (total_models - running_models - error_models))
    
    with col3:
        st.metric("运行中", running_models)
    
    with col4:
        st.metric("错误", error_models, delta=error_models if error_models > 0 else None)
    
    # 进度条
    if total_models > 0:
        progress = completed_models / total_models
        st.progress(progress)
        st.caption(f"整体进度: {completed_models}/{total_models} ({int(progress * 100)}%)")


def render_session_management(
    results: Dict[str, Dict[str, Any]],
    prompt: str = "",
    models_info: List[Dict[str, Any]] = None
) -> None:
    """
    渲染会话管理功能
    
    Args:
        results: 比较结果
        prompt: 当前提示词
        models_info: 模型信息列表
    """
    from ..services.session_service import SessionService
    
    st.subheader("💾 会话管理")
    
    session_service = SessionService()
    
    # 会话管理选项卡
    tab1, tab2, tab3 = st.tabs(["💾 保存会话", "📂 加载会话", "🗂️ 会话管理"])
    
    with tab1:
        render_save_session_tab(session_service, results, prompt, models_info or [])
    
    with tab2:
        render_load_session_tab(session_service)
    
    with tab3:
        render_session_management_tab(session_service)


def render_save_session_tab(
    session_service,
    results: Dict[str, Dict[str, Any]],
    prompt: str,
    models_info: List[Dict[str, Any]]
) -> None:
    """渲染保存会话选项卡"""
    if not results:
        st.info("暂无比较结果可保存")
        return
    
    st.write("**保存当前比较会话**")
    
    # 会话名称输入
    session_name = st.text_input(
        "会话名称（可选）",
        placeholder="输入自定义会话名称，留空则自动生成",
        help="为会话指定一个有意义的名称，便于后续查找"
    )
    
    # 显示会话摘要
    with st.expander("📋 会话摘要", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("模型数量", len(results))
        
        with col2:
            completed = len([r for r in results.values() if r.get('status') == 'completed'])
            st.metric("成功完成", completed)
        
        with col3:
            errors = len([r for r in results.values() if r.get('error')])
            st.metric("出现错误", errors)
        
        # 提示词预览
        if prompt:
            st.text_area(
                "提示词预览",
                value=prompt[:200] + "..." if len(prompt) > 200 else prompt,
                height=100,
                disabled=True
            )
    
    # 保存按钮
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 保存会话", use_container_width=True, type="primary"):
            try:
                session_id = session_service.save_session(
                    results=results,
                    prompt=prompt,
                    models_info=models_info,
                    session_name=session_name.strip() if session_name.strip() else None
                )
                st.success(f"✅ 会话已保存！会话ID: {session_id}")
                
                # 显示保存位置
                st.info(f"📁 保存位置: .sessions/session_{session_id}.json")
                
            except Exception as e:
                st.error(f"❌ 保存失败: {str(e)}")
    
    with col2:
        if st.button("🗑️ 清空当前结果", use_container_width=True):
            st.session_state.comparison_results = {}
            st.success("✅ 当前结果已清空")
            st.rerun()


def render_load_session_tab(session_service) -> None:
    """渲染加载会话选项卡"""
    st.write("**加载历史会话**")
    
    # 获取会话列表
    sessions = session_service.list_sessions(limit=20)  # 限制显示最近20个
    
    if not sessions:
        st.info("暂无保存的会话")
        return
    
    # 会话选择
    selected_session = st.selectbox(
        "选择要加载的会话",
        options=sessions,
        format_func=lambda x: f"{x['name']} ({x['created_at'][:19] if x['created_at'] else 'Unknown'}) - {x['total_models']}个模型",
        help="选择一个历史会话进行加载"
    )
    
    if selected_session:
        # 显示会话详情
        with st.expander("📋 会话详情", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("总模型数", selected_session['total_models'])
            
            with col2:
                st.metric("成功完成", selected_session['completed_models'])
            
            with col3:
                st.metric("出现错误", selected_session['error_models'])
            
            with col4:
                file_size_mb = selected_session['file_size'] / (1024 * 1024)
                st.metric("文件大小", f"{file_size_mb:.2f} MB")
            
            # 提示词预览
            if selected_session['prompt_preview']:
                st.text_area(
                    "提示词预览",
                    value=selected_session['prompt_preview'],
                    height=80,
                    disabled=True
                )
        
        # 加载按钮
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📂 加载会话", use_container_width=True, type="primary"):
                try:
                    session_data = session_service.load_session(selected_session['id'])
                    
                    # 更新会话状态
                    st.session_state.comparison_results = session_data['comparison_results']
                    st.session_state.loaded_session_info = session_data['session_info']
                    st.session_state.loaded_models_info = session_data['models_info']
                    
                    st.success(f"✅ 会话已加载: {session_data['session_info']['name']}")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ 加载失败: {str(e)}")
        
        with col2:
            if st.button("🗑️ 删除会话", use_container_width=True):
                if session_service.delete_session(selected_session['id']):
                    st.success("✅ 会话已删除")
                    st.rerun()
                else:
                    st.error("❌ 删除失败")


def render_session_management_tab(session_service) -> None:
    """渲染会话管理选项卡"""
    st.write("**会话管理和维护**")
    
    # 获取统计信息
    stats = session_service.get_session_statistics()
    
    if stats.get('total_sessions', 0) > 0:
        # 显示统计信息
        st.subheader("📊 会话统计")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("总会话数", stats.get('total_sessions', 0))
        
        with col2:
            st.metric("总大小", f"{stats.get('total_size_mb', 0):.1f} MB")
        
        with col3:
            avg_models = stats.get('average_models_per_session', 0)
            st.metric("平均模型数", f"{avg_models:.1f}")
        
        with col4:
            error_sessions = stats.get('sessions_with_errors', 0)
            st.metric("有错误的会话", error_sessions)
        
        # 最新和最旧会话信息
        if stats.get('latest_session'):
            latest = stats['latest_session']
            st.info(f"📅 最新会话: {latest['name']} ({latest['created_at'][:19] if latest['created_at'] else 'Unknown'})")
        
        # 管理操作
        st.subheader("🛠️ 管理操作")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 备份所有会话", use_container_width=True):
                try:
                    backup_path = session_service.backup_sessions()
                    st.success(f"✅ 备份完成: {backup_path}")
                except Exception as e:
                    st.error(f"❌ 备份失败: {str(e)}")
        
        with col2:
            if st.button("🗑️ 清空所有会话", use_container_width=True):
                # 确认对话框
                if st.session_state.get('confirm_clear_all', False):
                    if session_service.clear_all_sessions(create_backup=True):
                        st.success("✅ 所有会话已清空（已创建备份）")
                        st.session_state.confirm_clear_all = False
                        st.rerun()
                    else:
                        st.error("❌ 清空失败")
                else:
                    st.session_state.confirm_clear_all = True
                    st.warning("⚠️ 再次点击确认清空所有会话")
        
        with col3:
            # 恢复备份功能
            backup_dirs = list(session_service.backup_dir.glob("*"))
            backup_dirs = [d for d in backup_dirs if d.is_dir()]
            
            if backup_dirs:
                selected_backup = st.selectbox(
                    "选择备份恢复",
                    options=backup_dirs,
                    format_func=lambda x: x.name,
                    key="backup_restore_select"
                )
                
                if st.button("📥 恢复备份", use_container_width=True):
                    if session_service.restore_sessions(str(selected_backup), overwrite=False):
                        st.success("✅ 备份已恢复")
                        st.rerun()
                    else:
                        st.error("❌ 恢复失败")
    else:
        st.info("暂无会话数据")
        
        # 即使没有会话，也提供恢复功能
        backup_dirs = list(session_service.backup_dir.glob("*"))
        backup_dirs = [d for d in backup_dirs if d.is_dir()]
        
        if backup_dirs:
            st.subheader("📥 恢复备份")
            selected_backup = st.selectbox(
                "选择备份恢复",
                options=backup_dirs,
                format_func=lambda x: x.name
            )
            
            if st.button("📥 恢复备份", use_container_width=True):
                if session_service.restore_sessions(str(selected_backup), overwrite=False):
                    st.success("✅ 备份已恢复")
                    st.rerun()
                else:
                    st.error("❌ 恢复失败")


def render_export_options(results: Dict[str, Dict[str, Any]]) -> None:
    """
    渲染导出选项
    
    Args:
        results: 比较结果
    """
    if not results:
        st.info("暂无比较结果可导出")
        return
    
    st.subheader("📤 导出结果")
    
    # 导入导出服务
    from ..services.export_service import ExportService
    export_service = ExportService()
    
    # 导出选项配置
    col1, col2 = st.columns(2)
    
    with col1:
        include_metadata = st.checkbox(
            "包含元数据",
            value=True,
            help="包含导出时间、版本等信息"
        )
    
    with col2:
        include_stats = st.checkbox(
            "包含统计信息",
            value=True,
            help="包含性能统计和时间信息"
        )
    
    # 导出按钮
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📄 导出JSON", use_container_width=True):
            try:
                content = export_service.export_comparison_results(
                    results, 'json', include_metadata, include_stats
                )
                filename = export_service.get_filename('json')
                mime_type = export_service.get_mime_type('json')
                
                st.download_button(
                    label="💾 下载JSON文件",
                    data=content,
                    file_name=filename,
                    mime=mime_type,
                    use_container_width=True
                )
                st.success("✅ JSON导出准备完成")
            except Exception as e:
                st.error(f"❌ JSON导出失败: {str(e)}")
    
    with col2:
        if st.button("📊 导出CSV", use_container_width=True):
            try:
                content = export_service.export_comparison_results(
                    results, 'csv', include_metadata, include_stats
                )
                filename = export_service.get_filename('csv')
                mime_type = export_service.get_mime_type('csv')
                
                st.download_button(
                    label="💾 下载CSV文件",
                    data=content,
                    file_name=filename,
                    mime=mime_type,
                    use_container_width=True
                )
                st.success("✅ CSV导出准备完成")
            except Exception as e:
                st.error(f"❌ CSV导出失败: {str(e)}")
    
    with col3:
        if st.button("📝 导出Markdown", use_container_width=True):
            try:
                content = export_service.export_comparison_results(
                    results, 'markdown', include_metadata, include_stats
                )
                filename = export_service.get_filename('markdown')
                mime_type = export_service.get_mime_type('markdown')
                
                st.download_button(
                    label="💾 下载MD文件",
                    data=content,
                    file_name=filename,
                    mime=mime_type,
                    use_container_width=True
                )
                st.success("✅ Markdown导出准备完成")
            except Exception as e:
                st.error(f"❌ Markdown导出失败: {str(e)}")
    
    with col4:
        if st.button("📄 导出TXT", use_container_width=True):
            try:
                content = export_service.export_comparison_results(
                    results, 'txt', include_metadata, include_stats
                )
                filename = export_service.get_filename('txt')
                mime_type = export_service.get_mime_type('txt')
                
                st.download_button(
                    label="💾 下载TXT文件",
                    data=content,
                    file_name=filename,
                    mime=mime_type,
                    use_container_width=True
                )
                st.success("✅ TXT导出准备完成")
            except Exception as e:
                st.error(f"❌ TXT导出失败: {str(e)}")
    
    # 预览功能
    st.markdown("---")
    st.subheader("👀 导出预览")
    
    preview_format = st.selectbox(
        "选择预览格式",
        options=['json', 'csv', 'markdown', 'txt'],
        format_func=lambda x: {
            'json': '📄 JSON格式',
            'csv': '📊 CSV格式', 
            'markdown': '📝 Markdown格式',
            'txt': '📄 纯文本格式'
        }[x]
    )
    
    if st.button("🔍 生成预览", use_container_width=True):
        try:
            preview_content = export_service.export_comparison_results(
                results, preview_format, include_metadata, include_stats
            )
            
            st.subheader(f"📋 {preview_format.upper()}格式预览")
            
            # 根据格式选择合适的显示方式
            if preview_format == 'json':
                st.code(preview_content, language='json')
            elif preview_format == 'csv':
                st.code(preview_content, language='csv')
            elif preview_format == 'markdown':
                with st.expander("📖 渲染视图", expanded=True):
                    st.markdown(preview_content)
                with st.expander("📝 原始文本"):
                    st.code(preview_content, language='markdown')
            else:  # txt
                st.text_area(
                    "预览内容",
                    value=preview_content,
                    height=400,
                    disabled=True
                )
                
        except Exception as e:
            st.error(f"❌ 预览生成失败: {str(e)}")


def render_model_output_card(
    model_name: str,
    model_type: str,
    status: str,
    content: str = "",
    error: Optional[str] = None,
    stats: Optional[Dict[str, Any]] = None,
    show_raw: bool = False
) -> None:
    """
    渲染模型输出卡片，支持增强的Markdown渲染
    
    Args:
        model_name: 模型名称
        model_type: 模型类型
        status: 状态
        content: 输出内容
        error: 错误信息
        stats: 统计信息
        show_raw: 是否显示原始文本
    """
    # 模型类型图标
    type_icons = {
        'pytorch': '🔥',
        'gguf': '⚡'
    }
    
    # 状态指示器
    status_configs = {
        'pending': ('⏳', 'orange', '等待中'),
        'running': ('🔄', 'blue', '运行中'),
        'completed': ('✅', 'green', '已完成'),
        'error': ('❌', 'red', '错误')
    }
    
    status_icon, status_color, status_text = status_configs.get(status, ('❓', 'gray', '未知'))
    type_icon = type_icons.get(model_type.lower(), '📄')
    
    # 渲染卡片头部
    st.markdown(f"""
    <div style="border: 2px solid {status_color}; border-radius: 10px; padding: 15px; margin-bottom: 15px; background-color: rgba(255,255,255,0.05);">
        <h4 style="margin: 0; color: {status_color};">
            {status_icon} {type_icon} {model_name}
        </h4>
        <p style="margin: 5px 0 0 0; font-size: 0.8em; opacity: 0.8;">
            {model_type.upper()} | 状态: {status_text}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 进度指示器
    if status == 'running':
        st.progress(0.5)
        st.text("🔄 正在生成回答...")
    
    # 内容显示
    if error:
        st.error(f"❌ 错误: {error}")
    elif content:
        # 使用增强的内容查看器
        container_key = f"output_{model_name}_{hash(content[:50])}"
        render_content_viewer(
            content=content,
            title="模型输出",
            show_raw_toggle=True,
            show_search=True,
            max_height=400,
            container_key=container_key
        )
    elif status == 'pending':
        st.info("⏳ 等待开始...")
    elif status == 'running':
        st.info("🔄 正在生成中...")
    
    # 统计信息
    if status == 'completed' and stats:
        with st.expander("📈 性能统计", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                duration = stats.get('duration', 0)
                try:
                    duration_float = float(duration) if duration is not None else 0.0
                    st.metric("用时", f"{duration_float:.1f}s")
                except (ValueError, TypeError):
                    st.metric("用时", "N/A")
            
            with col2:
                token_count = stats.get('token_count', 0)
                try:
                    token_count_int = int(token_count) if token_count is not None else 0
                    st.metric("Token数", token_count_int)
                except (ValueError, TypeError):
                    st.metric("Token数", "N/A")
            
            with col3:
                try:
                    duration_float = float(duration) if duration is not None else 0.0
                    token_count_int = int(token_count) if token_count is not None else 0
                    if duration_float > 0:
                        tps = token_count_int / duration_float
                        st.metric("速度", f"{tps:.1f} t/s")
                    else:
                        st.metric("速度", "N/A")
                except (ValueError, TypeError, ZeroDivisionError):
                    st.metric("速度", "N/A")


def render_streaming_output(
    container_key: str,
    content: str,
    is_complete: bool = False
) -> None:
    """
    渲染流式输出
    
    Args:
        container_key: 容器键
        content: 内容
        is_complete: 是否完成
    """
    # 使用st.empty()创建可更新的容器
    if container_key not in st.session_state:
        st.session_state[container_key] = st.empty()
    
    container = st.session_state[container_key]
    
    # 更新内容
    with container.container():
        if content:
            st.markdown(content)
            if not is_complete:
                st.markdown("▋")  # 光标指示器
        else:
            st.text("等待输出...")


def render_responsive_columns(
    items: List[Any],
    render_func: callable,
    max_cols: int = 4
) -> None:
    """
    渲染响应式列布局
    
    Args:
        items: 要渲染的项目列表
        render_func: 渲染函数
        max_cols: 最大列数
    """
    if not items:
        return
    
    num_items = len(items)
    
    # 根据项目数量决定列数
    if num_items == 1:
        cols = st.columns(1)
        for i, item in enumerate(items):
            with cols[i]:
                render_func(item, i)
    elif num_items == 2:
        cols = st.columns(2)
        for i, item in enumerate(items):
            with cols[i]:
                render_func(item, i)
    elif num_items == 3:
        cols = st.columns(3)
        for i, item in enumerate(items):
            with cols[i]:
                render_func(item, i)
    elif num_items == 4:
        # 2x2 布局
        # 第一行
        cols_row1 = st.columns(2)
        for i in range(2):
            with cols_row1[i]:
                render_func(items[i], i)
        
        # 第二行
        cols_row2 = st.columns(2)
        for i in range(2, 4):
            with cols_row2[i - 2]:
                render_func(items[i], i)
    else:
        # 超过4个时使用滚动布局
        cols = st.columns(min(num_items, max_cols))
        for i, item in enumerate(items):
            col_index = i % len(cols)
            with cols[col_index]:
                render_func(item, i)


def render_markdown_content(
    content: str,
    enable_code_copy: bool = True,
    enable_math: bool = True,
    enable_tables: bool = True,
    container_key: Optional[str] = None
) -> None:
    """
    渲染Markdown内容，支持代码高亮、数学公式和表格
    
    Args:
        content: Markdown内容
        enable_code_copy: 是否启用代码复制功能
        enable_math: 是否启用数学公式渲染
        enable_tables: 是否启用表格格式化
        container_key: 容器键，用于唯一标识
    """
    if not content:
        st.info("暂无内容")
        return
    
    # 处理代码块
    if enable_code_copy:
        content = _enhance_code_blocks(content, container_key)
    
    # 处理数学公式
    if enable_math:
        content = _enhance_math_formulas(content)
    
    # 处理表格
    if enable_tables:
        content = _enhance_tables(content)
    
    # 渲染内容
    st.markdown(content, unsafe_allow_html=True)


def _enhance_code_blocks(content: str, container_key: Optional[str] = None) -> str:
    """
    增强代码块，添加复制功能和语法高亮
    
    Args:
        content: 原始内容
        container_key: 容器键
        
    Returns:
        增强后的内容
    """
    # 匹配代码块模式
    code_block_pattern = r'```(\w+)?\n(.*?)\n```'
    
    def replace_code_block(match):
        language = match.group(1) or 'text'
        code = match.group(2)
        
        # 生成唯一的代码块ID
        import hashlib
        code_id = hashlib.md5(code.encode()).hexdigest()[:8]
        if container_key:
            code_id = f"{container_key}_{code_id}"
        
        # 创建带复制按钮的代码块
        return f"""
<div style="position: relative; margin: 10px 0;">
    <div style="background-color: #f8f9fa; border: 1px solid #e9ecef; border-radius: 6px; padding: 16px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-size: 12px; color: #6c757d; font-weight: bold;">{language.upper()}</span>
            <button onclick="copyCode('{code_id}')" style="background: #007bff; color: white; border: none; border-radius: 4px; padding: 4px 8px; font-size: 12px; cursor: pointer;">
                📋 复制
            </button>
        </div>
        <pre id="{code_id}" style="margin: 0; background: transparent; border: none; padding: 0; overflow-x: auto;"><code class="language-{language}">{code}</code></pre>
    </div>
</div>

<script>
function copyCode(codeId) {{
    const codeElement = document.getElementById(codeId);
    const text = codeElement.textContent;
    navigator.clipboard.writeText(text).then(function() {{
        // 临时改变按钮文本
        const button = event.target;
        const originalText = button.textContent;
        button.textContent = '✅ 已复制';
        setTimeout(() => {{
            button.textContent = originalText;
        }}, 2000);
    }});
}}
</script>
"""
    
    # 替换所有代码块
    enhanced_content = re.sub(code_block_pattern, replace_code_block, content, flags=re.DOTALL)
    
    return enhanced_content


def _enhance_math_formulas(content: str) -> str:
    """
    增强数学公式渲染
    
    Args:
        content: 原始内容
        
    Returns:
        增强后的内容
    """
    # 先处理块级数学公式 $$...$$ (必须在行内公式之前处理)
    block_math_pattern = r'\$\$([^$]+?)\$\$'
    content = re.sub(block_math_pattern, r'\\[\1\\]', content, flags=re.DOTALL)
    
    # 处理行内数学公式 $...$
    inline_math_pattern = r'\$([^$]+?)\$'
    content = re.sub(inline_math_pattern, r'\\(\1\\)', content)
    
    return content


def _enhance_tables(content: str) -> str:
    """
    增强表格格式化
    
    Args:
        content: 原始内容
        
    Returns:
        增强后的内容
    """
    # Streamlit原生支持Markdown表格，这里可以添加额外的样式
    # 检测表格并添加样式
    lines = content.split('\n')
    enhanced_lines = []
    in_table = False
    
    for line in lines:
        # 检测表格行（包含 | 分隔符）
        if '|' in line and line.strip():
            if not in_table:
                # 表格开始，添加样式标记
                enhanced_lines.append('<div style="overflow-x: auto; margin: 10px 0;">')
                in_table = True
            enhanced_lines.append(line)
        else:
            if in_table:
                # 表格结束
                enhanced_lines.append('</div>')
                in_table = False
            enhanced_lines.append(line)
    
    # 如果内容以表格结束，确保关闭标签
    if in_table:
        enhanced_lines.append('</div>')
    
    return '\n'.join(enhanced_lines)


def render_content_viewer(
    content: str,
    title: str = "内容查看器",
    show_raw_toggle: bool = True,
    show_search: bool = True,
    max_height: int = 400,
    container_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    渲染内容查看器，支持渲染视图和原始文本切换
    
    Args:
        content: 内容
        title: 标题
        show_raw_toggle: 是否显示原始文本切换
        show_search: 是否显示搜索功能
        max_height: 最大高度
        container_key: 容器键
        
    Returns:
        查看器状态字典
    """
    if not content:
        st.info("暂无内容")
        return {"view_mode": "rendered", "search_term": ""}
    
    # 生成唯一键
    unique_key = container_key or f"viewer_{hash(content[:100])}"
    
    # 控制面板
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.subheader(title)
    
    view_mode = "rendered"
    search_term = ""
    
    if show_raw_toggle:
        with col2:
            view_mode = st.selectbox(
                "查看模式",
                options=["rendered", "raw"],
                format_func=lambda x: "📖 渲染视图" if x == "rendered" else "📝 原始文本",
                key=f"{unique_key}_view_mode"
            )
    
    if show_search:
        with col3:
            search_term = st.text_input(
                "搜索内容",
                placeholder="输入搜索词...",
                key=f"{unique_key}_search"
            )
    
    # 内容显示区域
    content_container = st.container()
    
    with content_container:
        if view_mode == "rendered":
            # 渲染视图
            if search_term:
                highlighted_content = _highlight_search_term(content, search_term)
                render_markdown_content(
                    highlighted_content,
                    container_key=f"{unique_key}_rendered"
                )
            else:
                render_markdown_content(
                    content,
                    container_key=f"{unique_key}_rendered"
                )
        else:
            # 原始文本视图
            display_content = content
            if search_term:
                display_content = _highlight_search_term(content, search_term, is_raw=True)
            
            st.text_area(
                "原始内容",
                value=display_content,
                height=max_height,
                disabled=True,
                key=f"{unique_key}_raw_text"
            )
    
    # 内容统计
    with st.expander("📊 内容统计", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            char_count = len(content)
            st.metric("字符数", f"{char_count:,}")
        
        with col2:
            word_count = len(content.split())
            st.metric("单词数", f"{word_count:,}")
        
        with col3:
            line_count = len(content.split('\n'))
            st.metric("行数", f"{line_count:,}")
        
        with col4:
            if search_term:
                match_count = content.lower().count(search_term.lower())
                st.metric("搜索匹配", f"{match_count}")
            else:
                st.metric("段落数", f"{len([p for p in content.split('\n\n') if p.strip()])}")
    
    return {
        "view_mode": view_mode,
        "search_term": search_term,
        "char_count": len(content),
        "word_count": len(content.split()),
        "line_count": len(content.split('\n'))
    }


def _highlight_search_term(content: str, search_term: str, is_raw: bool = False) -> str:
    """
    高亮搜索词
    
    Args:
        content: 内容
        search_term: 搜索词
        is_raw: 是否为原始文本模式
        
    Returns:
        高亮后的内容
    """
    if not search_term:
        return content
    
    # 转义特殊字符
    escaped_term = re.escape(search_term)
    
    if is_raw:
        # 原始文本模式，使用简单的标记
        pattern = re.compile(escaped_term, re.IGNORECASE)
        return pattern.sub(f">>>{search_term}<<<", content)
    else:
        # 渲染模式，使用HTML高亮
        pattern = re.compile(escaped_term, re.IGNORECASE)
        return pattern.sub(
            f'<mark style="background-color: yellow; padding: 2px 4px; border-radius: 3px;">{search_term}</mark>',
            content
        )


def render_collapsible_content(
    content: str,
    title: str,
    max_preview_length: int = 200,
    expanded: bool = False,
    container_key: Optional[str] = None
) -> None:
    """
    渲染可折叠的内容区域
    
    Args:
        content: 内容
        title: 标题
        max_preview_length: 预览最大长度
        expanded: 是否默认展开
        container_key: 容器键
    """
    if not content:
        st.info(f"{title}: 暂无内容")
        return
    
    # 生成预览文本
    preview = content[:max_preview_length]
    if len(content) > max_preview_length:
        preview += "..."
    
    # 显示预览和展开按钮
    if not expanded and len(content) > max_preview_length:
        st.markdown(f"**{title}**")
        st.markdown(preview)
        
        if st.button(f"📖 展开完整内容", key=f"{container_key}_expand" if container_key else None):
            st.session_state[f"{container_key}_expanded"] = True
            st.rerun()
    else:
        # 显示完整内容
        with st.expander(title, expanded=True):
            render_markdown_content(content, container_key=container_key)
            
            if len(content) > max_preview_length:
                if st.button(f"📄 折叠内容", key=f"{container_key}_collapse" if container_key else None):
                    st.session_state[f"{container_key}_expanded"] = False
                    st.rerun()


def render_advanced_text_viewer(
    content: str,
    title: str = "文本查看器",
    container_key: Optional[str] = None,
    enable_line_numbers: bool = True,
    enable_word_wrap: bool = True,
    enable_syntax_highlighting: bool = False,
    language: str = "text",
    max_height: int = 500
) -> Dict[str, Any]:
    """
    渲染高级文本查看器，支持多种查看选项
    
    Args:
        content: 文本内容
        title: 标题
        container_key: 容器键
        enable_line_numbers: 是否显示行号
        enable_word_wrap: 是否启用自动换行
        enable_syntax_highlighting: 是否启用语法高亮
        language: 语法高亮语言
        max_height: 最大高度
        
    Returns:
        查看器状态字典
    """
    if not content:
        st.info("暂无内容")
        return {"view_mode": "plain", "options": {}}
    
    # 生成唯一键
    unique_key = container_key or f"advanced_viewer_{hash(content[:100])}"
    
    # 控制面板
    st.subheader(title)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        view_mode = st.selectbox(
            "查看模式",
            options=["plain", "formatted", "code"],
            format_func=lambda x: {
                "plain": "📝 纯文本",
                "formatted": "📖 格式化",
                "code": "💻 代码"
            }[x],
            key=f"{unique_key}_view_mode"
        )
    
    with col2:
        show_line_numbers = st.checkbox(
            "显示行号",
            value=enable_line_numbers,
            key=f"{unique_key}_line_numbers"
        )
    
    with col3:
        word_wrap = st.checkbox(
            "自动换行",
            value=enable_word_wrap,
            key=f"{unique_key}_word_wrap"
        )
    
    with col4:
        search_term = st.text_input(
            "搜索",
            placeholder="输入搜索词...",
            key=f"{unique_key}_search"
        )
    
    # 内容处理
    display_content = content
    if search_term:
        display_content = _highlight_search_term(content, search_term, is_raw=(view_mode == "plain"))
    
    # 内容显示
    content_container = st.container()
    
    with content_container:
        if view_mode == "plain":
            # 纯文本模式
            if show_line_numbers:
                lines = display_content.split('\n')
                numbered_lines = []
                for i, line in enumerate(lines, 1):
                    numbered_lines.append(f"{i:4d} | {line}")
                display_content = '\n'.join(numbered_lines)
            
            st.text_area(
                "内容",
                value=display_content,
                height=max_height,
                disabled=True,
                key=f"{unique_key}_plain_text"
            )
        
        elif view_mode == "formatted":
            # 格式化模式（Markdown渲染）
            if show_line_numbers:
                # 为格式化内容添加行号（简化版）
                lines = display_content.split('\n')
                numbered_content = ""
                for i, line in enumerate(lines, 1):
                    numbered_content += f"`{i:4d}` {line}\n"
                display_content = numbered_content
            
            render_markdown_content(
                display_content,
                container_key=f"{unique_key}_formatted"
            )
        
        elif view_mode == "code":
            # 代码模式
            if enable_syntax_highlighting:
                st.code(display_content, language=language)
            else:
                st.code(display_content, language="text")
    
    # 内容操作工具栏
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📋 复制全部", key=f"{unique_key}_copy_all"):
            # 这里可以添加复制到剪贴板的JavaScript代码
            st.success("内容已复制到剪贴板")
    
    with col2:
        if st.button("💾 下载文本", key=f"{unique_key}_download"):
            st.download_button(
                label="下载为TXT",
                data=content,
                file_name=f"{title.replace(' ', '_')}.txt",
                mime="text/plain",
                key=f"{unique_key}_download_btn"
            )
    
    with col3:
        if st.button("🔍 查找替换", key=f"{unique_key}_find_replace"):
            st.session_state[f"{unique_key}_show_find_replace"] = True
    
    with col4:
        if st.button("📊 文本分析", key=f"{unique_key}_analyze"):
            st.session_state[f"{unique_key}_show_analysis"] = True
    
    # 查找替换功能
    if st.session_state.get(f"{unique_key}_show_find_replace", False):
        with st.expander("🔍 查找替换", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                find_text = st.text_input(
                    "查找",
                    key=f"{unique_key}_find_text"
                )
            
            with col2:
                replace_text = st.text_input(
                    "替换为",
                    key=f"{unique_key}_replace_text"
                )
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                case_sensitive = st.checkbox(
                    "区分大小写",
                    key=f"{unique_key}_case_sensitive"
                )
            
            with col2:
                if st.button("查找所有", key=f"{unique_key}_find_all"):
                    if find_text:
                        flags = 0 if case_sensitive else re.IGNORECASE
                        matches = re.findall(re.escape(find_text), content, flags)
                        st.info(f"找到 {len(matches)} 个匹配项")
            
            with col3:
                if st.button("关闭", key=f"{unique_key}_close_find_replace"):
                    st.session_state[f"{unique_key}_show_find_replace"] = False
                    st.rerun()
    
    # 文本分析
    if st.session_state.get(f"{unique_key}_show_analysis", False):
        with st.expander("📊 文本分析", expanded=True):
            _render_text_analysis(content, unique_key)
            
            if st.button("关闭分析", key=f"{unique_key}_close_analysis"):
                st.session_state[f"{unique_key}_show_analysis"] = False
                st.rerun()
    
    return {
        "view_mode": view_mode,
        "show_line_numbers": show_line_numbers,
        "word_wrap": word_wrap,
        "search_term": search_term,
        "char_count": len(content),
        "word_count": len(content.split()),
        "line_count": len(content.split('\n'))
    }


def _render_text_analysis(content: str, unique_key: str) -> None:
    """
    渲染文本分析结果
    
    Args:
        content: 文本内容
        unique_key: 唯一键
    """
    # 基础统计
    char_count = len(content)
    word_count = len(content.split())
    line_count = len(content.split('\n'))
    paragraph_count = len([p for p in content.split('\n\n') if p.strip()])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("字符数", f"{char_count:,}")
    
    with col2:
        st.metric("单词数", f"{word_count:,}")
    
    with col3:
        st.metric("行数", f"{line_count:,}")
    
    with col4:
        st.metric("段落数", f"{paragraph_count:,}")
    
    # 字符分布分析
    if char_count > 0:
        st.subheader("字符分布")
        
        # 统计不同类型字符
        alpha_count = sum(1 for c in content if c.isalpha())
        digit_count = sum(1 for c in content if c.isdigit())
        space_count = sum(1 for c in content if c.isspace())
        punct_count = sum(1 for c in content if not c.isalnum() and not c.isspace())
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("字母", f"{alpha_count:,}", f"{alpha_count/char_count*100:.1f}%")
        
        with col2:
            st.metric("数字", f"{digit_count:,}", f"{digit_count/char_count*100:.1f}%")
        
        with col3:
            st.metric("空白", f"{space_count:,}", f"{space_count/char_count*100:.1f}%")
        
        with col4:
            st.metric("标点", f"{punct_count:,}", f"{punct_count/char_count*100:.1f}%")
    
    # 词频分析（简化版）
    if word_count > 0:
        st.subheader("词频分析（前10个）")
        
        words = content.lower().split()
        # 过滤掉常见的停用词
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        filtered_words = [word.strip('.,!?;:"()[]{}') for word in words if word.lower() not in stop_words and len(word) > 2]
        
        from collections import Counter
        word_freq = Counter(filtered_words)
        top_words = word_freq.most_common(10)
        
        if top_words:
            for i, (word, count) in enumerate(top_words, 1):
                st.text(f"{i:2d}. {word}: {count} 次")
    
    # 行长度分析
    if line_count > 1:
        st.subheader("行长度分析")
        
        lines = content.split('\n')
        line_lengths = [len(line) for line in lines]
        
        if line_lengths:
            avg_length = sum(line_lengths) / len(line_lengths)
            max_length = max(line_lengths)
            min_length = min(line_lengths)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("平均长度", f"{avg_length:.1f}")
            
            with col2:
                st.metric("最长行", f"{max_length}")
            
            with col3:
                st.metric("最短行", f"{min_length}")


def render_text_comparison_viewer(
    contents: Dict[str, str],
    title: str = "文本比较查看器",
    container_key: Optional[str] = None
) -> None:
    """
    渲染文本比较查看器，支持多个文本的并排比较
    
    Args:
        contents: 文本内容字典 {标签: 内容}
        title: 标题
        container_key: 容器键
    """
    if not contents:
        st.info("暂无内容进行比较")
        return
    
    st.subheader(title)
    
    # 生成唯一键
    unique_key = container_key or f"comparison_viewer_{hash(str(contents.keys()))}"
    
    # 控制选项
    col1, col2, col3 = st.columns(3)
    
    with col1:
        view_mode = st.selectbox(
            "查看模式",
            options=["side_by_side", "stacked", "diff"],
            format_func=lambda x: {
                "side_by_side": "📊 并排显示",
                "stacked": "📚 堆叠显示", 
                "diff": "🔍 差异对比"
            }[x],
            key=f"{unique_key}_comparison_mode"
        )
    
    with col2:
        show_stats = st.checkbox(
            "显示统计",
            value=True,
            key=f"{unique_key}_show_stats"
        )
    
    with col3:
        sync_scroll = st.checkbox(
            "同步滚动",
            value=True,
            key=f"{unique_key}_sync_scroll"
        )
    
    # 内容显示
    if view_mode == "side_by_side":
        # 并排显示
        cols = st.columns(len(contents))
        
        for i, (label, content) in enumerate(contents.items()):
            with cols[i]:
                st.markdown(f"**{label}**")
                
                if show_stats:
                    char_count = len(content)
                    word_count = len(content.split())
                    st.caption(f"字符: {char_count:,} | 单词: {word_count:,}")
                
                st.text_area(
                    f"内容 - {label}",
                    value=content,
                    height=400,
                    disabled=True,
                    key=f"{unique_key}_content_{i}",
                    label_visibility="collapsed"
                )
    
    elif view_mode == "stacked":
        # 堆叠显示
        for label, content in contents.items():
            st.markdown(f"### {label}")
            
            if show_stats:
                char_count = len(content)
                word_count = len(content.split())
                st.caption(f"字符: {char_count:,} | 单词: {word_count:,}")
            
            st.text_area(
                f"内容 - {label}",
                value=content,
                height=300,
                disabled=True,
                key=f"{unique_key}_stacked_{hash(label)}",
                label_visibility="collapsed"
            )
            
            st.markdown("---")
    
    elif view_mode == "diff":
        # 差异对比（简化版）
        if len(contents) == 2:
            labels = list(contents.keys())
            content1, content2 = list(contents.values())
            
            st.markdown(f"**比较: {labels[0]} vs {labels[1]}**")
            
            # 简单的行级差异比较
            lines1 = content1.split('\n')
            lines2 = content2.split('\n')
            
            max_lines = max(len(lines1), len(lines2))
            
            diff_result = []
            for i in range(max_lines):
                line1 = lines1[i] if i < len(lines1) else ""
                line2 = lines2[i] if i < len(lines2) else ""
                
                if line1 == line2:
                    diff_result.append(f"  {i+1:4d} | {line1}")
                else:
                    if line1:
                        diff_result.append(f"- {i+1:4d} | {line1}")
                    if line2:
                        diff_result.append(f"+ {i+1:4d} | {line2}")
            
            st.code('\n'.join(diff_result), language="diff")
        else:
            st.warning("差异对比模式需要恰好两个文本内容")
    
    # 比较统计
    if show_stats and len(contents) > 1:
        st.subheader("📊 比较统计")
        
        stats_data = []
        for label, content in contents.items():
            stats_data.append({
                "标签": label,
                "字符数": len(content),
                "单词数": len(content.split()),
                "行数": len(content.split('\n')),
                "段落数": len([p for p in content.split('\n\n') if p.strip()])
            })
        
        # 显示统计表格
        import pandas as pd
        df = pd.DataFrame(stats_data)
        st.dataframe(df, use_container_width=True)


def render_content_formatting_options(
    content: str,
    container_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    渲染内容格式化选项控制面板
    
    Args:
        content: 内容
        container_key: 容器键
        
    Returns:
        格式化选项字典
    """
    if not content:
        return {}
    
    unique_key = container_key or f"format_options_{hash(content[:50])}"
    
    st.subheader("🎨 格式化选项")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # 文本处理选项
        st.markdown("**文本处理**")
        
        remove_extra_spaces = st.checkbox(
            "移除多余空格",
            key=f"{unique_key}_remove_spaces"
        )
        
        normalize_line_endings = st.checkbox(
            "标准化换行符",
            key=f"{unique_key}_normalize_lines"
        )
        
        trim_lines = st.checkbox(
            "去除行首尾空白",
            key=f"{unique_key}_trim_lines"
        )
    
    with col2:
        # 显示选项
        st.markdown("**显示选项**")
        
        show_whitespace = st.checkbox(
            "显示空白字符",
            key=f"{unique_key}_show_whitespace"
        )
        
        highlight_long_lines = st.checkbox(
            "高亮长行",
            key=f"{unique_key}_highlight_long"
        )
        
        max_line_length = st.number_input(
            "最大行长度",
            min_value=50,
            max_value=200,
            value=80,
            key=f"{unique_key}_max_line_length"
        )
    
    with col3:
        # 导出选项
        st.markdown("**导出选项**")
        
        export_format = st.selectbox(
            "导出格式",
            options=["txt", "md", "html", "json"],
            key=f"{unique_key}_export_format"
        )
        
        include_metadata = st.checkbox(
            "包含元数据",
            key=f"{unique_key}_include_metadata"
        )
        
        if st.button("📥 导出", key=f"{unique_key}_export_btn"):
            formatted_content = _apply_formatting_options(
                content,
                remove_extra_spaces=remove_extra_spaces,
                normalize_line_endings=normalize_line_endings,
                trim_lines=trim_lines,
                show_whitespace=show_whitespace,
                highlight_long_lines=highlight_long_lines,
                max_line_length=max_line_length
            )
            
            if export_format == "txt":
                st.download_button(
                    "下载TXT文件",
                    data=formatted_content,
                    file_name="formatted_content.txt",
                    mime="text/plain"
                )
            elif export_format == "md":
                st.download_button(
                    "下载Markdown文件",
                    data=formatted_content,
                    file_name="formatted_content.md",
                    mime="text/markdown"
                )
            elif export_format == "html":
                html_content = f"<html><body><pre>{formatted_content}</pre></body></html>"
                st.download_button(
                    "下载HTML文件",
                    data=html_content,
                    file_name="formatted_content.html",
                    mime="text/html"
                )
            elif export_format == "json":
                import json
                json_data = {
                    "content": formatted_content,
                    "metadata": {
                        "char_count": len(formatted_content),
                        "word_count": len(formatted_content.split()),
                        "line_count": len(formatted_content.split('\n'))
                    } if include_metadata else {}
                }
                st.download_button(
                    "下载JSON文件",
                    data=json.dumps(json_data, ensure_ascii=False, indent=2),
                    file_name="formatted_content.json",
                    mime="application/json"
                )
    
    return {
        "remove_extra_spaces": remove_extra_spaces,
        "normalize_line_endings": normalize_line_endings,
        "trim_lines": trim_lines,
        "show_whitespace": show_whitespace,
        "highlight_long_lines": highlight_long_lines,
        "max_line_length": max_line_length,
        "export_format": export_format,
        "include_metadata": include_metadata
    }


def _apply_formatting_options(
    content: str,
    remove_extra_spaces: bool = False,
    normalize_line_endings: bool = False,
    trim_lines: bool = False,
    show_whitespace: bool = False,
    highlight_long_lines: bool = False,
    max_line_length: int = 80
) -> str:
    """
    应用格式化选项到内容
    
    Args:
        content: 原始内容
        remove_extra_spaces: 是否移除多余空格
        normalize_line_endings: 是否标准化换行符
        trim_lines: 是否去除行首尾空白
        show_whitespace: 是否显示空白字符
        highlight_long_lines: 是否高亮长行
        max_line_length: 最大行长度
        
    Returns:
        格式化后的内容
    """
    result = content
    
    # 标准化换行符
    if normalize_line_endings:
        result = result.replace('\r\n', '\n').replace('\r', '\n')
    
    # 处理行
    lines = result.split('\n')
    processed_lines = []
    
    for line in lines:
        processed_line = line
        
        # 去除行首尾空白
        if trim_lines:
            processed_line = processed_line.strip()
        
        # 移除多余空格
        if remove_extra_spaces:
            processed_line = re.sub(r' +', ' ', processed_line)
        
        # 显示空白字符
        if show_whitespace:
            processed_line = processed_line.replace(' ', '·').replace('\t', '→')
        
        # 高亮长行
        if highlight_long_lines and len(processed_line) > max_line_length:
            processed_line = f"⚠️ {processed_line} ⚠️"
        
        processed_lines.append(processed_line)
    
    return '\n'.join(processed_lines)


def render_content_search_and_replace(
    content: str,
    container_key: Optional[str] = None
) -> str:
    """
    渲染内容搜索和替换功能
    
    Args:
        content: 原始内容
        container_key: 容器键
        
    Returns:
        处理后的内容
    """
    if not content:
        return content
    
    unique_key = container_key or f"search_replace_{hash(content[:50])}"
    
    st.subheader("🔍 搜索和替换")
    
    col1, col2 = st.columns(2)
    
    with col1:
        search_text = st.text_input(
            "搜索文本",
            key=f"{unique_key}_search_text"
        )
    
    with col2:
        replace_text = st.text_input(
            "替换为",
            key=f"{unique_key}_replace_text"
        )
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        case_sensitive = st.checkbox(
            "区分大小写",
            key=f"{unique_key}_case_sensitive"
        )
    
    with col2:
        use_regex = st.checkbox(
            "使用正则表达式",
            key=f"{unique_key}_use_regex"
        )
    
    with col3:
        whole_word = st.checkbox(
            "全词匹配",
            key=f"{unique_key}_whole_word"
        )
    
    with col4:
        preview_only = st.checkbox(
            "仅预览",
            value=True,
            key=f"{unique_key}_preview_only"
        )
    
    # 执行搜索和替换
    if search_text:
        try:
            if use_regex:
                flags = 0 if case_sensitive else re.IGNORECASE
                if whole_word:
                    pattern = rf'\b{re.escape(search_text)}\b'
                else:
                    pattern = search_text
                
                matches = re.findall(pattern, content, flags)
                match_count = len(matches)
                
                if not preview_only and replace_text is not None:
                    result_content = re.sub(pattern, replace_text, content, flags=flags)
                else:
                    result_content = content
            else:
                if case_sensitive:
                    search_func = content.count
                    replace_func = content.replace
                else:
                    search_func = content.lower().count
                    replace_func = lambda old, new: content.replace(old, new) if case_sensitive else content.lower().replace(old.lower(), new)
                
                if whole_word:
                    # 简化的全词匹配
                    words = content.split()
                    match_count = sum(1 for word in words if (word == search_text if case_sensitive else word.lower() == search_text.lower()))
                else:
                    match_count = search_func(search_text.lower() if not case_sensitive else search_text)
                
                if not preview_only and replace_text is not None:
                    if whole_word:
                        # 简化的全词替换
                        words = content.split()
                        new_words = []
                        for word in words:
                            if (word == search_text if case_sensitive else word.lower() == search_text.lower()):
                                new_words.append(replace_text)
                            else:
                                new_words.append(word)
                        result_content = ' '.join(new_words)
                    else:
                        result_content = content.replace(search_text, replace_text)
                else:
                    result_content = content
            
            # 显示搜索结果
            if match_count > 0:
                st.success(f"找到 {match_count} 个匹配项")
                
                if not preview_only and replace_text is not None:
                    st.info("替换已应用")
                    return result_content
                else:
                    # 显示预览
                    if replace_text is not None:
                        st.info("预览模式 - 替换未应用")
                        with st.expander("预览替换结果"):
                            preview_content = content.replace(search_text, f"**{replace_text}**") if not use_regex else content
                            st.markdown(preview_content)
            else:
                st.warning("未找到匹配项")
        
        except re.error as e:
            st.error(f"正则表达式错误: {e}")
        except Exception as e:
            st.error(f"搜索替换错误: {e}")
    
    return content