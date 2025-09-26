"""
Streamlit应用入口点

多LLM模型比较器的主应用程序。
"""

import streamlit as st
import os
import logging
import time
from pathlib import Path
from typing import Dict, Any, List

import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from multi_llm_comparator.services.model_manager import ModelManager
from multi_llm_comparator.services.inference_engine import InferenceEngine
from multi_llm_comparator.core.config import ConfigManager
from multi_llm_comparator.core.models import ModelType, ModelConfig
from multi_llm_comparator.core.exceptions import ModelSelectionError, ModelNotFoundError, ValidationError
from multi_llm_comparator.services.error_handler import handle_error, setup_error_handling
from multi_llm_comparator.services.recovery_manager import get_recovery_manager
from multi_llm_comparator.ui.enhancements import (
    initialize_ui_enhancements, show_enhanced_progress, show_operation_confirmation,
    render_undo_redo_controls, save_operation_state, render_loading_overlay,
    show_toast_notification, NotificationType, keyboard_manager, render_status_indicator,
    create_interactive_tutorial, render_advanced_text_viewer
)


# 配置日志 - 设置为WARNING级别以减少冗余输出
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def initialize_session_state():
    """初始化Streamlit会话状态"""
    # 设置错误处理
    if 'error_handling_setup' not in st.session_state:
        setup_error_handling("logs/streamlit_error.log")
        st.session_state.error_handling_setup = True
    
    # 初始化UI增强功能
    if 'ui_enhancements_initialized' not in st.session_state:
        initialize_ui_enhancements()
        st.session_state.ui_enhancements_initialized = True
    
    if 'model_manager' not in st.session_state:
        st.session_state.model_manager = ModelManager()
    
    if 'config_manager' not in st.session_state:
        st.session_state.config_manager = ConfigManager()
    
    if 'inference_engine' not in st.session_state:
        st.session_state.inference_engine = InferenceEngine()
    
    if 'recovery_manager' not in st.session_state:
        st.session_state.recovery_manager = get_recovery_manager()
    
    if 'models_scanned' not in st.session_state:
        st.session_state.models_scanned = False
    
    if 'comparison_running' not in st.session_state:
        st.session_state.comparison_running = False
    
    if 'comparison_results' not in st.session_state:
        st.session_state.comparison_results = {}
    
    if 'error_messages' not in st.session_state:
        st.session_state.error_messages = []
    
    # 初始化流式输出性能监控
    if 'streaming_performance' not in st.session_state:
        st.session_state.streaming_performance = {
            'updates_processed': 0,
            'ui_refreshes': 0,
            'last_refresh_time': time.time(),
            'refresh_intervals': []
        }


def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        st.header("🔧 配置")
        
        # 模型目录配置
        st.subheader("模型目录")
        
        # 默认模型目录
        default_dirs = ["models/pytorch", "models/gguf"]
        model_dirs = st.text_area(
            "模型目录（每行一个）",
            value="\n".join(default_dirs),
            help="输入要扫描的模型目录路径，每行一个"
        )
        
        # 扫描选项
        recursive_scan = st.checkbox("递归扫描子目录", value=True)
        
        # 扫描按钮
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔍 扫描模型", use_container_width=True):
                scan_models(model_dirs.strip().split('\n'), recursive_scan)
        
        with col2:
            if st.button("🔄 刷新", use_container_width=True):
                refresh_models(model_dirs.strip().split('\n'), recursive_scan)
        
        # 显示扫描状态
        if st.session_state.models_scanned:
            stats = st.session_state.model_manager.get_model_statistics()
            st.success(f"✅ 发现 {stats['total_models']} 个模型")
            
            # 显示统计信息
            with st.expander("📊 模型统计"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("PyTorch模型", stats['pytorch_models'])
                    st.metric("GGUF模型", stats['gguf_models'])
                with col2:
                    st.metric("API模型", stats.get('api_models', 0))
                    st.metric("总大小", f"{stats['total_size_gb']:.2f} GB")
        
        # 显示流式输出性能监控（仅在有活动时显示）
        if st.session_state.get('comparison_running', False) or st.session_state.get('streaming_mode', False):
            perf = st.session_state.get('streaming_performance', {})
            if perf.get('ui_refreshes', 0) > 0:
                with st.expander("⚡ 流式性能监控"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("UI刷新次数", perf.get('ui_refreshes', 0))
                        st.metric("处理更新数", perf.get('updates_processed', 0))
                    with col2:
                        intervals = perf.get('refresh_intervals', [])
                        if intervals:
                            avg_interval = sum(intervals) / len(intervals)
                            refresh_rate = 1 / avg_interval if avg_interval > 0 else 0
                            st.metric("平均刷新率", f"{refresh_rate:.1f} Hz")
                            st.metric("平均间隔", f"{avg_interval*1000:.0f} ms")
        
        # API模型管理
        st.divider()
        st.subheader("🌐 API模型")
        
        with st.expander("➕ 添加Doubao模型"):
            with st.form("add_doubao_model"):
                st.write("添加新的Doubao模型")
                
                doubao_model_id = st.text_input(
                    "模型ID",
                    value="doubao-seed-1-6-250615",
                    help="Doubao模型的ID，如doubao-seed-1-6-250615"
                )
                
                doubao_display_name = st.text_input(
                    "显示名称",
                    value="",
                    help="在界面中显示的名称"
                )
                
                doubao_api_key = st.text_input(
                    "API Key (可选)",
                    value="",
                    type="password",
                    help="留空则使用环境变量ARK_API_KEY"
                )
                
                doubao_base_url = st.text_input(
                    "Base URL (可选)",
                    value="https://ark.cn-beijing.volces.com/api/v3",
                    help="API基础URL"
                )
                
                if st.form_submit_button("添加模型"):
                    try:
                        model_info = st.session_state.model_manager.add_doubao_model(
                            model_id=doubao_model_id,
                            model_name=doubao_model_id.split('-')[-1] if '-' in doubao_model_id else doubao_model_id,
                            display_name=doubao_display_name or f"Doubao {doubao_model_id}"
                        )
                        st.success(f"✅ 已添加Doubao模型: {model_info.name}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"添加模型失败: {str(e)}")
        
        # 显示已添加的API模型
        api_models = st.session_state.model_manager.api_manager.get_api_models()
        if api_models:
            with st.expander("📋 已配置的API模型"):
                for model in api_models:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.text(f"🌐 {model.name}")
                        st.caption(f"ID: {model.path}")
                    with col2:
                        if st.button("🗑️", key=f"remove_{model.id}", help="删除模型"):
                            st.session_state.model_manager.api_manager.remove_model(model.id)
                            st.rerun()


def scan_models(directories: List[str], recursive: bool = True):
    """扫描模型"""
    try:
        # 保存操作状态
        save_operation_state("扫描模型前")
        
        # 使用增强的加载指示器
        render_loading_overlay("正在扫描模型目录...", show_progress=True, progress=0.3)
        
        # 过滤空目录
        valid_dirs = [d.strip() for d in directories if d.strip()]
        
        if not valid_dirs:
            show_toast_notification("请输入至少一个有效的目录路径", NotificationType.ERROR)
            return
        
        # 执行扫描
        result = st.session_state.model_manager.scan_models(valid_dirs, recursive)
        
        st.session_state.models_scanned = True
        
        # 显示扫描结果
        if result.valid_models > 0:
            show_toast_notification(f"扫描完成！发现 {result.valid_models} 个有效模型", NotificationType.SUCCESS)
        else:
            show_toast_notification("未发现任何有效模型", NotificationType.WARNING)
        
        # 显示错误信息
        if result.errors:
            with st.expander("⚠️ 扫描警告"):
                for error in result.errors:
                    st.warning(error)
    
    except Exception as e:
        # 使用错误处理器处理错误
        error_context = {
            'operation': 'scan_models',
            'directories': directories,
            'recursive': recursive
        }
        
        def show_error_callback(error_info):
            st.error(f"扫描失败: {error_info.user_message}")
            
            # 显示建议
            if error_info.suggestions:
                with st.expander("💡 解决建议"):
                    for suggestion in error_info.suggestions:
                        st.info(f"• {suggestion}")
        
        handle_error(e, error_context, show_error_callback)
        logger.error(f"模型扫描失败: {str(e)}")


def refresh_models(directories: List[str], recursive: bool = True):
    """刷新模型列表"""
    try:
        # 保存操作状态
        save_operation_state("刷新模型前")
        
        render_loading_overlay("正在刷新模型列表...", show_progress=True, progress=0.5)
        
        valid_dirs = [d.strip() for d in directories if d.strip()]
        
        if not valid_dirs:
            show_toast_notification("请输入至少一个有效的目录路径", NotificationType.ERROR)
            return
        
        result = st.session_state.model_manager.refresh_models(valid_dirs, recursive)
        st.session_state.models_scanned = True
        
        show_toast_notification(f"刷新完成！发现 {result.valid_models} 个有效模型", NotificationType.SUCCESS)
    
    except Exception as e:
        # 使用错误处理器处理错误
        error_context = {
            'operation': 'refresh_models',
            'directories': directories,
            'recursive': recursive
        }
        
        def show_error_callback(error_info):
            st.error(f"刷新失败: {error_info.user_message}")
            
            # 显示建议
            if error_info.suggestions:
                with st.expander("💡 解决建议"):
                    for suggestion in error_info.suggestions:
                        st.info(f"• {suggestion}")
        
        handle_error(e, error_context, show_error_callback)
        logger.error(f"模型刷新失败: {str(e)}")


def render_model_selection():
    """渲染模型选择区域"""
    st.subheader("📋 模型选择")
    
    if not st.session_state.models_scanned:
        st.info("请先在侧边栏扫描模型目录")
        return
    
    available_models = st.session_state.model_manager.get_available_models()
    
    if not available_models:
        st.warning("未发现任何可用模型，请检查模型目录配置")
        return
    
    # 模型选择界面
    selected_model_ids = st.session_state.model_manager.get_selected_model_ids()
    
    # 创建模型选择选项
    model_options = {}
    for model in available_models:
        display_name = f"{model.name} ({model.model_type.value.upper()}) - {model.size / (1024*1024):.1f}MB"
        model_options[display_name] = model.id
    
    # 多选框
    selected_displays = []
    for display_name, model_id in model_options.items():
        if model_id in selected_model_ids:
            selected_displays.append(display_name)
    
    new_selected_displays = st.multiselect(
        f"选择模型进行比较 (最多{st.session_state.model_manager.MAX_SELECTED_MODELS}个)",
        options=list(model_options.keys()),
        default=selected_displays,
        help="选择要进行比较的模型"
    )
    
    # 更新选择 - 只在选择实际改变时调用
    new_selected_ids = [model_options[display] for display in new_selected_displays]
    current_selected_ids = st.session_state.model_manager.get_selected_model_ids()
    
    # 检查选择是否发生了变化
    if set(new_selected_ids) != set(current_selected_ids):
        try:
            st.session_state.model_manager.select_models(new_selected_ids)
            
            # 显示选择状态
            if new_selected_ids:
                st.success(f"已选择 {len(new_selected_ids)} 个模型")
            
        except ModelSelectionError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"选择模型时发生错误: {str(e)}")
    elif new_selected_ids:
        # 选择没有变化，但仍有选中的模型，显示当前状态（不重复调用select_models）
        st.success(f"已选择 {len(new_selected_ids)} 个模型")


def render_model_config():
    """渲染模型配置区域"""
    selected_models = st.session_state.model_manager.get_selected_models()
    
    if not selected_models:
        return
    
    st.subheader("⚙️ 模型配置")
    
    # 为每个选中的模型创建配置界面
    for model in selected_models:
        with st.expander(f"配置 {model.name} ({model.model_type.value.upper()})"):
            render_single_model_config(model)


def render_single_model_config(model):
    """渲染单个模型的配置界面"""
    config_manager = st.session_state.config_manager
    
    # 获取当前配置
    current_config = config_manager.get_model_config(model.id)
    
    # 创建配置表单
    with st.form(f"config_form_{model.id}"):
        st.write(f"**{model.name}** 配置参数")
        
        # 通用参数
        col1, col2 = st.columns(2)
        
        with col1:
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=current_config.temperature,
                step=0.1,
                help="控制输出的随机性"
            )
            
            max_tokens = st.number_input(
                "Max Tokens",
                min_value=1,
                max_value=4096,
                value=current_config.max_tokens,
                help="最大生成token数量"
            )
        
        with col2:
            top_p = st.slider(
                "Top P",
                min_value=0.0,
                max_value=1.0,
                value=current_config.top_p,
                step=0.05,
                help="核采样参数"
            )
        
        # 模型特定参数
        if model.model_type == ModelType.PYTORCH:
            st.write("**PyTorch 特定参数**")
            do_sample = st.checkbox(
                "Do Sample",
                value=current_config.do_sample,
                help="是否使用采样生成"
            )
            
            torch_dtype = st.selectbox(
                "Torch Dtype",
                options=["auto", "float16", "float32", "bfloat16"],
                index=["auto", "float16", "float32", "bfloat16"].index(current_config.torch_dtype),
                help="PyTorch数据类型"
            )
            
            low_cpu_mem_usage = st.checkbox(
                "Low CPU Memory Usage",
                value=current_config.low_cpu_mem_usage,
                help="启用低CPU内存使用模式"
            )
        
        elif model.model_type == ModelType.GGUF:
            st.write("**GGUF 特定参数**")
            col3, col4 = st.columns(2)
            
            with col3:
                top_k = st.number_input(
                    "Top K",
                    min_value=1,
                    max_value=100,
                    value=current_config.top_k,
                    help="Top-K采样参数"
                )
                
                repeat_penalty = st.slider(
                    "Repeat Penalty",
                    min_value=1.0,
                    max_value=2.0,
                    value=current_config.repeat_penalty,
                    step=0.05,
                    help="重复惩罚参数"
                )
            
            with col4:
                n_ctx = st.number_input(
                    "Context Length",
                    min_value=512,
                    max_value=8192,
                    value=current_config.n_ctx,
                    help="上下文长度"
                )
                
                use_gpu = st.checkbox(
                    "Use GPU",
                    value=current_config.use_gpu,
                    help="启用GPU加速"
                )
        
        elif model.model_type == ModelType.OPENAI_API:
            st.write("**OpenAI API 特定参数**")
            col5, col6 = st.columns(2)
            
            with col5:
                api_key = st.text_input(
                    "API Key",
                    value=current_config.api_key or "",
                    type="password",
                    help="OpenAI API密钥（留空则使用环境变量）"
                )
                
                base_url = st.text_input(
                    "Base URL",
                    value=current_config.base_url or "https://ark.cn-beijing.volces.com/api/v3",
                    help="API基础URL"
                )
                
                model_name = st.text_input(
                    "Model Name",
                    value=current_config.model_name or "doubao-seed-1-6-250615",
                    help="模型名称"
                )
            
            with col6:
                presence_penalty = st.slider(
                    "Presence Penalty",
                    min_value=-2.0,
                    max_value=2.0,
                    value=current_config.presence_penalty,
                    step=0.1,
                    help="存在惩罚参数"
                )
                
                frequency_penalty = st.slider(
                    "Frequency Penalty",
                    min_value=-2.0,
                    max_value=2.0,
                    value=current_config.frequency_penalty,
                    step=0.1,
                    help="频率惩罚参数"
                )
                
                stream = st.checkbox(
                    "Enable Streaming",
                    value=current_config.stream,
                    help="启用流式输出"
                )
        
        # 保存按钮
        if st.form_submit_button("💾 保存配置"):
            try:
                # 创建新配置
                new_config = ModelConfig(
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    do_sample=do_sample if model.model_type == ModelType.PYTORCH else current_config.do_sample,
                    torch_dtype=torch_dtype if model.model_type == ModelType.PYTORCH else current_config.torch_dtype,
                    low_cpu_mem_usage=low_cpu_mem_usage if model.model_type == ModelType.PYTORCH else current_config.low_cpu_mem_usage,
                    top_k=top_k if model.model_type == ModelType.GGUF else current_config.top_k,
                    repeat_penalty=repeat_penalty if model.model_type == ModelType.GGUF else current_config.repeat_penalty,
                    n_ctx=n_ctx if model.model_type == ModelType.GGUF else current_config.n_ctx,
                    use_gpu=use_gpu if model.model_type == ModelType.GGUF else current_config.use_gpu,
                    api_key=api_key if model.model_type == ModelType.OPENAI_API else current_config.api_key,
                    base_url=base_url if model.model_type == ModelType.OPENAI_API else current_config.base_url,
                    model_name=model_name if model.model_type == ModelType.OPENAI_API else current_config.model_name,
                    stream=stream if model.model_type == ModelType.OPENAI_API else current_config.stream,
                    presence_penalty=presence_penalty if model.model_type == ModelType.OPENAI_API else current_config.presence_penalty,
                    frequency_penalty=frequency_penalty if model.model_type == ModelType.OPENAI_API else current_config.frequency_penalty,
                )
                
                # 验证并保存配置
                config_manager.save_model_config(model.id, new_config, model.model_type)
                st.success(f"✅ {model.name} 配置已保存")
                
            except ValidationError as e:
                st.error(f"配置验证失败: {str(e)}")
            except Exception as e:
                st.error(f"保存配置失败: {str(e)}")


def render_prompt_input():
    """渲染提示词输入区域"""
    st.subheader("💬 提示词输入")
    
    # 输出模式选择
    col_mode1, col_mode2 = st.columns(2)
    with col_mode1:
        output_mode = st.radio(
            "输出模式",
            options=["流式输出", "完整输出"],
            index=0,
            help="流式输出：实时显示生成过程；完整输出：等待生成完成后一次性显示"
        )
    
    with col_mode2:
        if output_mode == "完整输出":
            st.info("💡 完整输出模式将优化性能，减少资源消耗")
        else:
            st.info("💡 流式输出模式可实时查看生成进度")
    
    # 存储输出模式到会话状态
    st.session_state.streaming_mode = (output_mode == "流式输出")
    
    # 提示词输入
    prompt = st.text_area(
        "输入您的提示词",
        height=150,
        placeholder="请输入您想要各个模型回答的问题或完成的任务...",
        help="输入提示词，所有选中的模型将基于此提示词生成回答"
    )
    
    # 比较控制按钮
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        start_comparison = st.button(
            "🚀 开始比较",
            disabled=not prompt.strip() or not st.session_state.model_manager.get_selected_models() or st.session_state.comparison_running,
            use_container_width=True,
            type="primary",
            help="开始模型比较 (Ctrl+Enter)"
        )
    
    with col2:
        if st.button("⏹️ 停止", disabled=not st.session_state.comparison_running, use_container_width=True, help="停止当前比较"):
            if show_operation_confirmation("停止比较", "确定要停止当前的模型比较吗？"):
                st.session_state.comparison_running = False
                show_toast_notification("比较已停止", NotificationType.INFO)
                st.rerun()
    
    with col3:
        if st.button("🗑️ 清空结果", use_container_width=True, help="清空比较结果"):
            if show_operation_confirmation("清空结果", "确定要清空所有比较结果吗？此操作不可撤销。", danger=True):
                save_operation_state("清空结果前")
                st.session_state.comparison_results = {}
                show_toast_notification("结果已清空", NotificationType.INFO)
                st.rerun()
    
    with col4:
        # 添加撤销/重做按钮
        render_undo_redo_controls()
    
    # 存储当前提示词到会话状态
    st.session_state.current_prompt = prompt.strip()
    
    # 开始比较
    if start_comparison:
        start_model_comparison(prompt.strip(), st.session_state.get('streaming_mode', True))
    
    return prompt


def start_model_comparison(prompt: str, streaming: bool = True):
    """开始模型比较"""
    selected_models = st.session_state.model_manager.get_selected_models()
    
    if not selected_models:
        show_toast_notification("请先选择要比较的模型", NotificationType.ERROR)
        return
    
    # 保存操作状态
    save_operation_state("开始比较前")
    
    st.session_state.comparison_running = True
    st.session_state.comparison_results = {}
    st.session_state.streaming_mode = streaming
    
    # 初始化输出容器
    initialize_output_containers(selected_models)
    
    # 显示增强的进度指示器
    mode_text = "流式" if streaming else "完整"
    show_enhanced_progress(
        progress_id="model_comparison",
        title=f"模型比较 ({mode_text}模式)",
        current=0,
        total=len(selected_models),
        message=f"开始{mode_text}比较 {len(selected_models)} 个模型"
    )
    
    show_toast_notification(f"开始{mode_text}比较 {len(selected_models)} 个模型", NotificationType.INFO)
    
    # 执行真实的模型推理
    execute_real_model_comparison(prompt, selected_models, streaming)


def initialize_output_containers(selected_models):
    """初始化输出容器"""
    if 'output_containers' not in st.session_state:
        st.session_state.output_containers = {}
    
    if 'output_status' not in st.session_state:
        st.session_state.output_status = {}
    
    # 为每个模型创建输出容器
    for model in selected_models:
        st.session_state.output_containers[model.id] = {
            'content': "",
            'status': "pending",
            'start_time': None,
            'end_time': None,
            'token_count': 0,
            'error': None
        }
        st.session_state.output_status[model.id] = "pending"


def execute_real_model_comparison(prompt: str, selected_models, streaming: bool = True):
    """执行真实的模型比较"""
    import time
    import threading
    from queue import Queue
    
    # 创建结果队列用于线程间通信
    result_queue = Queue()
    
    # 获取必要的对象引用
    inference_engine = st.session_state.inference_engine
    
    def progress_callback(task_id: str, model_id: str, status: str):
        """进度回调函数 - 通过队列传递状态更新"""
        try:
            result_queue.put({
                'type': 'progress',
                'model_id': model_id,
                'status': status,
                'timestamp': time.time()
            })
        except Exception as e:
            logger.error(f"进度回调错误: {str(e)}")
    
    def run_inference():
        """在后台线程中运行推理"""
        try:
            # 为每个模型发送初始状态
            for model in selected_models:
                result_queue.put({
                    'type': 'init',
                    'model_id': model.id,
                    'status': 'pending',
                    'timestamp': time.time()
                })
            
            # 根据模式选择推理方法
            if streaming:
                inference_method = inference_engine.run_inference
            else:
                inference_method = inference_engine.run_non_streaming_inference
            
            # 执行推理
            for result in inference_method(prompt, selected_models, progress_callback):
                # 通过队列发送结果更新
                model_info = next((m for m in selected_models if m.id == result.model_id), None)
                if model_info:
                    result_queue.put({
                        'type': 'result',
                        'model_id': result.model_id,
                        'model_name': model_info.name,
                        'model_type': model_info.model_type.value,
                        'content': result.content,
                        'error': result.error,
                        'is_complete': result.is_complete,
                        'streaming': streaming,
                        'stats': {
                            'start_time': result.stats.start_time if result.stats else time.time(),
                            'end_time': result.stats.end_time if result.stats else time.time(),
                            'token_count': result.stats.token_count if result.stats else 0,
                            'duration': (result.stats.end_time - result.stats.start_time) if result.stats and result.stats.end_time else 0,
                            'tokens_per_second': result.stats.tokens_per_second if result.stats else 0
                        },
                        'timestamp': time.time()
                    })
        
        except Exception as e:
            logger.error(f"推理过程中出现错误: {str(e)}")
            # 为所有模型发送错误状态
            for model in selected_models:
                result_queue.put({
                    'type': 'error',
                    'model_id': model.id,
                    'model_name': model.name,
                    'model_type': model.model_type.value,
                    'error': str(e),
                    'timestamp': time.time()
                })
        
        finally:
            # 发送完成信号
            result_queue.put({
                'type': 'finished',
                'timestamp': time.time()
            })
    
    # 在后台线程中启动推理
    inference_thread = threading.Thread(target=run_inference, daemon=True)
    inference_thread.start()
    
    # 存储线程引用和结果队列
    st.session_state.inference_thread = inference_thread
    st.session_state.result_queue = result_queue
    
    # 立即处理一次队列中的结果
    process_inference_results()


def process_inference_results():
    """处理推理结果队列中的更新"""
    if 'result_queue' not in st.session_state:
        return
    
    result_queue = st.session_state.result_queue
    updates_processed = 0
    streaming_updates = 0
    
    # 处理队列中的所有结果，对流式更新给予更高优先级
    while not result_queue.empty() and updates_processed < 100:  # 增加处理数量以支持更频繁的流式更新
        try:
            update = result_queue.get_nowait()
            updates_processed += 1
            
            if update['type'] == 'init':
                # 初始化模型状态
                model_id = update['model_id']
                if 'output_containers' not in st.session_state:
                    st.session_state.output_containers = {}
                
                st.session_state.output_containers[model_id] = {
                    'content': "",
                    'status': update['status'],
                    'start_time': update['timestamp'],
                    'end_time': None,
                    'token_count': 0,
                    'error': None
                }
            
            elif update['type'] == 'progress':
                # 更新进度状态
                model_id = update['model_id']
                status = update['status']
                
                if model_id in st.session_state.output_containers:
                    container = st.session_state.output_containers[model_id]
                    
                    # 检查是否是流式更新
                    if status.startswith("streaming_update:") or status.startswith("streaming_final:"):
                        streaming_updates += 1
                        # 解析流式更新: streaming_update:content:token_count 或 streaming_final:content:token_count
                        parts = status.split(":", 2)
                        if len(parts) >= 3:
                            streaming_content = parts[1]
                            token_count = int(parts[2]) if parts[2].isdigit() else 0
                            is_final = status.startswith("streaming_final:")
                            
                            # 更新流式内容
                            container['content'] = streaming_content
                            container['token_count'] = token_count
                            container['status'] = 'completed' if is_final else 'streaming'
                            
                            # 同时更新比较结果以便实时显示
                            if model_id not in st.session_state.comparison_results:
                                st.session_state.comparison_results[model_id] = {}
                            
                            # 计算当前速度
                            start_time = container.get('start_time', time.time())
                            current_time = time.time()
                            duration = current_time - start_time
                            tokens_per_second = token_count / duration if duration > 0 else 0
                            
                            st.session_state.comparison_results[model_id].update({
                                'content': streaming_content,
                                'status': 'completed' if is_final else 'streaming',
                                'stats': {
                                    'token_count': token_count,
                                    'start_time': start_time,
                                    'end_time': current_time if is_final else None,
                                    'duration': duration,
                                    'tokens_per_second': tokens_per_second
                                }
                            })
                    elif "加载" in status:
                        container['status'] = 'loading'
                    elif "生成" in status or "运行" in status:
                        container['status'] = 'running'
                    elif "完成" in status:
                        container['status'] = 'completed'
                    elif "错误" in status or "失败" in status:
                        container['status'] = 'error'
            
            elif update['type'] == 'result':
                # 更新推理结果
                model_id = update['model_id']
                
                # 更新输出容器
                if model_id in st.session_state.output_containers:
                    container = st.session_state.output_containers[model_id]
                    container['content'] = update['content']
                    container['end_time'] = update['timestamp']
                    container['token_count'] = update['stats']['token_count']
                    
                    if update['error']:
                        container['status'] = 'error'
                        container['error'] = update['error']
                    else:
                        container['status'] = 'completed' if update['is_complete'] else 'running'
                
                # 更新比较结果
                result_status = 'completed' if update['is_complete'] else 'error' if update['error'] else 'running'
                
                # 如果之前是流式状态且现在完成，确保状态正确转换
                if model_id in st.session_state.comparison_results:
                    prev_status = st.session_state.comparison_results[model_id].get('status')
                    if prev_status == 'streaming' and update['is_complete']:
                        result_status = 'completed'
                
                st.session_state.comparison_results[model_id] = {
                    'model_name': update['model_name'],
                    'model_type': update['model_type'],
                    'status': result_status,
                    'content': update['content'],
                    'error': update['error'],
                    'stats': update['stats'],
                    'streaming': update.get('streaming', False)
                }
            
            elif update['type'] == 'error':
                # 处理错误
                model_id = update['model_id']
                
                if model_id in st.session_state.output_containers:
                    st.session_state.output_containers[model_id]['status'] = 'error'
                    st.session_state.output_containers[model_id]['error'] = update['error']
                
                st.session_state.comparison_results[model_id] = {
                    'model_name': update['model_name'],
                    'model_type': update['model_type'],
                    'status': 'error',
                    'content': '',
                    'error': update['error'],
                    'stats': {
                        'start_time': update['timestamp'],
                        'end_time': update['timestamp'],
                        'token_count': 0,
                        'duration': 0,
                        'tokens_per_second': 0
                    }
                }
            
            elif update['type'] == 'finished':
                # 推理完成
                st.session_state.comparison_running = False
                # 清理队列引用
                if 'result_queue' in st.session_state:
                    del st.session_state.result_queue
                break
                
        except Exception as e:
            logger.error(f"处理推理结果时出错: {str(e)}")
            break
    
    # 如果处理了更新，触发重新运行以刷新UI
    if updates_processed > 0:
        # 检查是否有流式更新
        has_streaming_update = any(
            container.get('status') == 'streaming' 
            for container in st.session_state.get('output_containers', {}).values()
        )
        
        # 检查是否有重要状态变化（开始、完成、错误）
        has_important_update = any(
            container.get('status') in ['loading', 'running', 'completed', 'error']
            for container in st.session_state.get('output_containers', {}).values()
        )
        
        # 在流式模式下更积极地刷新UI以实现实时显示
        if st.session_state.get('streaming_mode', True):
            # 流式模式：有任何更新都刷新，特别是流式更新
            if streaming_updates > 0 or has_streaming_update or has_important_update:
                # 更新性能监控
                current_time = time.time()
                perf = st.session_state.streaming_performance
                perf['ui_refreshes'] += 1
                perf['updates_processed'] += updates_processed
                
                # 记录刷新间隔
                if perf['last_refresh_time']:
                    interval = current_time - perf['last_refresh_time']
                    perf['refresh_intervals'].append(interval)
                    # 只保留最近50次刷新的记录
                    if len(perf['refresh_intervals']) > 50:
                        perf['refresh_intervals'] = perf['refresh_intervals'][-50:]
                
                perf['last_refresh_time'] = current_time
                st.rerun()
        elif has_important_update or st.session_state.get('comparison_running', False):
            # 非流式模式：只在重要状态变化时刷新
            st.rerun()


def render_model_outputs():
    """渲染模型输出显示区域"""
    # 检查是否有任何输出数据或正在运行的比较
    has_output_data = bool(st.session_state.get('output_containers', {}))
    has_results = bool(st.session_state.comparison_results)
    is_running = st.session_state.get('comparison_running', False)
    
    if not has_output_data and not has_results and not is_running:
        return
    
    selected_models = st.session_state.model_manager.get_selected_models()
    
    if not selected_models:
        return
    
    # 创建并排的列布局 - 优化布局以支持更多模型
    if len(selected_models) == 1:
        cols = st.columns(1)
    elif len(selected_models) == 2:
        cols = st.columns(2)
    elif len(selected_models) == 3:
        cols = st.columns(3)
    elif len(selected_models) == 4:
        cols = st.columns(2)  # 2x2布局
    else:
        # 超过4个模型时使用滚动布局
        cols = st.columns(min(len(selected_models), 3))
    
    # 为每个模型创建输出区域
    for i, model in enumerate(selected_models):
        if len(selected_models) <= 3:
            col_index = i
        elif len(selected_models) == 4:
            # 4个模型时使用2x2布局
            col_index = i % 2
        else:
            # 超过4个模型时循环使用列
            col_index = i % len(cols)
        
        with cols[col_index]:
            # 如果是4个模型的第3、4个，需要在第二行显示
            if len(selected_models) == 4 and i >= 2:
                if i == 2:  # 第三个模型，开始新行
                    st.markdown("---")  # 分隔线
            
            render_single_model_output(model, i >= 2 if len(selected_models) == 4 else False)


def render_streaming_output(model_id: str, content: str, token_count: int, start_time: float):
    """渲染优化的流式输出显示"""
    # 处理空内容的情况
    display_content = content if content.strip() else "正在等待模型开始生成..."
    
    # 使用JavaScript实现自动滚动到底部的流式显示
    streaming_html = f"""
    <div id="streaming_{model_id}" style="
        background-color: #f8f9fa; 
        padding: 15px; 
        border-radius: 8px; 
        max-height: 350px; 
        overflow-y: auto; 
        white-space: pre-wrap; 
        font-family: 'Courier New', monospace;
        border: 2px solid #28a745;
        font-size: 14px;
        line-height: 1.4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        min-height: 100px;
    ">
{display_content}
    </div>
    <script>
        // 自动滚动到底部
        var element = document.getElementById('streaming_{model_id}');
        if (element) {{
            element.scrollTop = element.scrollHeight;
        }}
    </script>
    """
    
    st.markdown(streaming_html, unsafe_allow_html=True)
    
    # 实时统计信息
    if start_time:
        duration = time.time() - start_time
        speed = token_count / duration if duration > 0 and token_count > 0 else 0
        
        # 使用列布局显示统计信息
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Tokens", token_count, delta=None)
        with col2:
            st.metric("用时", f"{duration:.1f}s", delta=None)
        with col3:
            if speed > 0:
                st.metric("速度", f"{speed:.1f} t/s", delta=None)
            else:
                st.metric("速度", "计算中...", delta=None)


def render_single_model_output(model, is_second_row=False):
    """渲染单个模型的输出区域"""
    # 模型标题和状态
    if model.model_type == ModelType.PYTORCH:
        model_type_icon = "🔥"
    elif model.model_type == ModelType.GGUF:
        model_type_icon = "⚡"
    elif model.model_type == ModelType.OPENAI_API:
        model_type_icon = "🌐"
    else:
        model_type_icon = "❓"
    
    # 获取输出状态 - 优先使用output_containers中的数据，因为它更实时
    output_data = st.session_state.output_containers.get(model.id, {})
    result_data = st.session_state.comparison_results.get(model.id, {})
    
    # 优先使用output_containers中的状态和内容，因为它们更实时
    status = output_data.get('status', result_data.get('status', 'pending'))
    content = output_data.get('content', result_data.get('content', ''))
    error = output_data.get('error') or result_data.get('error')
    
    # 状态指示器
    status_icons = {
        'pending': '⏳',
        'loading': '📥',
        'running': '🔄',
        'streaming': '📝',
        'completed': '✅',
        'error': '❌'
    }
    
    status_colors = {
        'pending': 'orange',
        'loading': 'blue',
        'running': 'blue',
        'streaming': 'green', 
        'completed': 'green',
        'error': 'red'
    }
    
    status_icon = status_icons.get(status, '❓')
    status_color = status_colors.get(status, 'gray')
    
    # 模型标题 - 优化样式，添加实时状态显示
    token_count = output_data.get('token_count', result_data.get('stats', {}).get('token_count', 0))
    status_text = status
    if status == 'streaming' and token_count > 0:
        status_text = f"流式生成中 ({token_count} tokens)"
    elif status == 'running':
        status_text = "正在生成"
    
    st.markdown(f"""
    <div style="
        border: 2px solid {status_color}; 
        border-radius: 10px; 
        padding: 12px; 
        margin-bottom: 15px;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    ">
        <h4 style="margin: 0; color: #333;">{status_icon} {model_type_icon} {model.name}</h4>
        <small style="color: #666;">{model.model_type.value.upper()} | {status_text}</small>
    </div>
    """, unsafe_allow_html=True)
    
    # 进度指示器 - 优化显示
    if status in ['running', 'streaming', 'loading']:
        if status == 'streaming' and token_count > 0:
            # 流式输出时显示动态进度条
            progress_value = min(token_count / 200, 1.0)  # 假设200个token为满进度
            st.progress(progress_value)
        elif status == 'loading':
            # 加载时显示不确定进度条
            st.progress(0.3)
        else:
            # 运行时显示中等进度
            st.progress(0.5)
    
    # 输出内容区域 - 改进逻辑以支持实时显示
    if error:
        st.error(f"❌ 错误: {error}")
    elif content or status in ['streaming', 'running']:
        # 即使内容为空，如果状态是streaming或running，也显示占位符
        if status == 'streaming':
            # 使用优化的流式输出组件
            start_time = output_data.get('start_time')
            if content:
                render_streaming_output(model.id, content, token_count, start_time)
            else:
                # 流式状态但还没有内容时显示等待提示
                st.info("📝 准备开始流式生成...")
        elif content:
            # 有内容时根据状态选择显示方式
            if status == 'completed':
                # 完成后使用增强的高级文本查看器
                container_key = f"main_output_{model.id}_{hash(content[:50])}"
                render_advanced_text_viewer(
                    content=content,
                    title="模型输出",
                    container_key=container_key,
                    enable_line_numbers=True,
                    enable_word_wrap=True,
                    enable_syntax_highlighting=True,
                    language="markdown",
                    max_height=400
                )
            else:
                # 其他状态下显示简单文本
                st.text_area(
                    "当前输出",
                    value=content,
                    height=300,
                    disabled=True,
                    key=f"output_{model.id}_{len(content)}"
                )
        else:
            # 根据状态显示相应的提示信息
            if status == 'pending':
                st.info("⏳ 等待开始...")
            elif status == 'loading':
                st.info("📥 正在加载模型...")
            elif status == 'running':
                st.info("🔄 正在生成中...")
    else:
        # 默认状态显示
        st.info("⏳ 等待开始...")
    
    # 统计信息 - 支持实时统计显示
    stats = result_data.get('stats', {}) if result_data else {}
    
    # 如果是流式状态，显示实时统计
    if status == 'streaming' and output_data.get('start_time'):
        start_time = output_data.get('start_time')
        current_time = time.time()
        duration = current_time - start_time
        speed = token_count / duration if duration > 0 and token_count > 0 else 0
        
        with st.expander("📊 实时统计", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("已生成", f"{token_count} tokens")
            with col2:
                st.metric("用时", f"{duration:.1f}s")
            with col3:
                st.metric("速度", f"{speed:.1f} t/s")
    
    # 如果已完成，显示最终统计
    elif status == 'completed' and stats:
        with st.expander("📈 性能统计"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                duration = stats.get('duration', 0)
                st.metric("用时", f"{duration:.1f}s")
            
            with col2:
                final_token_count = stats.get('token_count', 0)
                st.metric("Token数", final_token_count)
            
            with col3:
                if duration > 0:
                    tps = final_token_count / duration
                    st.metric("速度", f"{tps:.1f} t/s")
                else:
                    st.metric("速度", "N/A")


def render_comparison_status():
    """渲染比较状态和进度"""
    if st.session_state.comparison_running:
        st.subheader("🔄 比较进行中")
        
        # 显示进度信息
        selected_models = st.session_state.model_manager.get_selected_models()
        completed_count = len([r for r in st.session_state.comparison_results.values() 
                             if r.get('status') == 'completed'])
        
        # 使用增强的进度指示器
        show_enhanced_progress(
            progress_id="main_comparison",
            title="模型比较进度",
            current=completed_count,
            total=len(selected_models),
            message=f"正在处理第 {completed_count + 1} 个模型" if completed_count < len(selected_models) else "即将完成",
            show_eta=True
        )
        
        # 显示状态指示器
        render_status_indicator("running", f"正在比较 {len(selected_models)} 个模型", show_spinner=True)
        
        # 在比较进行中也显示实时流式输出
        if st.session_state.get('streaming_mode', True):
            st.subheader("📊 实时模型输出")
            render_model_outputs()
    
    # 无论是否在运行，只要有结果就显示（支持流式输出）
    if st.session_state.comparison_results:
        if not st.session_state.comparison_running:
            st.subheader("📊 比较结果摘要")
            
            # 显示加载的会话信息（如果有）
            if 'loaded_session_info' in st.session_state:
                session_info = st.session_state.loaded_session_info
                st.info(f"📂 已加载会话: {session_info.get('name', '未命名')} ({session_info.get('created_at', '')[:19]})")
            
            # 显示结果统计
            total_models = len(st.session_state.comparison_results)
            completed_models = len([r for r in st.session_state.comparison_results.values() 
                                   if r.get('status') == 'completed'])
            error_models = len([r for r in st.session_state.comparison_results.values() 
                               if r.get('error')])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("总模型数", total_models)
            with col2:
                st.metric("成功完成", completed_models)
            with col3:
                st.metric("出现错误", error_models)
            
            # 渲染模型输出
            render_model_outputs()
            
            # 添加导出和会话管理功能
            st.divider()
            
            # 创建选项卡
            export_tab, session_tab = st.tabs(["📤 导出结果", "💾 会话管理"])
            
            with export_tab:
                from multi_llm_comparator.ui.components import render_export_options
                render_export_options(st.session_state.comparison_results)
            
            with session_tab:
                from multi_llm_comparator.ui.components import render_session_management
                
                # 获取当前提示词和模型信息
                current_prompt = st.session_state.get('current_prompt', '')
                selected_models = st.session_state.model_manager.get_selected_models()
                models_info = [
                    {
                        'id': model.id,
                        'name': model.name,
                        'type': model.model_type.value,
                        'path': model.path,
                        'size': model.size
                    }
                    for model in selected_models
                ]
                
                render_session_management(
                    st.session_state.comparison_results,
                    current_prompt,
                    models_info
                )
        elif not st.session_state.get('streaming_mode', True):
            # 非流式模式下，即使在运行中也不显示输出，只在完成后显示
            pass


def main():
    """主应用程序入口点"""
    st.set_page_config(
        page_title="多LLM模型比较器",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 添加CSS优化以提高流式输出性能
    st.markdown("""
    <style>
    /* 优化流式输出显示性能 */
    .streaming-output {
        font-family: 'Courier New', monospace;
        background-color: #f8f9fa;
        border: 2px solid #28a745;
        border-radius: 8px;
        padding: 15px;
        max-height: 350px;
        overflow-y: auto;
        white-space: pre-wrap;
        font-size: 14px;
        line-height: 1.4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        scroll-behavior: smooth;
    }
    
    /* 优化模型标题样式 */
    .model-header {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* 减少不必要的动画以提高性能 */
    .stProgress > div > div > div > div {
        transition: width 0.1s ease-in-out;
    }
    
    /* 优化滚动条样式 */
    .streaming-output::-webkit-scrollbar {
        width: 8px;
    }
    
    .streaming-output::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    
    .streaming-output::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 4px;
    }
    
    .streaming-output::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 初始化会话状态
    initialize_session_state()
    
    # 处理推理结果更新（如果有正在进行的推理）
    if st.session_state.get('comparison_running', False) or 'result_queue' in st.session_state:
        process_inference_results()
    
    # 主标题
    st.title("🤖 多LLM模型比较器")
    st.markdown("本地部署的多模型比较工具")
    
    # 渲染侧边栏
    render_sidebar()
    
    # 添加主要内容标识符（用于无障碍功能）
    st.markdown('<div id="main-content"></div>', unsafe_allow_html=True)
    
    # 显示键盘快捷键帮助
    keyboard_manager.render_shortcuts_help()
    
    # 显示交互式教程（仅在首次使用时）
    if st.session_state.get('show_tutorial', True):
        create_interactive_tutorial()
        if st.session_state.get('tutorial_step', 0) >= 6:  # 教程完成
            st.session_state.show_tutorial = False
    
    # 主内容区域
    # 模型选择
    render_model_selection()
    
    # 模型配置
    render_model_config()
    
    # 分隔线
    st.divider()
    
    # 提示词输入
    render_prompt_input()
    
    # 比较状态显示
    render_comparison_status()
    
    # 如果有正在进行的推理，设置自动刷新
    if st.session_state.get('comparison_running', False):
        # 检查是否有流式输出正在进行
        has_streaming = any(
            container.get('status') == 'streaming' 
            for container in st.session_state.get('output_containers', {}).values()
        )
        
        if has_streaming and st.session_state.get('streaming_mode', True):
            # 流式模式下更频繁刷新以实现实时显示
            time.sleep(0.05)  # 减少延迟，提高响应性
        else:
            # 非流式模式下较慢刷新
            time.sleep(0.3)  # 稍微提高非流式模式的响应性
        
        st.rerun()


if __name__ == "__main__":
    main()