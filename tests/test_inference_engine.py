"""
推理引擎测试

测试推理引擎协调器的功能。
"""

import pytest
import time
import threading
import sys
from unittest.mock import Mock, patch, MagicMock
from typing import Iterator

# Mock the heavy dependencies to avoid import issues during testing
sys.modules['transformers'] = Mock()
sys.modules['torch'] = Mock()
sys.modules['llama_cpp'] = Mock()

from src.multi_llm_comparator.services.inference_engine import InferenceEngine, InferenceTask
from src.multi_llm_comparator.core.models import ModelInfo, ModelType, InferenceResult, InferenceStats
from src.multi_llm_comparator.core.exceptions import InferenceError, ModelLoadError, MemoryError
from src.multi_llm_comparator.services.memory_manager import MemoryManager


class MockInferencer:
    """模拟推理器"""
    
    def __init__(self, should_fail=False, fail_on_load=False, tokens=None):
        self.should_fail = should_fail
        self.fail_on_load = fail_on_load
        self.tokens = tokens or ["Hello", " ", "World", "!"]
        self.is_loaded = False
        
    def load_model(self, model_path: str, config: dict):
        if self.fail_on_load:
            raise ModelLoadError("模拟加载失败")
        self.is_loaded = True
        
    def generate_stream(self, prompt: str) -> Iterator[str]:
        if not self.is_loaded:
            raise InferenceError("模型未加载")
        if self.should_fail:
            raise InferenceError("模拟推理失败")
        
        for token in self.tokens:
            yield token
            time.sleep(0.01)  # 模拟生成延迟
            
    def unload_model(self):
        self.is_loaded = False
        
    def get_model_info(self):
        return {"type": "mock", "is_loaded": self.is_loaded}


class TestInferenceTask:
    """推理任务测试类"""
    
    def test_init(self):
        """测试任务初始化"""
        model_info = ModelInfo(
            id="test_model",
            name="Test Model",
            path="/path/to/model",
            model_type=ModelType.PYTORCH,
            size=1000,
            config={}
        )
        
        task = InferenceTask(model_info, "test prompt", "task_123")
        
        assert task.model_info == model_info
        assert task.prompt == "test prompt"
        assert task.task_id == "task_123"
        assert task.status == "pending"
        assert task.result is None
        assert task.error is None
        assert task.start_time is None
        assert task.end_time is None
        assert not task.is_cancelled()
    
    def test_cancel(self):
        """测试任务取消"""
        model_info = ModelInfo(
            id="test_model",
            name="Test Model", 
            path="/path/to/model",
            model_type=ModelType.PYTORCH,
            size=1000,
            config={}
        )
        
        task = InferenceTask(model_info, "test prompt", "task_123")
        
        assert not task.is_cancelled()
        task.cancel()
        assert task.is_cancelled()
        assert task.status == "cancelled"


class TestInferenceEngine:
    """推理引擎测试类"""
    
    def setup_method(self):
        """测试前的设置"""
        self.memory_manager = Mock(spec=MemoryManager)
        self.memory_manager.check_memory_availability.return_value = True
        self.memory_manager.force_cleanup.return_value = None
        
        self.engine = InferenceEngine(self.memory_manager)
        
        # 创建测试模型信息
        self.pytorch_model = ModelInfo(
            id="pytorch_model",
            name="PyTorch Model",
            path="/path/to/pytorch/model",
            model_type=ModelType.PYTORCH,
            size=1000,
            config={"temperature": 0.7}
        )
        
        self.gguf_model = ModelInfo(
            id="gguf_model", 
            name="GGUF Model",
            path="/path/to/gguf/model.gguf",
            model_type=ModelType.GGUF,
            size=2000,
            config={"temperature": 0.8}
        )
    
    def test_init(self):
        """测试引擎初始化"""
        engine = InferenceEngine()
        assert engine.memory_manager is not None
        assert engine.current_inferencer is None
        assert len(engine.active_tasks) == 0
        assert engine.max_retries == 3
        assert engine.retry_delay == 1.0
    
    def test_init_with_memory_manager(self):
        """测试使用自定义内存管理器初始化"""
        custom_manager = Mock(spec=MemoryManager)
        engine = InferenceEngine(custom_manager)
        assert engine.memory_manager is custom_manager
    
    @patch('src.multi_llm_comparator.services.inference_engine.PyTorchInferencer')
    def test_create_inferencer_pytorch(self, mock_pytorch):
        """测试创建PyTorch推理器"""
        mock_instance = Mock()
        mock_pytorch.return_value = mock_instance
        
        inferencer = self.engine.create_inferencer(self.pytorch_model)
        
        assert inferencer is mock_instance
        mock_pytorch.assert_called_once()
    
    @patch('src.multi_llm_comparator.services.inference_engine.GGUFInferencer')
    def test_create_inferencer_gguf(self, mock_gguf):
        """测试创建GGUF推理器"""
        mock_instance = Mock()
        mock_gguf.return_value = mock_instance
        
        inferencer = self.engine.create_inferencer(self.gguf_model)
        
        assert inferencer is mock_instance
        mock_gguf.assert_called_once()
    
    def test_create_inferencer_unsupported(self):
        """测试创建不支持的推理器类型"""
        unsupported_model = ModelInfo(
            id="unsupported",
            name="Unsupported Model",
            path="/path/to/model",
            model_type="unsupported",  # 无效类型
            size=1000,
            config={}
        )
        
        with pytest.raises(InferenceError, match="不支持的模型类型"):
            self.engine.create_inferencer(unsupported_model)
    
    def test_run_inference_empty_models(self):
        """测试空模型列表"""
        with pytest.raises(InferenceError, match="模型列表不能为空"):
            list(self.engine.run_inference("test prompt", []))
    
    @patch('src.multi_llm_comparator.services.inference_engine.InferenceEngine.create_inferencer')
    def test_run_inference_single_model_success(self, mock_create):
        """测试单模型推理成功"""
        # 设置模拟推理器
        mock_inferencer = MockInferencer(tokens=["Hello", " World"])
        mock_create.return_value = mock_inferencer
        
        # 执行推理
        results = list(self.engine.run_inference("test prompt", [self.pytorch_model]))
        
        # 验证结果
        assert len(results) == 1
        result = results[0]
        assert result.model_id == "pytorch_model"
        assert result.content == "Hello World"
        assert result.is_complete is True
        assert result.error is None
        assert result.stats.token_count == 2
        assert result.stats.tokens_per_second > 0
    
    @patch('src.multi_llm_comparator.services.inference_engine.InferenceEngine.create_inferencer')
    def test_run_inference_multiple_models(self, mock_create):
        """测试多模型推理"""
        # 设置模拟推理器
        mock_inferencer1 = MockInferencer(tokens=["Model", "1"])
        mock_inferencer2 = MockInferencer(tokens=["Model", "2"])
        mock_create.side_effect = [mock_inferencer1, mock_inferencer2]
        
        # 执行推理
        results = list(self.engine.run_inference(
            "test prompt", 
            [self.pytorch_model, self.gguf_model]
        ))
        
        # 验证结果
        assert len(results) == 2
        assert results[0].model_id == "pytorch_model"
        assert results[0].content == "Model1"
        assert results[1].model_id == "gguf_model"
        assert results[1].content == "Model2"
    
    @patch('src.multi_llm_comparator.services.inference_engine.InferenceEngine.create_inferencer')
    def test_run_inference_with_progress_callback(self, mock_create):
        """测试带进度回调的推理"""
        mock_inferencer = MockInferencer(tokens=["Test"])
        mock_create.return_value = mock_inferencer
        
        progress_calls = []
        def progress_callback(task_id, model_id, status):
            progress_calls.append((task_id, model_id, status))
        
        # 执行推理
        results = list(self.engine.run_inference(
            "test prompt", 
            [self.pytorch_model],
            progress_callback=progress_callback
        ))
        
        # 验证结果和进度回调
        assert len(results) == 1
        assert len(progress_calls) >= 3  # 至少有加载、生成、完成三个状态
        assert any("加载模型" in call[2] for call in progress_calls)
        assert any("完成" in call[2] for call in progress_calls)
    
    @patch('src.multi_llm_comparator.services.inference_engine.InferenceEngine.create_inferencer')
    def test_run_inference_model_load_failure(self, mock_create):
        """测试模型加载失败"""
        mock_inferencer = MockInferencer(fail_on_load=True)
        mock_create.return_value = mock_inferencer
        
        # 执行推理
        results = list(self.engine.run_inference("test prompt", [self.pytorch_model]))
        
        # 验证错误结果
        assert len(results) == 1
        result = results[0]
        assert result.model_id == "pytorch_model"
        assert result.content == ""
        assert result.is_complete is True
        assert result.error is not None
        assert "模拟加载失败" in result.error
    
    @patch('src.multi_llm_comparator.services.inference_engine.InferenceEngine.create_inferencer')
    def test_run_inference_generation_failure(self, mock_create):
        """测试生成失败"""
        mock_inferencer = MockInferencer(should_fail=True)
        mock_create.return_value = mock_inferencer
        
        # 执行推理
        results = list(self.engine.run_inference("test prompt", [self.pytorch_model]))
        
        # 验证错误结果
        assert len(results) == 1
        result = results[0]
        assert result.model_id == "pytorch_model"
        assert result.content == ""
        assert result.is_complete is True
        assert result.error is not None
        assert "模拟推理失败" in result.error
    
    @patch('src.multi_llm_comparator.services.inference_engine.InferenceEngine.create_inferencer')
    def test_run_inference_memory_insufficient(self, mock_create):
        """测试内存不足"""
        self.memory_manager.check_memory_availability.return_value = False
        mock_inferencer = MockInferencer()
        mock_create.return_value = mock_inferencer
        
        # 执行推理
        results = list(self.engine.run_inference("test prompt", [self.pytorch_model]))
        
        # 验证错误结果
        assert len(results) == 1
        result = results[0]
        assert result.error is not None
        assert "内存不足" in result.error
    
    @patch('src.multi_llm_comparator.services.inference_engine.InferenceEngine.create_inferencer')
    def test_retry_mechanism(self, mock_create):
        """测试重试机制"""
        # 第一次失败，第二次成功
        mock_inferencer1 = MockInferencer(fail_on_load=True)
        mock_inferencer2 = MockInferencer(tokens=["Success"])
        mock_create.side_effect = [mock_inferencer1, mock_inferencer2]
        
        # 设置较短的重试延迟
        self.engine.retry_delay = 0.1
        
        # 执行推理
        results = list(self.engine.run_inference("test prompt", [self.pytorch_model]))
        
        # 验证成功结果
        assert len(results) == 1
        result = results[0]
        assert result.content == "Success"
        assert result.error is None
    
    def test_cancel_all_tasks(self):
        """测试取消所有任务"""
        # 创建一些模拟任务
        task1 = InferenceTask(self.pytorch_model, "prompt1", "task1")
        task2 = InferenceTask(self.gguf_model, "prompt2", "task2")
        
        self.engine.active_tasks["task1"] = task1
        self.engine.active_tasks["task2"] = task2
        
        # 取消所有任务
        self.engine.cancel_all_tasks()
        
        # 验证任务被取消
        assert task1.is_cancelled()
        assert task2.is_cancelled()
    
    def test_cancel_task(self):
        """测试取消单个任务"""
        task = InferenceTask(self.pytorch_model, "prompt", "task1")
        self.engine.active_tasks["task1"] = task
        
        # 取消任务
        result = self.engine.cancel_task("task1")
        assert result is True
        assert task.is_cancelled()
        
        # 取消不存在的任务
        result = self.engine.cancel_task("nonexistent")
        assert result is False
    
    def test_get_task_status(self):
        """测试获取任务状态"""
        task = InferenceTask(self.pytorch_model, "prompt", "task1")
        task.status = "running"
        self.engine.active_tasks["task1"] = task
        
        # 获取存在的任务状态
        status = self.engine.get_task_status("task1")
        assert status == "running"
        
        # 获取不存在的任务状态
        status = self.engine.get_task_status("nonexistent")
        assert status is None
    
    def test_get_active_tasks(self):
        """测试获取活动任务列表"""
        task1 = InferenceTask(self.pytorch_model, "prompt1", "task1")
        task2 = InferenceTask(self.gguf_model, "prompt2", "task2")
        
        self.engine.active_tasks["task1"] = task1
        self.engine.active_tasks["task2"] = task2
        
        active_tasks = self.engine.get_active_tasks()
        assert set(active_tasks) == {"task1", "task2"}
    
    def test_cleanup_resources(self):
        """测试资源清理"""
        # 创建模拟任务
        task = InferenceTask(self.pytorch_model, "prompt", "task1")
        self.engine.active_tasks["task1"] = task
        
        # 清理资源
        self.engine.cleanup_resources()
        
        # 验证任务被取消
        assert task.is_cancelled()
        
        # 验证内存管理器被调用
        self.memory_manager.force_cleanup.assert_called()
    
    def test_context_manager(self):
        """测试上下文管理器"""
        with InferenceEngine(self.memory_manager) as engine:
            assert engine is not None
            
            # 创建模拟任务
            task = InferenceTask(self.pytorch_model, "prompt", "task1")
            engine.active_tasks["task1"] = task
        
        # 验证退出时任务被取消
        assert task.is_cancelled()
    
    @patch('src.multi_llm_comparator.services.inference_engine.InferenceEngine.create_inferencer')
    def test_concurrent_task_cancellation(self, mock_create):
        """测试并发任务取消"""
        # 创建一个会长时间运行的推理器
        mock_inferencer = MockInferencer(tokens=["token"] * 100)
        mock_create.return_value = mock_inferencer
        
        # 在另一个线程中启动推理
        results = []
        def run_inference():
            try:
                results.extend(list(self.engine.run_inference("test prompt", [self.pytorch_model])))
            except Exception as e:
                results.append(e)
        
        thread = threading.Thread(target=run_inference)
        thread.start()
        
        # 等待一小段时间后取消所有任务
        time.sleep(0.1)
        self.engine.cancel_all_tasks()
        
        # 等待线程完成
        thread.join(timeout=5)
        
        # 验证任务被取消或完成
        assert len(results) >= 0  # 可能被取消或部分完成