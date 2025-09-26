"""
PyTorch推理器单元测试
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from threading import Thread
import time

from src.multi_llm_comparator.inferencers.pytorch_inferencer import PyTorchInferencer
from src.multi_llm_comparator.core.exceptions import ModelLoadError, InferenceError


class TestPyTorchInferencer:
    """PyTorch推理器测试类"""
    
    def setup_method(self):
        """测试前的设置"""
        self.inferencer = PyTorchInferencer()
        self.test_config = {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 100,
            "do_sample": True,
            "torch_dtype": "auto",
            "low_cpu_mem_usage": True,
        }
    
    def test_init(self):
        """测试初始化"""
        assert self.inferencer.model is None
        assert self.inferencer.tokenizer is None
        assert self.inferencer.device is None
        assert not self.inferencer.is_loaded
    
    def test_detect_device_cuda(self):
        """测试CUDA设备检测"""
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.get_device_name', return_value='Tesla V100'):
            device = self.inferencer._detect_device()
            assert device == "cuda"
    
    def test_detect_device_mps(self):
        """测试MPS设备检测"""
        with patch('torch.cuda.is_available', return_value=False), \
             patch('torch.backends.mps.is_available', return_value=True):
            device = self.inferencer._detect_device()
            assert device == "mps"
    
    def test_detect_device_cpu(self):
        """测试CPU设备检测"""
        with patch('torch.cuda.is_available', return_value=False), \
             patch('torch.backends.mps.is_available', return_value=False):
            device = self.inferencer._detect_device()
            assert device == "cpu"
    
    def test_get_torch_dtype(self):
        """测试torch数据类型获取"""
        self.inferencer.device = "cuda"
        
        # 测试auto类型
        dtype = self.inferencer._get_torch_dtype("auto")
        assert dtype == torch.float16
        
        # 测试CPU设备的auto类型
        self.inferencer.device = "cpu"
        dtype = self.inferencer._get_torch_dtype("auto")
        assert dtype == torch.float32
        
        # 测试指定类型
        dtype = self.inferencer._get_torch_dtype("float16")
        assert dtype == torch.float16
        
        dtype = self.inferencer._get_torch_dtype("bfloat16")
        assert dtype == torch.bfloat16
    
    @patch('src.multi_llm_comparator.inferencers.pytorch_inferencer.AutoTokenizer')
    @patch('src.multi_llm_comparator.inferencers.pytorch_inferencer.AutoModelForCausalLM')
    def test_load_model_success(self, mock_model_class, mock_tokenizer_class):
        """测试成功加载模型"""
        # 设置mock对象
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer.pad_token_id = 1
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model.to.return_value = mock_model  # 确保to()方法返回模型本身
        mock_model_class.from_pretrained.return_value = mock_model
        
        with patch.object(self.inferencer, '_detect_device', return_value='cpu'):
            self.inferencer.load_model("/path/to/model", self.test_config)
        
        # 验证调用
        mock_tokenizer_class.from_pretrained.assert_called_once()
        mock_model_class.from_pretrained.assert_called_once()
        mock_model.eval.assert_called_once()
        mock_model.to.assert_called_once_with('cpu')
        
        # 验证状态
        assert self.inferencer.is_loaded
        assert self.inferencer.model == mock_model
        assert self.inferencer.tokenizer == mock_tokenizer
        assert mock_tokenizer.pad_token == "</s>"
    
    @patch('src.multi_llm_comparator.inferencers.pytorch_inferencer.AutoTokenizer')
    def test_load_model_tokenizer_failure(self, mock_tokenizer_class):
        """测试tokenizer加载失败"""
        mock_tokenizer_class.from_pretrained.side_effect = Exception("Tokenizer load failed")
        
        with pytest.raises(ModelLoadError, match="无法加载PyTorch模型"):
            self.inferencer.load_model("/path/to/model", self.test_config)
        
        assert not self.inferencer.is_loaded
    
    @patch('src.multi_llm_comparator.inferencers.pytorch_inferencer.AutoTokenizer')
    @patch('src.multi_llm_comparator.inferencers.pytorch_inferencer.AutoModelForCausalLM')
    def test_load_model_model_failure(self, mock_model_class, mock_tokenizer_class):
        """测试模型加载失败"""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "[PAD]"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model_class.from_pretrained.side_effect = Exception("Model load failed")
        
        with pytest.raises(ModelLoadError, match="无法加载PyTorch模型"):
            self.inferencer.load_model("/path/to/model", self.test_config)
        
        assert not self.inferencer.is_loaded
    
    def test_generate_stream_not_loaded(self):
        """测试未加载模型时的流式生成"""
        with pytest.raises(InferenceError, match="模型未加载"):
            list(self.inferencer.generate_stream("test prompt"))
    
    @patch('src.multi_llm_comparator.inferencers.pytorch_inferencer.TextIteratorStreamer')
    @patch('src.multi_llm_comparator.inferencers.pytorch_inferencer.GenerationConfig')
    @patch('src.multi_llm_comparator.inferencers.pytorch_inferencer.Thread')
    def test_generate_stream_success(self, mock_thread_class, mock_gen_config, mock_streamer_class):
        """测试成功的流式生成"""
        # 设置模型为已加载状态
        self.inferencer.is_loaded = True
        self.inferencer.model = Mock()
        self.inferencer.tokenizer = Mock()
        self.inferencer.device = "cpu"
        self.inferencer.config = self.test_config
        
        # 设置tokenizer mock
        mock_tensor = Mock()
        mock_tensor.to.return_value = mock_tensor
        mock_tensor.shape = [1, 10]  # 模拟tensor形状
        self.inferencer.tokenizer.encode.return_value = mock_tensor
        self.inferencer.tokenizer.pad_token_id = 0
        self.inferencer.tokenizer.eos_token_id = 1
        
        # 设置streamer mock
        mock_streamer = Mock()
        mock_streamer.__iter__ = Mock(return_value=iter(["Hello", " world", "!"]))
        mock_streamer_class.return_value = mock_streamer
        
        # 设置thread mock
        mock_thread = Mock()
        mock_thread_class.return_value = mock_thread
        
        # 模拟tokenizer.encode对新文本的调用
        def mock_encode_side_effect(text, **kwargs):
            if text == "test prompt":
                return mock_tensor
            else:
                # 对于新文本，返回简单的token列表
                return [1, 2]
        
        self.inferencer.tokenizer.encode.side_effect = mock_encode_side_effect
        
        # 执行流式生成
        result = list(self.inferencer.generate_stream("test prompt"))
        
        # 验证结果
        assert result == ["Hello", " world", "!"]
        
        # 验证调用
        self.inferencer.tokenizer.encode.assert_called()
        mock_streamer_class.assert_called_once()
        mock_thread_class.assert_called_once()
        mock_thread.start.assert_called_once()
        mock_thread.join.assert_called_once()
    
    @patch('src.multi_llm_comparator.inferencers.pytorch_inferencer.TextIteratorStreamer')
    def test_generate_stream_streamer_error(self, mock_streamer_class):
        """测试流式生成过程中的错误"""
        # 设置模型为已加载状态
        self.inferencer.is_loaded = True
        self.inferencer.model = Mock()
        self.inferencer.tokenizer = Mock()
        self.inferencer.device = "cpu"
        self.inferencer.config = self.test_config
        
        # 设置tokenizer mock
        mock_tensor = Mock()
        mock_tensor.to.return_value = mock_tensor
        mock_tensor.shape = [1, 10]  # 模拟tensor形状
        self.inferencer.tokenizer.encode.return_value = mock_tensor
        self.inferencer.tokenizer.pad_token_id = 0
        self.inferencer.tokenizer.eos_token_id = 1
        
        # 设置streamer抛出异常
        mock_streamer = Mock()
        mock_streamer.__iter__ = Mock(side_effect=Exception("Streamer error"))
        mock_streamer_class.return_value = mock_streamer
        
        with pytest.raises(InferenceError, match="流式生成失败"):
            list(self.inferencer.generate_stream("test prompt"))
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.empty_cache')
    @patch('torch.cuda.synchronize')
    def test_unload_model_with_cuda(self, mock_sync, mock_empty_cache, mock_cuda_available):
        """测试在CUDA环境下卸载模型"""
        # 设置模型为已加载状态
        self.inferencer.is_loaded = True
        self.inferencer.model = Mock()
        self.inferencer.tokenizer = Mock()
        
        self.inferencer.unload_model()
        
        # 验证CUDA缓存清理
        mock_empty_cache.assert_called_once()
        mock_sync.assert_called_once()
        
        # 验证状态
        assert not self.inferencer.is_loaded
        assert self.inferencer.model is None
        assert self.inferencer.tokenizer is None
    
    @patch('torch.backends.mps.is_available', return_value=True)
    @patch('torch.mps.empty_cache')
    def test_unload_model_with_mps(self, mock_mps_empty_cache, mock_mps_available):
        """测试在MPS环境下卸载模型"""
        # 设置模型为已加载状态
        self.inferencer.is_loaded = True
        self.inferencer.model = Mock()
        self.inferencer.tokenizer = Mock()
        
        with patch('torch.cuda.is_available', return_value=False):
            self.inferencer.unload_model()
        
        # 验证MPS缓存清理
        mock_mps_empty_cache.assert_called_once()
        
        # 验证状态
        assert not self.inferencer.is_loaded
    
    def test_cleanup_resources(self):
        """测试资源清理"""
        # 设置一些资源
        self.inferencer.model = Mock()
        self.inferencer.tokenizer = Mock()
        self.inferencer.device = "cuda"
        self.inferencer.model_path = "/path/to/model"
        self.inferencer.config = {"test": "config"}
        
        self.inferencer._cleanup_resources()
        
        # 验证资源被清理
        assert self.inferencer.model is None
        assert self.inferencer.tokenizer is None
        assert self.inferencer.device is None
        assert self.inferencer.model_path is None
        assert self.inferencer.config is None
    
    def test_get_model_info_not_loaded(self):
        """测试获取未加载模型的信息"""
        info = self.inferencer.get_model_info()
        
        expected = {
            "type": "pytorch",
            "is_loaded": False,
            "model_path": None,
            "device": None,
        }
        
        assert info == expected
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.memory_allocated', return_value=1024)
    @patch('torch.cuda.memory_reserved', return_value=2048)
    def test_get_model_info_loaded(self, mock_reserved, mock_allocated, mock_cuda_available):
        """测试获取已加载模型的信息"""
        # 设置模型为已加载状态
        self.inferencer.is_loaded = True
        self.inferencer.model_path = "/path/to/model"
        self.inferencer.device = "cuda"
        
        # 创建mock模型
        mock_model = Mock()
        mock_param = Mock()
        mock_param.numel.return_value = 1000
        mock_model.parameters.return_value = [mock_param, mock_param]  # 2个参数
        self.inferencer.model = mock_model
        
        info = self.inferencer.get_model_info()
        
        # 验证信息
        assert info["type"] == "pytorch"
        assert info["is_loaded"] is True
        assert info["model_path"] == "/path/to/model"
        assert info["device"] == "cuda"
        assert info["parameter_count"] == 2000  # 2 * 1000
        assert info["gpu_memory_allocated"] == 1024
        assert info["gpu_memory_reserved"] == 2048
    
    def test_get_model_info_error_handling(self):
        """测试获取模型信息时的错误处理"""
        # 设置模型为已加载状态但有问题的模型
        self.inferencer.is_loaded = True
        self.inferencer.model_path = "/path/to/model"
        self.inferencer.device = "cpu"
        
        mock_model = Mock()
        mock_model.parameters.side_effect = Exception("Parameter error")
        self.inferencer.model = mock_model
        
        # 应该不抛出异常，只记录警告
        info = self.inferencer.get_model_info()
        
        # 基本信息应该存在
        assert info["type"] == "pytorch"
        assert info["is_loaded"] is True
        assert "parameter_count" not in info  # 错误时不应包含这个字段
    
    def test_get_inference_stats_no_inference(self):
        """测试未进行推理时获取统计信息"""
        stats = self.inferencer.get_inference_stats()
        assert stats is None
    
    @patch('src.multi_llm_comparator.inferencers.pytorch_inferencer.TextIteratorStreamer')
    @patch('src.multi_llm_comparator.inferencers.pytorch_inferencer.GenerationConfig')
    @patch('src.multi_llm_comparator.inferencers.pytorch_inferencer.Thread')
    def test_inference_stats_collection(self, mock_thread_class, mock_gen_config, mock_streamer_class):
        """测试推理统计信息收集"""
        # 设置模型为已加载状态
        self.inferencer.is_loaded = True
        self.inferencer.model = Mock()
        self.inferencer.tokenizer = Mock()
        self.inferencer.device = "cpu"
        self.inferencer.config = self.test_config
        
        # 设置tokenizer mock
        mock_tensor = Mock()
        mock_tensor.to.return_value = mock_tensor
        mock_tensor.shape = [1, 10]  # 模拟输入长度
        
        def mock_encode_side_effect(text, **kwargs):
            if text == "test prompt":
                return mock_tensor
            else:
                return [1, 2]
        
        self.inferencer.tokenizer.encode.side_effect = mock_encode_side_effect
        self.inferencer.tokenizer.pad_token_id = 0
        self.inferencer.tokenizer.eos_token_id = 1
        
        # 设置streamer mock
        mock_streamer = Mock()
        mock_streamer.__iter__ = Mock(return_value=iter(["Hello", " world", "!"]))
        mock_streamer_class.return_value = mock_streamer
        
        # 设置thread mock
        mock_thread = Mock()
        mock_thread.is_alive.return_value = False
        mock_thread_class.return_value = mock_thread
        
        # 执行流式生成
        result = list(self.inferencer.generate_stream("test prompt"))
        
        # 验证结果
        assert result == ["Hello", " world", "!"]
        
        # 验证统计信息
        stats = self.inferencer.get_inference_stats()
        assert stats is not None
        assert stats.start_time > 0
        assert stats.end_time is not None
        assert stats.end_time >= stats.start_time
        assert stats.token_count == 3  # 3个文本片段
        assert stats.tokens_per_second is not None
        assert stats.tokens_per_second >= 0
    
    @patch('src.multi_llm_comparator.inferencers.pytorch_inferencer.TextIteratorStreamer')
    def test_generate_stream_keyboard_interrupt(self, mock_streamer_class):
        """测试生成过程中的键盘中断"""
        # 设置模型为已加载状态
        self.inferencer.is_loaded = True
        self.inferencer.model = Mock()
        self.inferencer.tokenizer = Mock()
        self.inferencer.device = "cpu"
        self.inferencer.config = self.test_config
        
        # 设置tokenizer mock
        mock_tensor = Mock()
        mock_tensor.to.return_value = mock_tensor
        mock_tensor.shape = [1, 10]
        self.inferencer.tokenizer.encode.return_value = mock_tensor
        self.inferencer.tokenizer.pad_token_id = 0
        self.inferencer.tokenizer.eos_token_id = 1
        
        # 设置streamer抛出KeyboardInterrupt
        mock_streamer = Mock()
        mock_streamer.__iter__ = Mock(side_effect=KeyboardInterrupt("User interrupt"))
        mock_streamer_class.return_value = mock_streamer
        
        with pytest.raises(InferenceError, match="生成过程被用户中断"):
            list(self.inferencer.generate_stream("test prompt"))
        
        # 验证统计信息仍然被设置
        stats = self.inferencer.get_inference_stats()
        assert stats is not None
        assert stats.end_time is not None