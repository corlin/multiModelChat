"""
推理器测试

测试不同类型推理器的功能。
"""

import pytest
from unittest.mock import Mock, patch

from src.multi_llm_comparator.inferencers.base import BaseInferencer
from src.multi_llm_comparator.inferencers.pytorch_inferencer import PyTorchInferencer
from src.multi_llm_comparator.inferencers.gguf_inferencer import GGUFInferencer


class TestBaseInferencer:
    """基础推理器测试类"""
    
    def test_abstract_methods(self):
        """测试抽象方法"""
        # BaseInferencer是抽象类，不能直接实例化
        with pytest.raises(TypeError):
            BaseInferencer()


class TestPyTorchInferencer:
    """PyTorch推理器测试类"""
    
    def setup_method(self):
        """测试前的设置"""
        self.inferencer = PyTorchInferencer()
    
    def test_init(self):
        """测试初始化"""
        assert self.inferencer.model is None
        assert self.inferencer.tokenizer is None
        assert self.inferencer.is_loaded is False
    
    def test_unload_model(self):
        """测试卸载模型"""
        self.inferencer.unload_model()
        assert self.inferencer.model is None
        assert self.inferencer.tokenizer is None
        assert self.inferencer.is_loaded is False
    
    def test_get_model_info(self):
        """测试获取模型信息"""
        info = self.inferencer.get_model_info()
        assert info["type"] == "pytorch"
        assert info["is_loaded"] is False
    
    def test_load_model_failure(self):
        """测试加载模型失败"""
        from src.multi_llm_comparator.core.exceptions import ModelLoadError
        with pytest.raises(ModelLoadError):
            self.inferencer.load_model("invalid_path", {})
    
    def test_generate_stream_not_loaded(self):
        """测试未加载模型时的流式生成"""
        from src.multi_llm_comparator.core.exceptions import InferenceError
        with pytest.raises(InferenceError, match="模型未加载"):
            list(self.inferencer.generate_stream("prompt"))


class TestGGUFInferencer:
    """GGUF推理器测试类"""
    
    def setup_method(self):
        """测试前的设置"""
        self.inferencer = GGUFInferencer()
    
    def test_init(self):
        """测试初始化"""
        assert self.inferencer.model is None
        assert self.inferencer.llm is None
        assert self.inferencer.is_loaded is False
    
    def test_unload_model(self):
        """测试卸载模型"""
        self.inferencer.unload_model()
        assert self.inferencer.llm is None
        assert self.inferencer.is_loaded is False
    
    def test_get_model_info(self):
        """测试获取模型信息"""
        info = self.inferencer.get_model_info()
        assert info["type"] == "GGUF"
        assert info["is_loaded"] is False
    
    def test_load_model_failure(self):
        """测试加载模型失败"""
        from src.multi_llm_comparator.core.exceptions import ModelLoadError
        with pytest.raises(ModelLoadError):
            self.inferencer.load_model("invalid_path", {})
    
    def test_generate_stream_not_loaded(self):
        """测试未加载模型时的流式生成"""
        from src.multi_llm_comparator.core.exceptions import InferenceError
        with pytest.raises(InferenceError, match="模型未加载"):
            list(self.inferencer.generate_stream("prompt"))