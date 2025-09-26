"""
GGUF推理器测试

测试GGUF推理器的功能。
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from src.multi_llm_comparator.inferencers.gguf_inferencer import GGUFInferencer
from src.multi_llm_comparator.core.exceptions import ModelLoadError, InferenceError, MemoryError


class TestGGUFInferencer:
    """GGUF推理器测试类"""
    
    def setup_method(self):
        """测试前准备"""
        self.inferencer = GGUFInferencer()
        self.test_config = {
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 40,
            'max_tokens': 512,
            'repeat_penalty': 1.1,
            'n_ctx': 2048,
            'n_threads': 4,
            'use_gpu': True
        }
    
    def test_init(self):
        """测试初始化"""
        assert self.inferencer.llm is None
        assert self.inferencer.model_path == ""
        assert self.inferencer.config == {}
        assert not self.inferencer.is_loaded
        assert self.inferencer._last_stats is None
    
    @patch('src.multi_llm_comparator.inferencers.gguf_inferencer.Llama')
    def test_load_model_success(self, mock_llama):
        """测试成功加载模型"""
        # 准备模拟对象
        mock_llm_instance = Mock()
        mock_llama.return_value = mock_llm_instance
        
        # 创建临时文件模拟模型文件
        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # 执行加载
            self.inferencer.load_model(temp_path, self.test_config)
            
            # 验证结果
            assert self.inferencer.is_loaded
            assert self.inferencer.model_path == temp_path
            assert self.inferencer.config == self.test_config
            assert self.inferencer.llm == mock_llm_instance
            
            # 验证Llama构造函数调用
            mock_llama.assert_called_once_with(
                model_path=temp_path,
                n_ctx=2048,
                n_threads=4,
                n_gpu_layers=-1,  # use_gpu=True时应该是-1
                verbose=False,
                use_mmap=True,
                use_mlock=False,
            )
        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @patch('src.multi_llm_comparator.inferencers.gguf_inferencer.Llama')
    def test_load_model_cpu_only(self, mock_llama):
        """测试仅CPU模式加载模型"""
        mock_llm_instance = Mock()
        mock_llama.return_value = mock_llm_instance
        
        config = self.test_config.copy()
        config['use_gpu'] = False
        
        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            self.inferencer.load_model(temp_path, config)
            
            # 验证GPU层数设置为0
            mock_llama.assert_called_once()
            call_args = mock_llama.call_args[1]
            assert call_args['n_gpu_layers'] == 0
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_load_model_file_not_found(self):
        """测试加载不存在的模型文件"""
        non_existent_path = "/path/to/non/existent/model.gguf"
        
        with pytest.raises(ModelLoadError) as exc_info:
            self.inferencer.load_model(non_existent_path, self.test_config)
        
        assert "模型文件未找到" in str(exc_info.value)
        assert not self.inferencer.is_loaded
    
    @patch('src.multi_llm_comparator.inferencers.gguf_inferencer.Llama')
    def test_load_model_exception(self, mock_llama):
        """测试加载模型时发生异常"""
        mock_llama.side_effect = Exception("模拟加载错误")
        
        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            with pytest.raises(ModelLoadError) as exc_info:
                self.inferencer.load_model(temp_path, self.test_config)
            
            assert "GGUF模型加载失败" in str(exc_info.value)
            assert not self.inferencer.is_loaded
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @patch('src.multi_llm_comparator.inferencers.gguf_inferencer.Llama')
    def test_generate_stream_success(self, mock_llama):
        """测试成功的流式生成"""
        # 准备模拟对象
        mock_llm_instance = Mock()
        mock_llama.return_value = mock_llm_instance
        
        # 模拟流式输出
        mock_stream_output = [
            {'choices': [{'text': 'Hello', 'finish_reason': None}]},
            {'choices': [{'text': ' world', 'finish_reason': None}]},
            {'choices': [{'text': '!', 'finish_reason': 'stop'}]},
        ]
        mock_llm_instance.return_value = mock_stream_output
        
        # 先加载模型
        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            self.inferencer.load_model(temp_path, self.test_config)
            
            # 执行流式生成
            prompt = "Test prompt"
            result_chunks = list(self.inferencer.generate_stream(prompt))
            
            # 验证结果
            assert result_chunks == ['Hello', ' world', '!']
            
            # 验证统计信息
            stats = self.inferencer.get_inference_stats()
            assert stats is not None
            assert stats.token_count == 3
            assert stats.start_time > 0
            assert stats.end_time > stats.start_time
            assert stats.tokens_per_second > 0
            
            # 验证llm调用参数
            mock_llm_instance.assert_called_once_with(
                prompt,
                max_tokens=512,
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                repeat_penalty=1.1,
                stream=True,
                stop=["</s>", "<|endoftext|>", "<|im_end|>"],
                echo=False,
            )
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_generate_stream_model_not_loaded(self):
        """测试模型未加载时的流式生成"""
        with pytest.raises(InferenceError) as exc_info:
            list(self.inferencer.generate_stream("test prompt"))
        
        assert "模型未加载" in str(exc_info.value)
    
    @patch('src.multi_llm_comparator.inferencers.gguf_inferencer.Llama')
    def test_generate_stream_exception(self, mock_llama):
        """测试流式生成时发生异常"""
        mock_llm_instance = Mock()
        mock_llama.return_value = mock_llm_instance
        mock_llm_instance.side_effect = Exception("模拟推理错误")
        
        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            self.inferencer.load_model(temp_path, self.test_config)
            
            with pytest.raises(InferenceError) as exc_info:
                list(self.inferencer.generate_stream("test prompt"))
            
            assert "启动流式生成失败" in str(exc_info.value)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @patch('src.multi_llm_comparator.inferencers.gguf_inferencer.Llama')
    def test_unload_model_success(self, mock_llama):
        """测试成功卸载模型"""
        mock_llm_instance = Mock()
        mock_llama.return_value = mock_llm_instance
        
        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # 先加载模型
            self.inferencer.load_model(temp_path, self.test_config)
            assert self.inferencer.is_loaded
            
            # 卸载模型
            self.inferencer.unload_model()
            
            # 验证卸载结果
            assert not self.inferencer.is_loaded
            assert self.inferencer.llm is None
            assert self.inferencer.model_path == ""
            assert self.inferencer.config == {}
            assert self.inferencer._last_stats is None
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_unload_model_not_loaded(self):
        """测试卸载未加载的模型"""
        # 应该不会抛出异常
        self.inferencer.unload_model()
        assert not self.inferencer.is_loaded
    
    @patch('src.multi_llm_comparator.inferencers.gguf_inferencer.Llama')
    def test_get_model_info_loaded(self, mock_llama):
        """测试获取已加载模型的信息"""
        mock_llm_instance = Mock()
        mock_llm_instance.n_ctx.return_value = 2048
        mock_llm_instance.n_vocab.return_value = 32000
        mock_llama.return_value = mock_llm_instance
        
        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            self.inferencer.load_model(temp_path, self.test_config)
            
            info = self.inferencer.get_model_info()
            
            assert info['type'] == 'GGUF'
            assert info['path'] == temp_path
            assert info['is_loaded'] is True
            assert info['config'] == self.test_config
            assert info['n_ctx'] == 2048
            assert info['n_vocab'] == 32000
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_get_model_info_not_loaded(self):
        """测试获取未加载模型的信息"""
        info = self.inferencer.get_model_info()
        
        assert info['type'] == 'GGUF'
        assert info['path'] == ""
        assert info['is_loaded'] is False
        assert info['config'] == {}
        assert 'n_ctx' not in info
        assert 'n_vocab' not in info
    
    def test_validate_config_valid(self):
        """测试有效配置的验证"""
        errors = self.inferencer.validate_config(self.test_config)
        assert errors == {}
    
    def test_validate_config_invalid_n_ctx(self):
        """测试无效n_ctx的验证"""
        config = self.test_config.copy()
        config['n_ctx'] = 0
        
        errors = self.inferencer.validate_config(config)
        assert 'n_ctx' in errors
        assert "上下文长度必须是1-32768之间的整数" in errors['n_ctx']
    
    def test_validate_config_invalid_n_threads(self):
        """测试无效n_threads的验证"""
        config = self.test_config.copy()
        config['n_threads'] = 0
        
        errors = self.inferencer.validate_config(config)
        assert 'n_threads' in errors
        assert "线程数必须是正整数" in errors['n_threads']
    
    def test_validate_config_invalid_top_k(self):
        """测试无效top_k的验证"""
        config = self.test_config.copy()
        config['top_k'] = 0
        
        errors = self.inferencer.validate_config(config)
        assert 'top_k' in errors
        assert "top_k必须是1-200之间的整数" in errors['top_k']
    
    def test_validate_config_invalid_repeat_penalty(self):
        """测试无效repeat_penalty的验证"""
        config = self.test_config.copy()
        config['repeat_penalty'] = 0.05
        
        errors = self.inferencer.validate_config(config)
        assert 'repeat_penalty' in errors
        assert "重复惩罚必须是0.1-2.0之间的数值" in errors['repeat_penalty']
    
    def test_validate_config_multiple_errors(self):
        """测试多个无效参数的验证"""
        config = {
            'n_ctx': 0,
            'top_k': 300,
            'repeat_penalty': 3.0,
        }
        
        errors = self.inferencer.validate_config(config)
        assert len(errors) == 3
        assert 'n_ctx' in errors
        assert 'top_k' in errors
        assert 'repeat_penalty' in errors
    
    @patch('src.multi_llm_comparator.inferencers.gguf_inferencer.Llama')
    def test_generate_stream_with_parameter_validation(self, mock_llama):
        """测试流式生成时的参数验证和调整"""
        mock_llm_instance = Mock()
        mock_llama.return_value = mock_llm_instance
        
        # 模拟流式输出
        mock_stream_output = [
            {'choices': [{'text': 'Test', 'finish_reason': 'stop'}]},
        ]
        mock_llm_instance.return_value = mock_stream_output
        
        # 使用极端参数值
        extreme_config = {
            'temperature': 5.0,  # 超出范围，应该被调整为2.0
            'top_p': 1.5,        # 超出范围，应该被调整为1.0
            'top_k': 500,        # 超出范围，应该被调整为200
            'repeat_penalty': 0.05,  # 超出范围，应该被调整为0.1
            'max_tokens': 100,
        }
        
        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            self.inferencer.load_model(temp_path, extreme_config)
            
            # 执行流式生成
            result_chunks = list(self.inferencer.generate_stream("test"))
            assert result_chunks == ['Test']
            
            # 验证参数被正确调整
            call_args = mock_llm_instance.call_args[1]
            assert call_args['temperature'] == 2.0
            assert call_args['top_p'] == 1.0
            assert call_args['top_k'] == 200
            assert call_args['repeat_penalty'] == 0.1
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @patch('src.multi_llm_comparator.inferencers.gguf_inferencer.Llama')
    def test_generate_stream_with_error_in_output(self, mock_llama):
        """测试流式生成时处理输出错误"""
        mock_llm_instance = Mock()
        mock_llama.return_value = mock_llm_instance
        
        # 模拟包含错误的输出
        mock_stream_output = [
            {'choices': [{'text': 'Hello', 'finish_reason': None}]},
            {'error': 'Some error occurred'},  # 错误输出
            {'choices': [{'text': ' world', 'finish_reason': 'stop'}]},
        ]
        mock_llm_instance.return_value = mock_stream_output
        
        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            self.inferencer.load_model(temp_path, self.test_config)
            
            # 执行流式生成，应该在遇到错误时抛出异常
            with pytest.raises(InferenceError) as exc_info:
                list(self.inferencer.generate_stream("test"))
            
            assert "推理过程中出现错误" in str(exc_info.value)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @patch('src.multi_llm_comparator.inferencers.gguf_inferencer.Llama')
    def test_generate_stream_with_malformed_output(self, mock_llama):
        """测试流式生成时处理格式错误的输出"""
        mock_llm_instance = Mock()
        mock_llama.return_value = mock_llm_instance
        
        # 模拟格式错误的输出
        mock_stream_output = [
            {'choices': [{'text': 'Hello', 'finish_reason': None}]},
            "invalid_output",  # 非字典格式
            {'choices': []},   # 空choices
            {'choices': [{'text': ' world', 'finish_reason': 'stop'}]},
        ]
        mock_llm_instance.return_value = mock_stream_output
        
        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            self.inferencer.load_model(temp_path, self.test_config)
            
            # 执行流式生成，应该跳过错误格式的输出
            result_chunks = list(self.inferencer.generate_stream("test"))
            assert result_chunks == ['Hello', ' world']
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @patch('src.multi_llm_comparator.inferencers.gguf_inferencer.Llama')
    def test_estimate_memory_usage(self, mock_llama):
        """测试内存使用估算"""
        mock_llm_instance = Mock()
        mock_llama.return_value = mock_llm_instance
        
        # 测试未加载模型时
        memory_info = self.inferencer.estimate_memory_usage()
        assert memory_info['model_loaded'] is False
        assert memory_info['estimated_vram_mb'] == 0
        assert memory_info['estimated_ram_mb'] == 0
        
        # 测试加载模型后
        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as temp_file:
            temp_file.write(b'0' * 1024 * 1024)  # 写入1MB数据
            temp_path = temp_file.name
        
        try:
            self.inferencer.load_model(temp_path, self.test_config)
            
            memory_info = self.inferencer.estimate_memory_usage()
            assert memory_info['model_loaded'] is True
            assert memory_info['model_file_size_mb'] == 1.0
            assert memory_info['estimated_vram_mb'] == 1.2  # use_gpu=True
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_cancel_inference(self):
        """测试取消推理"""
        # 这个方法主要用于接口一致性，不会抛出异常
        self.inferencer.cancel_inference()


class TestGGUFInferencerWithoutLlamaCpp:
    """测试没有llama-cpp-python库时的情况"""
    
    @patch('src.multi_llm_comparator.inferencers.gguf_inferencer.Llama', None)
    def test_init_without_llama_cpp(self):
        """测试没有llama-cpp-python库时的初始化"""
        with pytest.raises(ModelLoadError) as exc_info:
            GGUFInferencer()
        
        assert "llama-cpp-python库未安装" in str(exc_info.value)