"""
模型文件扫描器的单元测试
"""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.multi_llm_comparator.services.model_scanner import ModelFileScanner, ScanResult
from src.multi_llm_comparator.core.models import ModelType, ModelInfo


class TestModelFileScanner:
    """模型文件扫描器测试类"""
    
    def setup_method(self):
        """测试方法设置"""
        self.scanner = ModelFileScanner()
    
    def test_supported_extensions(self):
        """测试支持的文件扩展名"""
        expected_extensions = {'.bin', '.pt', '.pth', '.safetensors', '.gguf'}
        assert self.scanner.get_supported_extensions() == expected_extensions
    
    def test_is_supported_file(self):
        """测试文件格式支持检查"""
        # 支持的格式
        assert self.scanner.is_supported_file("model.bin") is True
        assert self.scanner.is_supported_file("model.pt") is True
        assert self.scanner.is_supported_file("model.pth") is True
        assert self.scanner.is_supported_file("model.safetensors") is True
        assert self.scanner.is_supported_file("model.gguf") is True
        
        # 不支持的格式
        assert self.scanner.is_supported_file("model.txt") is False
        assert self.scanner.is_supported_file("model.json") is False
        assert self.scanner.is_supported_file("model.py") is False
        
        # 大小写不敏感
        assert self.scanner.is_supported_file("MODEL.BIN") is True
        assert self.scanner.is_supported_file("Model.Pt") is True
    
    def test_scan_nonexistent_directory(self):
        """测试扫描不存在的目录"""
        result = self.scanner.scan_directory("/nonexistent/directory")
        
        assert isinstance(result, ScanResult)
        assert len(result.models) == 0
        assert len(result.errors) == 1
        assert "目录不存在" in result.errors[0]
        assert result.scanned_files == 0
        assert result.valid_models == 0
    
    def test_scan_file_instead_of_directory(self):
        """测试扫描文件而不是目录"""
        with tempfile.NamedTemporaryFile() as temp_file:
            result = self.scanner.scan_directory(temp_file.name)
            
            assert isinstance(result, ScanResult)
            assert len(result.models) == 0
            assert len(result.errors) == 1
            assert "路径不是目录" in result.errors[0]
    
    def test_scan_empty_directory(self):
        """测试扫描空目录"""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self.scanner.scan_directory(temp_dir)
            
            assert isinstance(result, ScanResult)
            assert len(result.models) == 0
            assert len(result.errors) == 0
            assert result.scanned_files == 0
            assert result.valid_models == 0
    
    def test_scan_directory_with_model_files(self):
        """测试扫描包含模型文件的目录"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 创建测试文件
            (temp_path / "model1.bin").write_text("fake pytorch model")
            (temp_path / "model2.gguf").write_text("fake gguf model")
            (temp_path / "model3.safetensors").write_text("fake safetensors model")
            (temp_path / "readme.txt").write_text("not a model file")
            
            result = self.scanner.scan_directory(temp_dir)
            
            assert isinstance(result, ScanResult)
            assert len(result.models) == 3
            assert len(result.errors) == 0
            assert result.scanned_files == 4  # 包括txt文件
            assert result.valid_models == 3
            
            # 检查模型类型
            model_types = {model.model_type for model in result.models}
            assert ModelType.PYTORCH in model_types
            assert ModelType.GGUF in model_types
    
    def test_scan_directory_recursive(self):
        """测试递归扫描子目录"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 创建子目录和文件
            sub_dir = temp_path / "subdir"
            sub_dir.mkdir()
            
            (temp_path / "model1.bin").write_text("model in root")
            (sub_dir / "model2.pt").write_text("model in subdir")
            
            # 递归扫描
            result = self.scanner.scan_directory(temp_dir, recursive=True)
            assert len(result.models) == 2
            
            # 非递归扫描
            result = self.scanner.scan_directory(temp_dir, recursive=False)
            assert len(result.models) == 1
    
    def test_analyze_pytorch_file(self):
        """测试分析PyTorch模型文件"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            model_file = temp_path / "test_model.bin"
            model_file.write_text("fake pytorch model content")
            
            model_info = self.scanner._analyze_file(model_file)
            
            assert model_info is not None
            assert isinstance(model_info, ModelInfo)
            assert model_info.name == "test_model"
            assert model_info.model_type == ModelType.PYTORCH
            assert model_info.size > 0
            assert model_info.path == str(model_file)
            assert 'framework' in model_info.config
            assert model_info.config['framework'] == 'pytorch'
    
    def test_analyze_gguf_file(self):
        """测试分析GGUF模型文件"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            model_file = temp_path / "test_model_q4_0.gguf"
            model_file.write_text("fake gguf model content")
            
            model_info = self.scanner._analyze_file(model_file)
            
            assert model_info is not None
            assert isinstance(model_info, ModelInfo)
            assert model_info.name == "test_model_q4_0"
            assert model_info.model_type == ModelType.GGUF
            assert model_info.size > 0
            assert 'framework' in model_info.config
            assert model_info.config['framework'] == 'gguf'
            assert model_info.config['quantization'] == 'Q4_0'
    
    def test_analyze_safetensors_file(self):
        """测试分析safetensors文件"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            model_file = temp_path / "model.safetensors"
            model_file.write_text("fake safetensors content")
            
            model_info = self.scanner._analyze_file(model_file)
            
            assert model_info is not None
            assert model_info.model_type == ModelType.PYTORCH
            assert model_info.config['format'] == 'safetensors'
            assert model_info.config['memory_efficient'] is True
    
    def test_analyze_unsupported_file(self):
        """测试分析不支持的文件格式"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            text_file = temp_path / "readme.txt"
            text_file.write_text("not a model file")
            
            model_info = self.scanner._analyze_file(text_file)
            assert model_info is None
    
    def test_pytorch_metadata_collection(self):
        """测试PyTorch模型元数据收集"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 测试普通PyTorch文件
            pt_file = temp_path / "model.pt"
            pt_file.write_text("content")
            metadata = self.scanner._get_pytorch_metadata(pt_file)
            
            assert metadata['framework'] == 'pytorch'
            assert metadata['format'] == 'pytorch'
            assert metadata['supports_streaming'] is True
            assert metadata['requires_tokenizer'] is True
            assert metadata['memory_efficient'] is False
            
            # 测试safetensors文件
            safetensors_file = temp_path / "model.safetensors"
            safetensors_file.write_text("content")
            metadata = self.scanner._get_pytorch_metadata(safetensors_file)
            
            assert metadata['format'] == 'safetensors'
            assert metadata['memory_efficient'] is True
    
    def test_gguf_metadata_collection(self):
        """测试GGUF模型元数据收集"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 测试不同量化格式的GGUF文件
            test_cases = [
                ("model_q4_0.gguf", "Q4_0"),
                ("model_q4_1.gguf", "Q4_1"),
                ("model_q5_0.gguf", "Q5_0"),
                ("model_q5_1.gguf", "Q5_1"),
                ("model_q8_0.gguf", "Q8_0"),
                ("model_f16.gguf", "F16"),
                ("model_f32.gguf", "F32"),
                ("model.gguf", "unknown"),
            ]
            
            for filename, expected_quantization in test_cases:
                gguf_file = temp_path / filename
                gguf_file.write_text("content")
                metadata = self.scanner._get_gguf_metadata(gguf_file)
                
                assert metadata['framework'] == 'gguf'
                assert metadata['format'] == 'gguf'
                assert metadata['supports_streaming'] is True
                assert metadata['requires_tokenizer'] is False
                assert metadata['cpu_optimized'] is True
                assert metadata['quantization'] == expected_quantization
    
    def test_scan_with_file_access_error(self):
        """测试文件访问错误的处理"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            model_file = temp_path / "model.bin"
            model_file.write_text("content")
            
            # 模拟文件分析时的错误，而不是目录访问错误
            with patch.object(self.scanner, '_analyze_file', side_effect=PermissionError("Access denied")):
                result = self.scanner.scan_directory(temp_dir)
                
                # 应该有错误记录，但不应该崩溃
                assert len(result.errors) > 0
    
    def test_model_id_generation(self):
        """测试模型ID生成的唯一性"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 创建同名但在不同目录的文件
            sub_dir1 = temp_path / "dir1"
            sub_dir2 = temp_path / "dir2"
            sub_dir1.mkdir()
            sub_dir2.mkdir()
            
            (sub_dir1 / "model.bin").write_text("content1")
            (sub_dir2 / "model.bin").write_text("content2")
            
            result = self.scanner.scan_directory(temp_dir)
            
            assert len(result.models) == 2
            # 确保ID是唯一的
            model_ids = {model.id for model in result.models}
            assert len(model_ids) == 2
            
            # 确保ID包含路径信息以保证唯一性
            for model in result.models:
                assert "dir1" in model.id or "dir2" in model.id


if __name__ == "__main__":
    pytest.main([__file__])