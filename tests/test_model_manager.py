"""
模型管理器的单元测试
"""

import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.multi_llm_comparator.core.models import ModelInfo, ModelType
from src.multi_llm_comparator.core.exceptions import ModelSelectionError, ModelNotFoundError
from src.multi_llm_comparator.services.model_manager import ModelManager
from src.multi_llm_comparator.services.model_scanner import ScanResult


class TestModelManager:
    """模型管理器测试类"""
    
    def setup_method(self):
        """测试方法设置"""
        # 使用临时文件作为缓存文件
        self.temp_cache = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        self.temp_cache.close()
        self.manager = ModelManager(cache_file=self.temp_cache.name)
    
    def teardown_method(self):
        """测试方法清理"""
        # 清理临时文件
        Path(self.temp_cache.name).unlink(missing_ok=True)
    
    def test_model_manager_initialization(self):
        """测试模型管理器初始化"""
        assert self.manager is not None
        assert len(self.manager.get_available_models()) == 0
        assert len(self.manager.get_selected_models()) == 0
        assert self.manager.get_selection_count() == 0
        assert self.manager.can_select_more_models() is True
    
    def test_max_selection_limit(self):
        """测试最大选择数量限制"""
        assert self.manager.MAX_SELECTED_MODELS == 4
    
    def test_scan_models_with_mock(self):
        """测试模型扫描功能（使用mock）"""
        # 创建测试模型
        test_models = [
            ModelInfo(
                id="model1.bin",
                name="model1",
                path="/path/to/model1.bin",
                model_type=ModelType.PYTORCH,
                size=1000,
                config={}
            ),
            ModelInfo(
                id="model2.gguf",
                name="model2",
                path="/path/to/model2.gguf",
                model_type=ModelType.GGUF,
                size=2000,
                config={}
            )
        ]
        
        # Mock扫描结果
        mock_result = ScanResult(
            models=test_models,
            errors=[],
            scanned_files=2,
            valid_models=2
        )
        
        with patch.object(self.manager.scanner, 'scan_directory', return_value=mock_result):
            result = self.manager.scan_models(["/test/directory"])
            
            assert len(result.models) == 2
            assert result.valid_models == 2
            assert len(self.manager.get_available_models()) == 2
    
    def test_get_models_by_type(self):
        """测试按类型获取模型"""
        # 添加测试模型
        pytorch_model = ModelInfo(
            id="pytorch_model",
            name="PyTorch Model",
            path="/path/to/pytorch.bin",
            model_type=ModelType.PYTORCH,
            size=1000,
            config={}
        )
        
        gguf_model = ModelInfo(
            id="gguf_model",
            name="GGUF Model",
            path="/path/to/gguf.gguf",
            model_type=ModelType.GGUF,
            size=2000,
            config={}
        )
        
        self.manager._available_models["pytorch_model"] = pytorch_model
        self.manager._available_models["gguf_model"] = gguf_model
        
        pytorch_models = self.manager.get_models_by_type(ModelType.PYTORCH)
        gguf_models = self.manager.get_models_by_type(ModelType.GGUF)
        
        assert len(pytorch_models) == 1
        assert len(gguf_models) == 1
        assert pytorch_models[0].model_type == ModelType.PYTORCH
        assert gguf_models[0].model_type == ModelType.GGUF
    
    def test_model_selection_valid(self):
        """测试有效的模型选择"""
        # 添加测试模型
        test_models = {
            "model1": ModelInfo("model1", "Model 1", "/path1", ModelType.PYTORCH, 1000, {}),
            "model2": ModelInfo("model2", "Model 2", "/path2", ModelType.GGUF, 2000, {}),
        }
        
        self.manager._available_models.update(test_models)
        
        # 选择模型
        self.manager.select_models(["model1", "model2"])
        
        selected = self.manager.get_selected_models()
        selected_ids = self.manager.get_selected_model_ids()
        
        assert len(selected) == 2
        assert len(selected_ids) == 2
        assert "model1" in selected_ids
        assert "model2" in selected_ids
        assert self.manager.is_model_selected("model1") is True
        assert self.manager.is_model_selected("model2") is True
        assert self.manager.get_selection_count() == 2
    
    def test_model_selection_too_many(self):
        """测试选择过多模型的错误处理"""
        # 添加5个测试模型
        for i in range(5):
            model_id = f"model{i}"
            model = ModelInfo(model_id, f"Model {i}", f"/path{i}", ModelType.PYTORCH, 1000, {})
            self.manager._available_models[model_id] = model
        
        # 尝试选择5个模型（超过限制）
        with pytest.raises(ModelSelectionError) as exc_info:
            self.manager.select_models([f"model{i}" for i in range(5)])
        
        assert "最多只能选择 4 个模型" in str(exc_info.value)
    
    def test_model_selection_nonexistent(self):
        """测试选择不存在的模型"""
        with pytest.raises(ModelNotFoundError) as exc_info:
            self.manager.select_models(["nonexistent_model"])
        
        assert "以下模型不存在" in str(exc_info.value)
    
    def test_add_remove_selected_model(self):
        """测试添加和移除选中的模型"""
        # 添加测试模型
        test_model = ModelInfo("test_model", "Test Model", "/path", ModelType.PYTORCH, 1000, {})
        self.manager._available_models["test_model"] = test_model
        
        # 添加到选中列表
        self.manager.add_selected_model("test_model")
        assert self.manager.is_model_selected("test_model") is True
        assert self.manager.get_selection_count() == 1
        
        # 从选中列表移除
        self.manager.remove_selected_model("test_model")
        assert self.manager.is_model_selected("test_model") is False
        assert self.manager.get_selection_count() == 0
    
    def test_add_selected_model_limit(self):
        """测试添加模型时的数量限制"""
        # 添加4个测试模型并全部选中
        for i in range(4):
            model_id = f"model{i}"
            model = ModelInfo(model_id, f"Model {i}", f"/path{i}", ModelType.PYTORCH, 1000, {})
            self.manager._available_models[model_id] = model
            self.manager.add_selected_model(model_id)
        
        # 尝试添加第5个模型
        model5 = ModelInfo("model5", "Model 5", "/path5", ModelType.PYTORCH, 1000, {})
        self.manager._available_models["model5"] = model5
        
        with pytest.raises(ModelSelectionError) as exc_info:
            self.manager.add_selected_model("model5")
        
        assert "已达到最大选择数量限制" in str(exc_info.value)
        assert self.manager.can_select_more_models() is False
    
    def test_clear_selected_models(self):
        """测试清空选中的模型"""
        # 添加并选择测试模型
        test_model = ModelInfo("test_model", "Test Model", "/path", ModelType.PYTORCH, 1000, {})
        self.manager._available_models["test_model"] = test_model
        self.manager.add_selected_model("test_model")
        
        assert self.manager.get_selection_count() == 1
        
        # 清空选中列表
        self.manager.clear_selected_models()
        assert self.manager.get_selection_count() == 0
        assert self.manager.can_select_more_models() is True
    
    def test_get_model_by_id(self):
        """测试根据ID获取模型"""
        test_model = ModelInfo("test_model", "Test Model", "/path", ModelType.PYTORCH, 1000, {})
        self.manager._available_models["test_model"] = test_model
        
        # 存在的模型
        found_model = self.manager.get_model_by_id("test_model")
        assert found_model is not None
        assert found_model.id == "test_model"
        
        # 不存在的模型
        not_found = self.manager.get_model_by_id("nonexistent")
        assert not_found is None
    
    def test_get_model_statistics(self):
        """测试获取模型统计信息"""
        # 添加不同类型的模型
        pytorch_model = ModelInfo("pytorch", "PyTorch", "/path1", ModelType.PYTORCH, 1000, {})
        gguf_model1 = ModelInfo("gguf1", "GGUF 1", "/path2", ModelType.GGUF, 2000, {})
        gguf_model2 = ModelInfo("gguf2", "GGUF 2", "/path3", ModelType.GGUF, 3000, {})
        
        self.manager._available_models.update({
            "pytorch": pytorch_model,
            "gguf1": gguf_model1,
            "gguf2": gguf_model2,
        })
        
        # 选择一个模型
        self.manager.select_models(["pytorch"])
        
        stats = self.manager.get_model_statistics()
        
        assert stats['total_models'] == 3
        assert stats['pytorch_models'] == 1
        assert stats['gguf_models'] == 2
        assert stats['selected_models'] == 1
        assert stats['max_selection'] == 4
        assert stats['total_size_bytes'] == 6000
        assert abs(stats['total_size_mb'] - (6000 / (1024 * 1024))) < 0.01
    
    def test_cache_save_and_load(self):
        """测试缓存的保存和加载"""
        # 添加测试模型
        test_model = ModelInfo("test_model", "Test Model", "/path", ModelType.PYTORCH, 1000, {})
        self.manager._available_models["test_model"] = test_model
        self.manager.select_models(["test_model"])
        self.manager._last_scan_directories = ["/test/dir"]
        
        # 保存缓存
        self.manager._save_cache()
        
        # 创建新的管理器实例来测试加载
        new_manager = ModelManager(cache_file=self.temp_cache.name)
        
        # 验证缓存加载
        assert len(new_manager.get_available_models()) == 1
        assert len(new_manager.get_selected_models()) == 1
        assert new_manager.get_selected_models()[0].id == "test_model"
        assert new_manager._last_scan_directories == ["/test/dir"]
    
    def test_cleanup_invalid_selections(self):
        """测试清理无效选中模型"""
        # 添加模型并选中
        test_model = ModelInfo("test_model", "Test Model", "/path", ModelType.PYTORCH, 1000, {})
        self.manager._available_models["test_model"] = test_model
        self.manager._selected_model_ids.add("test_model")
        self.manager._selected_model_ids.add("invalid_model")  # 不存在的模型
        
        assert len(self.manager._selected_model_ids) == 2
        
        # 清理无效选择
        self.manager._cleanup_invalid_selections()
        
        assert len(self.manager._selected_model_ids) == 1
        assert "test_model" in self.manager._selected_model_ids
        assert "invalid_model" not in self.manager._selected_model_ids
    
    def test_refresh_models(self):
        """测试刷新模型列表"""
        mock_result = ScanResult(models=[], errors=[], scanned_files=0, valid_models=0)
        
        with patch.object(self.manager.scanner, 'scan_directory', return_value=mock_result) as mock_scan:
            # 第一次扫描
            self.manager.scan_models(["/test/dir"])
            assert mock_scan.call_count == 1
            
            # 再次扫描相同目录（应该使用缓存）
            self.manager.scan_models(["/test/dir"])
            assert mock_scan.call_count == 1  # 没有增加
            
            # 强制刷新
            self.manager.refresh_models(["/test/dir"])
            assert mock_scan.call_count == 2  # 增加了
    
    def test_clear_cache(self):
        """测试清空缓存"""
        # 保存一些数据到缓存
        test_model = ModelInfo("test_model", "Test Model", "/path", ModelType.PYTORCH, 1000, {})
        self.manager._available_models["test_model"] = test_model
        self.manager._save_cache()
        
        # 确认缓存文件存在
        assert Path(self.temp_cache.name).exists()
        
        # 清空缓存
        self.manager.clear_cache()
        
        # 确认缓存文件被删除
        assert not Path(self.temp_cache.name).exists()


if __name__ == "__main__":
    pytest.main([__file__])