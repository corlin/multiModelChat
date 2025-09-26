"""
参数验证器测试

测试参数验证功能的正确性和边界条件。
"""

import pytest
from typing import Dict, Any

from src.multi_llm_comparator.core.validators import ParameterValidator, RealTimeValidator
from src.multi_llm_comparator.core.models import ModelConfig, ModelType


class TestParameterValidator:
    """参数验证器测试类"""
    
    def setup_method(self):
        """测试前的设置"""
        self.validator = ParameterValidator()
    
    def test_validate_temperature_valid_range(self):
        """测试temperature参数的有效范围"""
        # 有效值
        is_valid, error = self.validator.validate_parameter("temperature", 0.7)
        assert is_valid
        assert error is None
        
        is_valid, error = self.validator.validate_parameter("temperature", 0.0)
        assert is_valid
        assert error is None
        
        is_valid, error = self.validator.validate_parameter("temperature", 2.0)
        assert is_valid
        assert error is None
    
    def test_validate_temperature_invalid_range(self):
        """测试temperature参数的无效范围"""
        # 超出范围的值
        is_valid, error = self.validator.validate_parameter("temperature", -0.1)
        assert not is_valid
        assert "超出有效范围" in error
        
        is_valid, error = self.validator.validate_parameter("temperature", 2.1)
        assert not is_valid
        assert "超出有效范围" in error
    
    def test_validate_temperature_invalid_type(self):
        """测试temperature参数的类型验证"""
        is_valid, error = self.validator.validate_parameter("temperature", "0.7")
        assert not is_valid
        assert "类型错误" in error
    
    def test_validate_max_tokens_range(self):
        """测试max_tokens参数的范围验证"""
        # 有效值
        is_valid, error = self.validator.validate_parameter("max_tokens", 512)
        assert is_valid
        assert error is None
        
        # 边界值
        is_valid, error = self.validator.validate_parameter("max_tokens", 1)
        assert is_valid
        assert error is None
        
        is_valid, error = self.validator.validate_parameter("max_tokens", 8192)
        assert is_valid
        assert error is None
        
        # 无效值
        is_valid, error = self.validator.validate_parameter("max_tokens", 0)
        assert not is_valid
        assert "超出有效范围" in error
        
        is_valid, error = self.validator.validate_parameter("max_tokens", 10000)
        assert not is_valid
        assert "超出有效范围" in error
    
    def test_validate_top_p_range(self):
        """测试top_p参数的范围验证"""
        # 有效值
        is_valid, error = self.validator.validate_parameter("top_p", 0.9)
        assert is_valid
        assert error is None
        
        # 边界值
        is_valid, error = self.validator.validate_parameter("top_p", 0.0)
        assert is_valid
        assert error is None
        
        is_valid, error = self.validator.validate_parameter("top_p", 1.0)
        assert is_valid
        assert error is None
        
        # 无效值
        is_valid, error = self.validator.validate_parameter("top_p", -0.1)
        assert not is_valid
        assert "超出有效范围" in error
        
        is_valid, error = self.validator.validate_parameter("top_p", 1.1)
        assert not is_valid
        assert "超出有效范围" in error
    
    def test_validate_torch_dtype_special_values(self):
        """测试torch_dtype参数的特殊值验证"""
        # 有效值
        valid_dtypes = ["auto", "float16", "bfloat16", "float32", "int8", "int4"]
        for dtype in valid_dtypes:
            is_valid, error = self.validator.validate_parameter("torch_dtype", dtype)
            assert is_valid, f"torch_dtype '{dtype}' should be valid"
            assert error is None
        
        # 无效值
        is_valid, error = self.validator.validate_parameter("torch_dtype", "invalid_dtype")
        assert not is_valid
        assert "值无效" in error
    
    def test_validate_boolean_parameters(self):
        """测试布尔类型参数的验证"""
        bool_params = ["do_sample", "use_gpu", "low_cpu_mem_usage"]
        
        for param in bool_params:
            # 有效值
            is_valid, error = self.validator.validate_parameter(param, True)
            assert is_valid
            assert error is None
            
            is_valid, error = self.validator.validate_parameter(param, False)
            assert is_valid
            assert error is None
            
            # 无效值
            is_valid, error = self.validator.validate_parameter(param, "true")
            assert not is_valid
            assert "类型错误" in error
    
    def test_validate_optional_parameters(self):
        """测试可选参数的验证"""
        optional_params = ["pad_token_id", "eos_token_id", "n_threads"]
        
        for param in optional_params:
            # None值应该有效
            is_valid, error = self.validator.validate_parameter(param, None)
            assert is_valid
            assert error is None
            
            # 有效的非None值
            is_valid, error = self.validator.validate_parameter(param, 1)
            assert is_valid
            assert error is None
    
    def test_model_type_compatibility(self):
        """测试模型类型兼容性检查"""
        # PyTorch特定参数
        is_valid, error = self.validator.validate_parameter(
            "do_sample", True, ModelType.PYTORCH
        )
        assert is_valid
        assert error is None
        
        # PyTorch参数用于GGUF模型应该失败
        is_valid, error = self.validator.validate_parameter(
            "do_sample", True, ModelType.GGUF
        )
        assert not is_valid
        assert "不适用于" in error
        
        # GGUF特定参数
        is_valid, error = self.validator.validate_parameter(
            "top_k", 40, ModelType.GGUF
        )
        assert is_valid
        assert error is None
        
        # GGUF参数用于PyTorch模型应该失败
        is_valid, error = self.validator.validate_parameter(
            "top_k", 40, ModelType.PYTORCH
        )
        assert not is_valid
        assert "不适用于" in error
    
    def test_validate_config_complete(self):
        """测试完整配置的验证"""
        # 有效的PyTorch配置
        pytorch_config = ModelConfig(
            temperature=0.7,
            max_tokens=512,
            top_p=0.9,
            do_sample=True,
            torch_dtype="float16"
        )
        
        is_valid, errors = self.validator.validate_config(pytorch_config, ModelType.PYTORCH)
        if not is_valid:
            print(f"Validation errors: {errors}")
        assert is_valid
        assert len(errors) == 0
        
        # 有效的GGUF配置
        gguf_config = ModelConfig(
            temperature=0.8,
            max_tokens=1024,
            top_p=0.95,
            top_k=50,
            repeat_penalty=1.1,
            n_ctx=2048,
            use_gpu=True
        )
        
        is_valid, errors = self.validator.validate_config(gguf_config, ModelType.GGUF)
        assert is_valid
        assert len(errors) == 0
    
    def test_validate_config_with_errors(self):
        """测试包含错误的配置验证"""
        # 包含错误的配置
        invalid_config = {
            "temperature": 3.0,  # 超出范围
            "max_tokens": 0,     # 超出范围
            "top_p": 1.5,        # 超出范围
            "torch_dtype": "invalid"  # 无效值
        }
        
        is_valid, errors = self.validator.validate_config(invalid_config)
        assert not is_valid
        assert len(errors) > 0
        
        # 检查错误消息
        error_text = " ".join(errors)
        assert "temperature" in error_text
        assert "max_tokens" in error_text
        assert "top_p" in error_text
        assert "torch_dtype" in error_text
    
    def test_parameter_compatibility_checks(self):
        """测试参数兼容性检查"""
        # temperature=0和top_p<1的组合
        config = {
            "temperature": 0.0,
            "top_p": 0.9
        }
        
        is_valid, errors = self.validator.validate_config(config)
        assert not is_valid
        assert any("temperature为0" in error for error in errors)
        
        # GGUF模型中max_tokens > n_ctx
        gguf_config = {
            "max_tokens": 4096,
            "n_ctx": 2048
        }
        
        is_valid, errors = self.validator.validate_config(gguf_config, ModelType.GGUF)
        assert not is_valid
        assert any("max_tokens" in error and "n_ctx" in error for error in errors)
    
    def test_get_parameter_suggestions(self):
        """测试参数建议功能"""
        # 超出范围的值
        suggestions = self.validator.get_parameter_suggestions("temperature", 3.0)
        assert len(suggestions) > 0
        assert any("不大于" in suggestion for suggestion in suggestions)
        
        # 低于范围的值
        suggestions = self.validator.get_parameter_suggestions("temperature", -0.5)
        assert len(suggestions) > 0
        assert any("不小于" in suggestion for suggestion in suggestions)
        
        # 无效的torch_dtype
        suggestions = self.validator.get_parameter_suggestions("torch_dtype", "invalid")
        assert len(suggestions) > 0
        assert any("建议将" in suggestion for suggestion in suggestions)
        
        # 不兼容的参数
        suggestions = self.validator.get_parameter_suggestions(
            "do_sample", True, ModelType.GGUF
        )
        assert len(suggestions) > 0
        assert any("不适用于" in suggestion for suggestion in suggestions)


class TestRealTimeValidator:
    """实时验证器测试类"""
    
    def setup_method(self):
        """测试前的设置"""
        self.rt_validator = RealTimeValidator()
    
    def test_validate_on_change_valid(self):
        """测试有效参数的实时验证"""
        result = self.rt_validator.validate_on_change("temperature", 0.7)
        
        assert result["is_valid"] is True
        assert result["error_message"] is None
        assert result["parameter"] == "temperature"
        assert result["value"] == 0.7
        assert result["model_type"] is None
        assert isinstance(result["suggestions"], list)
    
    def test_validate_on_change_invalid(self):
        """测试无效参数的实时验证"""
        result = self.rt_validator.validate_on_change("temperature", 3.0)
        
        assert result["is_valid"] is False
        assert result["error_message"] is not None
        assert "超出有效范围" in result["error_message"]
        assert result["parameter"] == "temperature"
        assert result["value"] == 3.0
        assert len(result["suggestions"]) > 0
    
    def test_validate_on_change_with_model_type(self):
        """测试带模型类型的实时验证"""
        # 兼容的参数
        result = self.rt_validator.validate_on_change(
            "do_sample", True, ModelType.PYTORCH
        )
        assert result["is_valid"] is True
        assert result["model_type"] == "pytorch"
        
        # 不兼容的参数
        result = self.rt_validator.validate_on_change(
            "do_sample", True, ModelType.GGUF
        )
        assert result["is_valid"] is False
        assert result["model_type"] == "gguf"
        assert "不适用于" in result["error_message"]
    
    def test_validation_caching(self):
        """测试验证结果缓存"""
        # 第一次验证
        result1 = self.rt_validator.validate_on_change("temperature", 0.7)
        
        # 第二次相同的验证应该使用缓存
        result2 = self.rt_validator.validate_on_change("temperature", 0.7)
        
        assert result1 == result2
        
        # 清除缓存
        self.rt_validator.clear_cache()
        
        # 缓存清除后应该重新验证
        result3 = self.rt_validator.validate_on_change("temperature", 0.7)
        assert result3 == result1  # 结果应该相同，但是重新计算的
    
    def test_different_values_different_cache(self):
        """测试不同值使用不同缓存"""
        result1 = self.rt_validator.validate_on_change("temperature", 0.7)
        result2 = self.rt_validator.validate_on_change("temperature", 0.8)
        
        assert result1["value"] != result2["value"]
        assert result1["is_valid"] is True
        assert result2["is_valid"] is True


class TestParameterValidatorEdgeCases:
    """参数验证器边界情况测试"""
    
    def setup_method(self):
        """测试前的设置"""
        self.validator = ParameterValidator()
    
    def test_unknown_parameter(self):
        """测试未知参数的处理"""
        is_valid, error = self.validator.validate_parameter("unknown_param", "value")
        assert is_valid  # 未知参数应该通过验证
        assert error is None
    
    def test_none_values_for_optional_params(self):
        """测试可选参数的None值"""
        optional_params = ["pad_token_id", "eos_token_id", "n_threads"]
        
        for param in optional_params:
            is_valid, error = self.validator.validate_parameter(param, None)
            assert is_valid
            assert error is None
    
    def test_boundary_values(self):
        """测试边界值"""
        # 测试各种参数的边界值
        boundary_tests = [
            ("temperature", 0.0, True),
            ("temperature", 2.0, True),
            ("max_tokens", 1, True),
            ("max_tokens", 8192, True),
            ("top_p", 0.0, True),
            ("top_p", 1.0, True),
            ("top_k", 1, True),
            ("top_k", 100, True),
        ]
        
        for param, value, expected in boundary_tests:
            is_valid, error = self.validator.validate_parameter(param, value)
            assert is_valid == expected, f"Parameter {param} with value {value} should be {expected}"
    
    def test_type_coercion_behavior(self):
        """测试类型强制转换行为"""
        # float参数接受int值
        is_valid, error = self.validator.validate_parameter("temperature", 1)  # int instead of float
        assert is_valid
        assert error is None
        
        # 但不接受字符串
        is_valid, error = self.validator.validate_parameter("temperature", "1.0")
        assert not is_valid
        assert "类型错误" in error
    
    def test_config_dict_validation(self):
        """测试字典格式配置的验证"""
        config_dict = {
            "temperature": 0.7,
            "max_tokens": 512,
            "top_p": 0.9,
            "do_sample": True
        }
        
        is_valid, errors = self.validator.validate_config(config_dict, ModelType.PYTORCH)
        assert is_valid
        assert len(errors) == 0
    
    def test_empty_config_validation(self):
        """测试空配置的验证"""
        empty_config = {}
        
        is_valid, errors = self.validator.validate_config(empty_config)
        assert is_valid  # 空配置应该有效
        assert len(errors) == 0


class TestConfigManagerIntegration:
    """配置管理器集成测试"""
    
    def setup_method(self):
        """测试前的设置"""
        import tempfile
        self.temp_dir = tempfile.mkdtemp()
        
        from src.multi_llm_comparator.core.config import ConfigManager
        self.config_manager = ConfigManager(self.temp_dir)
    
    def test_config_manager_validation(self):
        """测试配置管理器的验证功能"""
        from src.multi_llm_comparator.core.models import ModelConfig, ModelType
        from src.multi_llm_comparator.core.exceptions import ValidationError
        
        # 有效配置应该保存成功
        valid_config = ModelConfig(temperature=0.7, max_tokens=512)
        self.config_manager.save_model_config("test_model", valid_config, ModelType.PYTORCH)
        
        # 无效配置应该抛出异常
        invalid_config = ModelConfig(temperature=3.0)  # 超出范围
        
        with pytest.raises(ValidationError):
            self.config_manager.save_model_config("test_model", invalid_config, ModelType.PYTORCH)
    
    def test_config_manager_parameter_validation(self):
        """测试配置管理器的参数验证"""
        from src.multi_llm_comparator.core.models import ModelType
        
        # 有效参数
        is_valid, error = self.config_manager.validate_parameter("temperature", 0.7, ModelType.PYTORCH)
        assert is_valid
        assert error is None
        
        # 无效参数
        is_valid, error = self.config_manager.validate_parameter("temperature", 3.0, ModelType.PYTORCH)
        assert not is_valid
        assert error is not None
    
    def test_config_manager_suggestions(self):
        """测试配置管理器的建议功能"""
        from src.multi_llm_comparator.core.models import ModelType
        
        suggestions = self.config_manager.get_parameter_suggestions("temperature", 3.0, ModelType.PYTORCH)
        assert len(suggestions) > 0
        assert any("不大于" in suggestion for suggestion in suggestions)


if __name__ == "__main__":
    pytest.main([__file__])