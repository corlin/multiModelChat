# 开发指南

本文档为开发者提供详细的开发环境设置、开发流程和最佳实践指导。

## 开发环境设置

### 系统要求

- **操作系统**: Windows 10+, macOS 10.15+, Ubuntu 18.04+
- **Python**: 3.12或更高版本
- **内存**: 最少8GB RAM，推荐16GB+
- **存储**: 至少10GB可用空间
- **GPU**: 可选，NVIDIA GPU支持CUDA 11.8+

### 开发工具推荐

#### 代码编辑器
- **VS Code** (推荐)
  - Python扩展
  - Pylance扩展
  - Black Formatter扩展
  - Ruff扩展
- **PyCharm Professional**
- **Vim/Neovim** (高级用户)

#### 版本控制
- **Git** 2.30+
- **GitHub CLI** (可选)

### 环境配置步骤

#### 1. 安装Python 3.12

**Windows**:
```powershell
# 使用winget
winget install Python.Python.3.12

# 或从官网下载安装包
# https://www.python.org/downloads/windows/
```

**macOS**:
```bash
# 使用Homebrew
brew install python@3.12

# 或使用pyenv
pyenv install 3.12.0
pyenv global 3.12.0
```

**Linux (Ubuntu/Debian)**:
```bash
# 添加deadsnakes PPA
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev

# 设置默认Python版本
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1
```

#### 2. 安装uv包管理器

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows PowerShell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 或使用pip
pip install uv

# 验证安装
uv --version
```

#### 3. 克隆项目

```bash
# 克隆主仓库
git clone https://github.com/your-org/multi-llm-comparator.git
cd multi-llm-comparator

# 或克隆你的fork
git clone https://github.com/your-username/multi-llm-comparator.git
cd multi-llm-comparator

# 添加上游仓库
git remote add upstream https://github.com/your-org/multi-llm-comparator.git
```

#### 4. 设置开发环境

```bash
# 创建虚拟环境并安装依赖
uv sync --dev

# 安装pre-commit钩子
uv run pre-commit install

# 验证环境
uv run python --version
uv run pytest --version
uv run streamlit --version
```

#### 5. 配置IDE

**VS Code配置** (`.vscode/settings.json`):
```json
{
    "python.defaultInterpreterPath": ".venv/bin/python",
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"],
    "files.exclude": {
        "**/__pycache__": true,
        "**/.pytest_cache": true,
        "**/node_modules": true
    }
}
```

**PyCharm配置**:
1. 打开项目目录
2. 设置Python解释器为 `.venv/bin/python`
3. 配置代码格式化工具为Black
4. 启用Ruff代码检查
5. 配置测试运行器为pytest

## 开发工作流

### 分支管理策略

我们使用Git Flow工作流的简化版本：

```
main (生产分支)
├── develop (开发分支)
│   ├── feature/model-discovery (功能分支)
│   ├── feature/gguf-support (功能分支)
│   └── bugfix/memory-leak (修复分支)
└── hotfix/critical-bug (热修复分支)
```

### 开发流程

#### 1. 开始新功能开发

```bash
# 切换到develop分支并更新
git checkout develop
git pull upstream develop

# 创建功能分支
git checkout -b feature/your-feature-name

# 或者修复bug
git checkout -b bugfix/issue-description
```

#### 2. 开发过程

```bash
# 编写代码
# 运行测试确保功能正常
uv run pytest tests/

# 检查代码质量
uv run black src/ tests/
uv run ruff check src/ tests/
uv run mypy src/

# 运行应用测试
uv run streamlit run src/multi_llm_comparator/main.py
```

#### 3. 提交代码

```bash
# 添加文件
git add .

# 提交（遵循Conventional Commits规范）
git commit -m "feat: add GGUF model support"

# 推送到远程分支
git push origin feature/your-feature-name
```

#### 4. 创建Pull Request

1. 在GitHub上创建Pull Request
2. 填写PR模板
3. 请求代码审查
4. 根据反馈修改代码
5. 等待合并

### 代码质量检查

#### 自动化检查

项目配置了pre-commit钩子，每次提交时自动运行：

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.9.1
    hooks:
      - id: black
        language_version: python3.12

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.0
    hooks:
      - id: ruff
        args: [--fix]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.6.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

#### 手动检查

```bash
# 代码格式化
uv run black src/ tests/

# 代码检查
uv run ruff check src/ tests/ --fix

# 类型检查
uv run mypy src/

# 安全检查
uv run bandit -r src/

# 复杂度检查
uv run radon cc src/ -a
```

## 测试开发

### 测试结构

```
tests/
├── unit/                   # 单元测试
│   ├── core/
│   │   ├── test_models.py
│   │   └── test_config.py
│   ├── services/
│   │   ├── test_model_manager.py
│   │   └── test_inference_engine.py
│   └── inferencers/
│       ├── test_pytorch_inferencer.py
│       └── test_gguf_inferencer.py
├── integration/            # 集成测试
│   ├── test_end_to_end.py
│   └── test_ui_integration.py
├── performance/            # 性能测试
│   ├── test_memory_usage.py
│   └── test_inference_speed.py
├── fixtures/               # 测试数据
│   ├── models/
│   │   ├── fake_pytorch_model.bin
│   │   └── fake_gguf_model.gguf
│   └── configs/
│       └── test_config.json
└── conftest.py            # pytest配置
```

### 编写测试

#### 单元测试示例

```python
# tests/unit/services/test_model_manager.py
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from multi_llm_comparator.services.model_manager import ModelManager
from multi_llm_comparator.core.models import ModelInfo, ModelType

class TestModelManager:
    """模型管理器测试类。"""
    
    @pytest.fixture
    def model_manager(self):
        """创建模型管理器实例。"""
        return ModelManager()
    
    @pytest.fixture
    def temp_models_dir(self, tmp_path):
        """创建临时模型目录。"""
        pytorch_dir = tmp_path / "pytorch"
        gguf_dir = tmp_path / "gguf"
        pytorch_dir.mkdir()
        gguf_dir.mkdir()
        
        # 创建假模型文件
        (pytorch_dir / "model1.bin").write_text("fake pytorch model")
        (gguf_dir / "model2.gguf").write_text("fake gguf model")
        
        return tmp_path
    
    def test_scan_models_finds_pytorch_models(self, model_manager, temp_models_dir):
        """测试扫描PyTorch模型。"""
        models = model_manager.scan_models(str(temp_models_dir))
        
        pytorch_models = [m for m in models if m.model_type == ModelType.PYTORCH]
        assert len(pytorch_models) == 1
        assert pytorch_models[0].name == "model1"
    
    def test_scan_models_finds_gguf_models(self, model_manager, temp_models_dir):
        """测试扫描GGUF模型。"""
        models = model_manager.scan_models(str(temp_models_dir))
        
        gguf_models = [m for m in models if m.model_type == ModelType.GGUF]
        assert len(gguf_models) == 1
        assert gguf_models[0].name == "model2"
    
    def test_select_models_success(self, model_manager, temp_models_dir):
        """测试成功选择模型。"""
        models = model_manager.scan_models(str(temp_models_dir))
        model_ids = [m.id for m in models[:2]]
        
        result = model_manager.select_models(model_ids)
        assert result is True
        assert len(model_manager.get_selected_models()) == 2
    
    def test_select_too_many_models_fails(self, model_manager):
        """测试选择过多模型失败。"""
        model_ids = ["id1", "id2", "id3", "id4", "id5"]  # 超过4个
        
        with pytest.raises(ValueError, match="Cannot select more than 4 models"):
            model_manager.select_models(model_ids)
    
    @patch('os.path.getsize')
    def test_model_size_calculation(self, mock_getsize, model_manager, temp_models_dir):
        """测试模型大小计算。"""
        mock_getsize.return_value = 1024 * 1024 * 100  # 100MB
        
        models = model_manager.scan_models(str(temp_models_dir))
        assert all(m.size == 1024 * 1024 * 100 for m in models)
```

#### 集成测试示例

```python
# tests/integration/test_end_to_end.py
import pytest
from unittest.mock import Mock

from multi_llm_comparator.services.model_manager import ModelManager
from multi_llm_comparator.services.inference_engine import InferenceEngine
from multi_llm_comparator.core.models import ModelConfig

class TestEndToEnd:
    """端到端测试。"""
    
    @pytest.fixture
    def setup_environment(self, tmp_path):
        """设置测试环境。"""
        # 创建模型目录和文件
        models_dir = tmp_path / "models"
        pytorch_dir = models_dir / "pytorch"
        pytorch_dir.mkdir(parents=True)
        
        # 创建假模型文件
        model_file = pytorch_dir / "test_model.bin"
        model_file.write_text("fake model data")
        
        return {
            "models_dir": str(models_dir),
            "model_file": str(model_file)
        }
    
    def test_complete_inference_workflow(self, setup_environment):
        """测试完整的推理工作流。"""
        # 初始化组件
        model_manager = ModelManager(setup_environment["models_dir"])
        inference_engine = InferenceEngine()
        
        # 扫描模型
        models = model_manager.scan_models()
        assert len(models) > 0
        
        # 选择模型
        model_manager.select_models([models[0].id])
        selected_models = model_manager.get_selected_models()
        assert len(selected_models) == 1
        
        # 配置参数
        configs = {
            models[0].id: ModelConfig(temperature=0.7, max_tokens=100)
        }
        
        # 模拟推理过程（使用Mock避免实际加载模型）
        with patch.object(inference_engine, 'create_inferencer') as mock_create:
            mock_inferencer = Mock()
            mock_inferencer.generate_stream.return_value = iter(["Hello", " World", "!"])
            mock_create.return_value = mock_inferencer
            
            # 执行推理
            prompt = "Test prompt"
            results = list(inference_engine.run_inference(prompt, selected_models, configs))
            
            # 验证结果
            assert len(results) > 0
            final_result = results[-1]
            assert models[0].id in final_result
            assert final_result[models[0].id].content == "Hello World!"
```

#### 性能测试示例

```python
# tests/performance/test_memory_usage.py
import pytest
import psutil
import time
from unittest.mock import Mock

from multi_llm_comparator.services.memory_manager import MemoryManager

class TestMemoryPerformance:
    """内存性能测试。"""
    
    def test_memory_cleanup_effectiveness(self):
        """测试内存清理效果。"""
        memory_manager = MemoryManager()
        
        # 记录初始内存使用
        initial_memory = psutil.virtual_memory().used
        
        # 模拟内存使用
        large_objects = []
        for i in range(100):
            large_objects.append([0] * 10000)  # 创建大对象
        
        # 记录峰值内存使用
        peak_memory = psutil.virtual_memory().used
        
        # 清理内存
        del large_objects
        memory_manager.cleanup()
        time.sleep(1)  # 等待垃圾回收
        
        # 记录清理后内存使用
        final_memory = psutil.virtual_memory().used
        
        # 验证内存清理效果
        memory_increase = peak_memory - initial_memory
        memory_recovered = peak_memory - final_memory
        recovery_rate = memory_recovered / memory_increase
        
        assert recovery_rate > 0.8, f"Memory recovery rate too low: {recovery_rate:.2%}"
    
    @pytest.mark.slow
    def test_inference_memory_stability(self):
        """测试推理过程内存稳定性。"""
        # 这个测试需要较长时间运行，标记为slow
        memory_manager = MemoryManager()
        memory_readings = []
        
        # 模拟多次推理过程
        for i in range(10):
            # 模拟模型加载和推理
            mock_model = Mock()
            memory_manager.track_model_loading(f"model_{i}", mock_model)
            
            # 记录内存使用
            memory_readings.append(psutil.virtual_memory().used)
            
            # 模拟推理完成，清理资源
            memory_manager.cleanup_model(f"model_{i}")
            memory_manager.cleanup()
            time.sleep(0.5)
        
        # 分析内存趋势
        memory_trend = [memory_readings[i] - memory_readings[0] for i in range(len(memory_readings))]
        
        # 验证没有明显的内存泄漏
        final_increase = memory_trend[-1]
        max_increase = max(memory_trend)
        
        assert final_increase < max_increase * 0.5, "Potential memory leak detected"
```

### 运行测试

```bash
# 运行所有测试
uv run pytest

# 运行特定测试文件
uv run pytest tests/unit/services/test_model_manager.py

# 运行特定测试方法
uv run pytest tests/unit/services/test_model_manager.py::TestModelManager::test_scan_models_success

# 运行带标记的测试
uv run pytest -m "not slow"  # 跳过慢测试
uv run pytest -m "integration"  # 只运行集成测试

# 生成覆盖率报告
uv run pytest --cov=src/multi_llm_comparator --cov-report=html --cov-report=term

# 并行运行测试
uv run pytest -n auto  # 需要安装pytest-xdist

# 详细输出
uv run pytest -v -s
```

## 调试技巧

### 1. 使用调试器

**VS Code调试配置** (`.vscode/launch.json`):
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug Streamlit App",
            "type": "python",
            "request": "launch",
            "module": "streamlit",
            "args": ["run", "src/multi_llm_comparator/main.py"],
            "console": "integratedTerminal",
            "justMyCode": false
        },
        {
            "name": "Debug Tests",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": ["tests/", "-v"],
            "console": "integratedTerminal",
            "justMyCode": false
        }
    ]
}
```

**PyCharm调试**:
1. 设置断点
2. 右键选择"Debug"
3. 使用调试控制台检查变量

### 2. 日志调试

```python
import logging
import structlog

# 配置结构化日志
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# 使用日志
logger = structlog.get_logger()

def debug_function():
    logger.info("Function started", function="debug_function")
    try:
        # 业务逻辑
        result = complex_operation()
        logger.info("Operation completed", result=result)
        return result
    except Exception as e:
        logger.error("Operation failed", error=str(e), exc_info=True)
        raise
```

### 3. 性能分析

```python
import cProfile
import pstats
from functools import wraps

def profile(func):
    """性能分析装饰器。"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()
        
        stats = pstats.Stats(pr)
        stats.sort_stats('cumulative')
        stats.print_stats(10)  # 显示前10个最耗时的函数
        
        return result
    return wrapper

@profile
def slow_function():
    # 需要分析的函数
    pass
```

### 4. 内存分析

```python
import tracemalloc
import psutil

def memory_usage_analysis():
    """内存使用分析。"""
    # 开始跟踪内存分配
    tracemalloc.start()
    
    # 记录初始内存
    initial_memory = psutil.Process().memory_info().rss
    
    # 执行操作
    result = memory_intensive_operation()
    
    # 获取内存快照
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    
    print("Top 10 memory allocations:")
    for stat in top_stats[:10]:
        print(stat)
    
    # 记录最终内存
    final_memory = psutil.Process().memory_info().rss
    print(f"Memory increase: {(final_memory - initial_memory) / 1024 / 1024:.2f} MB")
    
    return result
```

## 最佳实践

### 1. 代码组织

```python
# 好的做法：清晰的模块结构
from multi_llm_comparator.core.models import ModelInfo
from multi_llm_comparator.services.model_manager import ModelManager
from multi_llm_comparator.inferencers.base import BaseInferencer

# 避免：循环导入
# from multi_llm_comparator.services import *  # 不好

# 好的做法：使用类型注解
def process_models(models: List[ModelInfo]) -> Dict[str, Any]:
    """处理模型列表。"""
    return {"count": len(models)}

# 好的做法：使用数据类
@dataclass
class ProcessingResult:
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
```

### 2. 错误处理

```python
# 好的做法：具体的异常处理
try:
    model = load_model(model_path)
except FileNotFoundError:
    logger.error("Model file not found", path=model_path)
    raise ModelNotFoundError(f"Model file not found: {model_path}")
except MemoryError:
    logger.error("Insufficient memory to load model", path=model_path)
    raise InsufficientMemoryError("Not enough memory to load model")
except Exception as e:
    logger.error("Unexpected error loading model", error=str(e), path=model_path)
    raise ModelLoadError(f"Failed to load model: {e}") from e

# 避免：捕获所有异常
# try:
#     model = load_model(model_path)
# except Exception:
#     pass  # 不好：忽略所有错误
```

### 3. 资源管理

```python
# 好的做法：使用上下文管理器
class ModelInferencer:
    def __enter__(self):
        self.load_model()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload_model()
        self.cleanup_resources()

# 使用
with ModelInferencer() as inferencer:
    result = inferencer.generate(prompt)
# 自动清理资源

# 好的做法：显式资源清理
def inference_with_cleanup(model_path: str, prompt: str) -> str:
    inferencer = None
    try:
        inferencer = create_inferencer(model_path)
        return inferencer.generate(prompt)
    finally:
        if inferencer:
            inferencer.cleanup()
```

### 4. 测试策略

```python
# 好的做法：使用fixture和参数化测试
@pytest.fixture
def sample_models():
    return [
        ModelInfo(id="1", name="model1", path="/path/1", model_type=ModelType.PYTORCH, size=1000, config={}),
        ModelInfo(id="2", name="model2", path="/path/2", model_type=ModelType.GGUF, size=2000, config={})
    ]

@pytest.mark.parametrize("temperature,expected_range", [
    (0.1, (0.05, 0.15)),
    (0.7, (0.6, 0.8)),
    (1.5, (1.4, 1.6))
])
def test_temperature_validation(temperature, expected_range):
    config = ModelConfig(temperature=temperature)
    assert expected_range[0] <= config.temperature <= expected_range[1]

# 好的做法：模拟外部依赖
@patch('torch.load')
def test_model_loading_failure(mock_torch_load):
    mock_torch_load.side_effect = RuntimeError("CUDA out of memory")
    
    with pytest.raises(ModelLoadError):
        load_pytorch_model("/fake/path")
```

## 常见问题解决

### 1. 依赖冲突

```bash
# 清理环境重新安装
uv cache clean
rm -rf .venv
uv sync

# 检查依赖冲突
uv tree
```

### 2. 测试失败

```bash
# 详细输出查看错误
uv run pytest -v -s --tb=long

# 只运行失败的测试
uv run pytest --lf

# 停在第一个失败的测试
uv run pytest -x
```

### 3. 性能问题

```bash
# 性能分析
uv run python -m cProfile -o profile.stats your_script.py

# 查看分析结果
uv run python -c "
import pstats
stats = pstats.Stats('profile.stats')
stats.sort_stats('cumulative')
stats.print_stats(20)
"
```

### 4. 内存问题

```python
# 内存监控
import psutil
import gc

def monitor_memory():
    process = psutil.Process()
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    print(f"Memory percent: {process.memory_percent():.2f}%")
    
    # 强制垃圾回收
    gc.collect()
```

这个开发指南提供了完整的开发环境设置和最佳实践，帮助开发者快速上手项目开发。