# 开发者贡献指南

感谢您对多LLM模型比较器项目的关注！本文档将指导您如何为项目做出贡献。

## 开发环境设置

### 前置要求

- Python 3.12+
- uv包管理器
- Git
- 推荐：VS Code或PyCharm

### 环境配置

1. **Fork并克隆项目**
```bash
git clone https://github.com/your-username/multi-llm-comparator.git
cd multi-llm-comparator
```

2. **设置开发环境**
```bash
# 安装uv（如果尚未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建虚拟环境并安装依赖
uv sync --dev

# 安装pre-commit钩子
uv run pre-commit install
```

3. **验证安装**
```bash
# 运行测试
uv run pytest

# 检查代码格式
uv run black --check src/ tests/
uv run ruff check src/ tests/

# 启动应用
uv run streamlit run src/multi_llm_comparator/main.py
```

## 开发工作流

### 分支策略

- `main`: 主分支，包含稳定的生产代码
- `develop`: 开发分支，包含最新的开发代码
- `feature/*`: 功能分支，用于开发新功能
- `bugfix/*`: 修复分支，用于修复bug
- `hotfix/*`: 热修复分支，用于紧急修复

### 开发流程

1. **创建功能分支**
```bash
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name
```

2. **开发和测试**
```bash
# 编写代码
# 运行测试
uv run pytest tests/

# 检查代码质量
uv run black src/ tests/
uv run ruff check src/ tests/
uv run mypy src/
```

3. **提交代码**
```bash
git add .
git commit -m "feat: add your feature description"
```

4. **推送并创建PR**
```bash
git push origin feature/your-feature-name
# 在GitHub上创建Pull Request
```

## 代码规范

### Python代码风格

我们使用以下工具确保代码质量：

- **Black**: 代码格式化
- **Ruff**: 代码检查和导入排序
- **MyPy**: 类型检查
- **Pytest**: 单元测试

### 代码格式化

```bash
# 自动格式化代码
uv run black src/ tests/

# 检查格式
uv run black --check src/ tests/

# 修复导入和代码问题
uv run ruff check --fix src/ tests/
```

### 类型注解

所有新代码都应该包含类型注解：

```python
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class ModelInfo:
    id: str
    name: str
    path: str
    model_type: ModelType
    size: int
    config: Dict[str, Any]

def load_model(model_path: str, config: Dict[str, Any]) -> Optional[Any]:
    """加载模型文件。
    
    Args:
        model_path: 模型文件路径
        config: 模型配置参数
        
    Returns:
        加载的模型对象，失败时返回None
    """
    try:
        # 实现代码
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None
```

### 文档字符串

使用Google风格的文档字符串：

```python
def generate_stream(self, prompt: str, config: Dict[str, Any]) -> Iterator[str]:
    """生成流式文本输出。
    
    Args:
        prompt: 输入提示词
        config: 生成配置参数
        
    Yields:
        生成的文本片段
        
    Raises:
        ModelNotLoadedError: 模型未加载时抛出
        InferenceError: 推理过程出错时抛出
        
    Example:
        >>> inferencer = PyTorchInferencer()
        >>> inferencer.load_model("path/to/model", {})
        >>> for text in inferencer.generate_stream("Hello", {}):
        ...     print(text, end="")
    """
```

### 错误处理

使用自定义异常类和适当的错误处理：

```python
from multi_llm_comparator.core.exceptions import ModelNotFoundError, InferenceError

def load_model(self, model_path: str) -> None:
    """加载模型。"""
    if not os.path.exists(model_path):
        raise ModelNotFoundError(f"Model file not found: {model_path}")
    
    try:
        self.model = torch.load(model_path)
    except Exception as e:
        raise InferenceError(f"Failed to load model: {e}") from e
```

## 测试指南

### 测试结构

```
tests/
├── unit/                   # 单元测试
│   ├── test_model_manager.py
│   ├── test_inferencers.py
│   └── test_config.py
├── integration/            # 集成测试
│   ├── test_inference_engine.py
│   └── test_ui_integration.py
├── fixtures/               # 测试数据
│   ├── models/
│   └── configs/
└── conftest.py            # pytest配置
```

### 编写测试

```python
import pytest
from unittest.mock import Mock, patch
from multi_llm_comparator.services.model_manager import ModelManager

class TestModelManager:
    """模型管理器测试类。"""
    
    @pytest.fixture
    def model_manager(self):
        """创建模型管理器实例。"""
        return ModelManager()
    
    def test_scan_models_success(self, model_manager, tmp_path):
        """测试模型扫描成功场景。"""
        # 创建测试模型文件
        model_file = tmp_path / "test_model.bin"
        model_file.write_text("fake model data")
        
        # 执行扫描
        models = model_manager.scan_models(str(tmp_path))
        
        # 验证结果
        assert len(models) == 1
        assert models[0].name == "test_model"
        assert models[0].model_type == ModelType.PYTORCH
    
    def test_scan_models_empty_directory(self, model_manager, tmp_path):
        """测试空目录扫描。"""
        models = model_manager.scan_models(str(tmp_path))
        assert len(models) == 0
    
    @patch('torch.load')
    def test_load_model_failure(self, mock_torch_load, model_manager):
        """测试模型加载失败。"""
        mock_torch_load.side_effect = RuntimeError("CUDA out of memory")
        
        with pytest.raises(InferenceError):
            model_manager.load_model("fake_path.bin", {})
```

### 运行测试

```bash
# 运行所有测试
uv run pytest

# 运行特定测试文件
uv run pytest tests/unit/test_model_manager.py

# 运行特定测试方法
uv run pytest tests/unit/test_model_manager.py::TestModelManager::test_scan_models_success

# 生成覆盖率报告
uv run pytest --cov=src/multi_llm_comparator --cov-report=html

# 运行性能测试
uv run pytest tests/performance/ -v
```

## 提交规范

### 提交消息格式

使用[Conventional Commits](https://www.conventionalcommits.org/)规范：

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### 提交类型

- `feat`: 新功能
- `fix`: 修复bug
- `docs`: 文档更新
- `style`: 代码格式化（不影响功能）
- `refactor`: 代码重构
- `test`: 添加或修改测试
- `chore`: 构建过程或辅助工具的变动

### 示例

```bash
# 新功能
git commit -m "feat(inference): add GGUF model support"

# 修复bug
git commit -m "fix(ui): resolve memory leak in model display"

# 文档更新
git commit -m "docs: update installation guide for Windows"

# 重构
git commit -m "refactor(core): simplify model loading logic"
```

## Pull Request指南

### PR标题

使用与提交消息相同的格式：
```
feat(inference): add streaming output support
```

### PR描述模板

```markdown
## 变更类型
- [ ] 新功能
- [ ] Bug修复
- [ ] 文档更新
- [ ] 代码重构
- [ ] 性能优化

## 变更描述
简要描述此PR的变更内容。

## 测试
- [ ] 添加了新的测试用例
- [ ] 所有现有测试通过
- [ ] 手动测试通过

## 检查清单
- [ ] 代码遵循项目规范
- [ ] 添加了必要的文档
- [ ] 更新了CHANGELOG.md
- [ ] 没有引入破坏性变更

## 相关Issue
Closes #123
```

### 代码审查

所有PR都需要经过代码审查：

1. **自我审查**：提交前仔细检查代码
2. **同行审查**：至少一个维护者的批准
3. **自动检查**：通过所有CI检查

## 发布流程

### 版本号规范

使用[语义化版本](https://semver.org/)：
- `MAJOR.MINOR.PATCH`
- 例如：`1.2.3`

### 发布步骤

1. **更新版本号**
```bash
# 更新pyproject.toml中的版本号
version = "1.2.3"
```

2. **更新CHANGELOG**
```markdown
## [1.2.3] - 2024-01-15

### Added
- 新增GGUF模型支持

### Fixed
- 修复内存泄漏问题

### Changed
- 优化模型加载性能
```

3. **创建发布标签**
```bash
git tag -a v1.2.3 -m "Release version 1.2.3"
git push origin v1.2.3
```

## 社区参与

### 报告问题

使用GitHub Issues报告bug或请求功能：

1. 搜索现有Issues避免重复
2. 使用适当的Issue模板
3. 提供详细的复现步骤
4. 包含系统环境信息

### 参与讨论

- GitHub Discussions：项目相关讨论
- Code Review：参与代码审查
- Documentation：改进文档

### 行为准则

请遵循我们的[行为准则](CODE_OF_CONDUCT.md)，营造友好的社区环境。

## 资源链接

- [项目主页](https://github.com/your-org/multi-llm-comparator)
- [问题跟踪](https://github.com/your-org/multi-llm-comparator/issues)
- [讨论区](https://github.com/your-org/multi-llm-comparator/discussions)
- [Wiki](https://github.com/your-org/multi-llm-comparator/wiki)

感谢您的贡献！🎉