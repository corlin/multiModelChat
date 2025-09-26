# 多LLM模型比较器

一个基于Python 3.12和uv包管理器开发的本地部署Streamlit Web应用，用于同时比较多个大语言模型的输出结果。

## 特性

- 🚀 支持PyTorch和GGUF格式的本地模型
- 🔄 最多同时比较4个模型
- 💾 内存优化策略，动态加载和卸载模型
- ⚡ 流式输出，实时显示生成过程
- 📝 Markdown渲染和格式化显示
- 📊 性能统计和结果导出
- 🎯 基于uv的现代Python项目管理

## 快速开始

### 环境要求

- Python 3.12+
- uv包管理器
- 至少8GB RAM（推荐16GB+）
- 可选：NVIDIA GPU（用于CUDA加速）

### 安装uv

#### Windows
```bash
# 使用PowerShell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 或使用pip
pip install uv
```

#### macOS/Linux
```bash
# 使用curl
curl -LsSf https://astral.sh/uv/install.sh | sh

# 或使用pip
pip install uv
```

### 项目安装

1. 克隆项目
```bash
git clone https://github.com/your-org/multi-llm-comparator.git
cd multi-llm-comparator
```

2. 使用uv创建虚拟环境并安装依赖
```bash
uv sync
```

3. 验证环境设置
```bash
# 验证uv环境和项目配置
uv run python verify_setup.py
```

4. 准备模型文件
```bash
# 创建模型目录（验证脚本会自动创建）
mkdir -p models/pytorch models/gguf

# 将PyTorch模型放入 models/pytorch/
# 将GGUF模型放入 models/gguf/
```

5. 运行应用

使用便捷脚本：
```bash
# 方式1：使用启动脚本
uv run python run_app.py

# 方式2：直接运行
uv run streamlit run src/multi_llm_comparator/main.py

# 方式3：使用开发工具
uv run python dev_tools.py run
```

应用将在 http://localhost:8501 启动。

## 使用指南

### 模型准备

#### PyTorch模型
支持的文件格式：
- `.bin` - 标准PyTorch模型文件
- `.pt` / `.pth` - PyTorch保存格式
- `.safetensors` - 安全张量格式（推荐）

将模型文件放入 `models/pytorch/` 目录，系统会自动发现并识别。

#### GGUF模型
支持的文件格式：
- `.gguf` - GGUF量化模型格式

将模型文件放入 `models/gguf/` 目录。

### 基本使用流程

1. **启动应用**：运行 `uv run streamlit run src/multi_llm_comparator/main.py`
2. **选择模型**：在侧边栏选择1-4个要比较的模型
3. **配置参数**：为每个模型设置推理参数
4. **输入提示词**：在主界面输入要测试的提示词
5. **开始比较**：点击"开始比较"按钮
6. **查看结果**：实时查看各模型的输出和性能统计
7. **导出结果**：可选择导出为JSON、CSV、Markdown等格式

### 参数配置

#### PyTorch模型参数
- `temperature`: 控制输出随机性 (0.1-2.0)
- `top_p`: 核采样参数 (0.1-1.0)
- `max_tokens`: 最大生成token数 (1-4096)
- `do_sample`: 是否启用采样

#### GGUF模型参数
- `temperature`: 控制输出随机性 (0.1-2.0)
- `top_k`: Top-K采样参数 (1-100)
- `top_p`: 核采样参数 (0.1-1.0)
- `repeat_penalty`: 重复惩罚 (1.0-1.5)
- `n_ctx`: 上下文长度 (512-8192)

## 项目结构

```
multi-llm-comparator/
├── pyproject.toml              # uv项目配置
├── uv.lock                     # 依赖锁定文件
├── README.md                   # 项目说明
├── src/
│   └── multi_llm_comparator/
│       ├── main.py             # Streamlit应用入口
│       ├── core/               # 核心数据模型
│       ├── services/           # 业务服务层
│       ├── inferencers/        # 推理器实现
│       └── ui/                 # UI组件
├── tests/                      # 测试文件
├── models/                     # 模型存储目录
│   ├── pytorch/               # PyTorch模型
│   └── gguf/                  # GGUF模型
└── docs/                      # 文档目录
```

## 开发

### 开发环境设置

1. 安装开发依赖
```bash
uv sync --dev
```

2. 使用开发工具脚本
```bash
# 安装依赖
uv run python dev_tools.py install

# 运行测试
uv run python dev_tools.py test

# 代码格式化
uv run python dev_tools.py format

# 代码检查
uv run python dev_tools.py lint

# 类型检查
uv run python dev_tools.py typecheck

# 流式输出测试
uv run python dev_tools.py streaming-test

# 运行完整开发流程
uv run python dev_tools.py all
```

3. 手动运行工具
```bash
# 运行测试
uv run pytest tests/ -v

# 代码格式化
uv run black src/ tests/
uv run ruff check --fix src/ tests/

# 类型检查
uv run mypy src/
```

### 贡献指南

请参阅 [CONTRIBUTING.md](docs/CONTRIBUTING.md) 了解如何为项目做贡献。

## 文档

- [模型配置指南](docs/MODEL_CONFIGURATION.md)
- [故障排除](docs/TROUBLESHOOTING.md)
- [开发者指南](docs/DEVELOPMENT.md)
- [API文档](docs/API.md)
- [架构说明](docs/ARCHITECTURE.md)

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 支持

如果遇到问题，请：
1. 查看 [故障排除指南](docs/TROUBLESHOOTING.md)
2. 搜索现有的 [Issues](../../issues)
3. 创建新的 Issue 描述问题

## 更新日志

查看 [CHANGELOG.md](CHANGELOG.md) 了解版本更新信息。