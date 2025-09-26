# 故障排除指南

本文档提供常见问题的解决方案和故障排除步骤。

## 安装问题

### uv安装失败

**问题**: uv安装命令执行失败
```
curl: command not found
```

**解决方案**:
1. **Windows**: 使用PowerShell或安装curl
```powershell
# 使用PowerShell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 或先安装curl
winget install curl
```

2. **macOS**: 安装Xcode命令行工具
```bash
xcode-select --install
```

3. **Linux**: 安装curl
```bash
# Ubuntu/Debian
sudo apt-get install curl

# CentOS/RHEL
sudo yum install curl
```

### Python版本不兼容

**问题**: Python版本低于3.12
```
error: Python 3.12+ is required
```

**解决方案**:
1. 安装Python 3.12+
```bash
# 使用pyenv
pyenv install 3.12.0
pyenv global 3.12.0

# 或从官网下载安装
# https://www.python.org/downloads/
```

2. 指定Python版本
```bash
uv python install 3.12
uv sync --python 3.12
```

### 依赖安装失败

**问题**: 某些包安装失败
```
error: Failed to build package
```

**解决方案**:
1. 更新uv到最新版本
```bash
uv self update
```

2. 清理缓存重新安装
```bash
uv cache clean
uv sync --reinstall
```

3. 检查系统依赖
```bash
# Ubuntu/Debian
sudo apt-get install build-essential python3-dev

# macOS
xcode-select --install

# Windows
# 安装Visual Studio Build Tools
```

## 模型加载问题

### 模型文件未找到

**问题**: 系统无法发现模型文件
```
No models found in directory
```

**解决方案**:
1. 检查模型目录结构
```bash
ls -la models/
ls -la models/pytorch/
ls -la models/gguf/
```

2. 确认文件权限
```bash
chmod -R 755 models/
```

3. 检查文件格式
- PyTorch: `.bin`, `.pt`, `.pth`, `.safetensors`
- GGUF: `.gguf`

### 内存不足错误

**问题**: 模型加载时内存不足
```
RuntimeError: CUDA out of memory
OutOfMemoryError: Unable to allocate memory
```

**解决方案**:
1. 减少同时比较的模型数量
2. 使用量化模型（GGUF Q4_0, Q8_0）
3. 调整模型配置
```python
# PyTorch模型
config = {
    "low_cpu_mem_usage": True,
    "torch_dtype": "float16",  # 使用半精度
    "device_map": "auto"
}

# GGUF模型
config = {
    "n_ctx": 1024,  # 减少上下文长度
    "n_gpu_layers": 0  # 禁用GPU加速
}
```

### 模型格式不支持

**问题**: 模型格式无法识别
```
Unsupported model format
```

**解决方案**:
1. 转换模型格式
```bash
# 转换为safetensors
python -c "
from transformers import AutoModel
model = AutoModel.from_pretrained('path/to/model')
model.save_pretrained('path/to/output', safe_serialization=True)
"

# 转换为GGUF (需要llama.cpp)
python convert.py path/to/model --outtype f16 --outfile model.gguf
```

2. 检查模型完整性
```bash
# 检查PyTorch模型
python -c "
import torch
try:
    model = torch.load('model.bin', map_location='cpu')
    print('Model loaded successfully')
except Exception as e:
    print(f'Error: {e}')
"
```

## 推理问题

### 生成速度过慢

**问题**: 模型生成速度很慢

**解决方案**:
1. 启用GPU加速
```python
# 检查GPU可用性
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
```

2. 优化推理参数
```python
# 减少生成长度
config = {
    "max_tokens": 256,  # 减少最大token数
    "do_sample": False,  # 使用贪婪解码
}
```

3. 使用更小的模型或量化版本

### 生成内容质量差

**问题**: 模型输出质量不佳

**解决方案**:
1. 调整采样参数
```python
# 提高质量的配置
config = {
    "temperature": 0.3,  # 降低随机性
    "top_p": 0.95,       # 提高top_p
    "repeat_penalty": 1.1  # 避免重复
}
```

2. 改进提示词
- 提供更清晰的指令
- 添加示例和上下文
- 使用结构化的提示格式

### 生成中断或错误

**问题**: 推理过程中出现异常
```
RuntimeError: The size of tensor a must match...
```

**解决方案**:
1. 检查输入长度
```python
# 限制输入长度
max_input_length = 2048
if len(prompt) > max_input_length:
    prompt = prompt[:max_input_length]
```

2. 重启推理引擎
```python
# 清理资源
inference_engine.cleanup_resources()
torch.cuda.empty_cache()  # 清理GPU缓存
```

## 界面问题

### Streamlit启动失败

**问题**: Streamlit无法启动
```
ModuleNotFoundError: No module named 'streamlit'
```

**解决方案**:
1. 确认在正确的虚拟环境中
```bash
uv run which python
uv run pip list | grep streamlit
```

2. 重新安装依赖
```bash
uv sync --reinstall
```

### 界面响应缓慢

**问题**: Web界面响应很慢

**解决方案**:
1. 检查系统资源使用
```bash
# 监控系统资源
top
htop
nvidia-smi  # GPU使用情况
```

2. 优化Streamlit配置
```toml
# .streamlit/config.toml
[server]
maxUploadSize = 200
maxMessageSize = 200

[browser]
gatherUsageStats = false
```

### 输出显示异常

**问题**: 模型输出显示不正确

**解决方案**:
1. 检查编码设置
```python
# 确保UTF-8编码
import locale
print(locale.getpreferredencoding())
```

2. 清理浏览器缓存
- 按Ctrl+F5强制刷新
- 清除浏览器缓存和Cookie

## 性能优化

### 内存使用优化

1. **监控内存使用**
```python
import psutil
import torch

# 系统内存
memory = psutil.virtual_memory()
print(f"Available memory: {memory.available / 1024**3:.2f} GB")

# GPU内存
if torch.cuda.is_available():
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
```

2. **内存清理**
```python
# 强制垃圾回收
import gc
gc.collect()

# 清理GPU缓存
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

### 推理速度优化

1. **使用编译优化**
```python
# PyTorch 2.0+ 编译优化
model = torch.compile(model)
```

2. **批处理优化**
```python
# 批量处理多个提示
batch_size = 4
prompts = [prompt1, prompt2, prompt3, prompt4]
```

## 日志和调试

### 启用详细日志

1. **设置日志级别**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

2. **查看系统日志**
```bash
# 查看应用日志
tail -f ~/.multi_llm_comparator/logs/app.log

# 查看错误日志
tail -f ~/.multi_llm_comparator/logs/error.log
```

### 调试模式

1. **启用调试模式**
```bash
export DEBUG=1
uv run streamlit run src/multi_llm_comparator/main.py
```

2. **性能分析**
```python
import cProfile
cProfile.run('your_function()')
```

## 获取帮助

如果以上解决方案都无法解决问题：

1. **收集系统信息**
```bash
uv run python -c "
import sys, torch, transformers, streamlit
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'Streamlit: {streamlit.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

2. **创建Issue**
- 访问项目GitHub页面
- 创建新的Issue
- 提供详细的错误信息和系统信息

3. **社区支持**
- 查看项目Wiki
- 搜索相关讨论
- 参与社区交流