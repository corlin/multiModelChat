# 模型配置和使用说明

本文档详细介绍如何配置和使用多LLM模型比较器中的各种模型。

## 支持的模型格式

### PyTorch模型

#### 支持的文件格式
- `.bin` - 标准PyTorch模型文件
- `.pt` / `.pth` - PyTorch保存格式
- `.safetensors` - 安全张量格式（推荐使用）

#### 目录结构
```
models/pytorch/
├── model1/
│   ├── pytorch_model.bin
│   ├── config.json
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── model2/
│   ├── model.safetensors
│   ├── config.json
│   └── tokenizer files...
└── single_file_model.bin
```

#### 配置参数详解

**temperature** (0.1 - 2.0, 默认: 0.7)
- 控制输出的随机性和创造性
- 较低值(0.1-0.5)：更确定性、一致的输出
- 较高值(1.0-2.0)：更随机、创造性的输出

**top_p** (0.1 - 1.0, 默认: 0.9)
- 核采样参数，控制候选token的累积概率
- 较低值：更保守的选择
- 较高值：更多样化的输出

**max_tokens** (1 - 4096, 默认: 512)
- 生成的最大token数量
- 注意：实际输出可能因EOS token而提前结束

**do_sample** (true/false, 默认: true)
- 是否启用采样
- false：使用贪婪解码
- true：使用采样策略

#### 推荐配置

**创意写作**
```json
{
  "temperature": 1.0,
  "top_p": 0.9,
  "max_tokens": 1024,
  "do_sample": true
}
```

**代码生成**
```json
{
  "temperature": 0.2,
  "top_p": 0.95,
  "max_tokens": 512,
  "do_sample": true
}
```

**问答任务**
```json
{
  "temperature": 0.3,
  "top_p": 0.9,
  "max_tokens": 256,
  "do_sample": true
}
```

### GGUF模型

#### 支持的文件格式
- `.gguf` - GGUF量化模型格式

#### 目录结构
```
models/gguf/
├── model1.gguf
├── model2-q4_0.gguf
├── model3-q8_0.gguf
└── subfolder/
    └── model4.gguf
```

#### 配置参数详解

**temperature** (0.1 - 2.0, 默认: 0.7)
- 与PyTorch模型相同，控制输出随机性

**top_k** (1 - 100, 默认: 40)
- Top-K采样，限制候选token数量
- 较低值：更保守的选择
- 较高值：更多样化的输出

**top_p** (0.1 - 1.0, 默认: 0.9)
- 核采样参数，与top_k结合使用

**repeat_penalty** (1.0 - 1.5, 默认: 1.1)
- 重复惩罚，避免生成重复内容
- 1.0：无惩罚
- >1.0：惩罚重复

**n_ctx** (512 - 8192, 默认: 2048)
- 上下文窗口大小
- 影响模型能处理的最大输入长度
- 较大值需要更多内存

#### 推荐配置

**对话生成**
```json
{
  "temperature": 0.8,
  "top_k": 40,
  "top_p": 0.9,
  "repeat_penalty": 1.1,
  "n_ctx": 2048
}
```

**技术文档**
```json
{
  "temperature": 0.3,
  "top_k": 20,
  "top_p": 0.95,
  "repeat_penalty": 1.05,
  "n_ctx": 4096
}
```

**创意写作**
```json
{
  "temperature": 1.2,
  "top_k": 60,
  "top_p": 0.85,
  "repeat_penalty": 1.15,
  "n_ctx": 2048
}
```

## 模型选择建议

### 内存使用指南

| 模型大小 | 推荐RAM | 同时比较数量 |
|---------|---------|-------------|
| 7B参数  | 16GB    | 2-3个       |
| 13B参数 | 32GB    | 1-2个       |
| 30B参数 | 64GB    | 1个         |

### 性能优化建议

1. **使用GGUF量化模型**
   - Q4_0: 4位量化，显著减少内存使用
   - Q8_0: 8位量化，平衡质量和性能
   - F16: 半精度，最佳质量但内存使用较大

2. **PyTorch模型优化**
   - 使用safetensors格式
   - 启用low_cpu_mem_usage
   - 考虑使用torch.compile()（需要PyTorch 2.0+）

3. **硬件加速**
   - CUDA: NVIDIA GPU加速
   - MPS: Apple Silicon GPU加速
   - OpenCL: 通用GPU加速

## 常见模型来源

### Hugging Face Hub
```bash
# 下载PyTorch模型
git lfs clone https://huggingface.co/model-name

# 或使用huggingface-hub
pip install huggingface-hub
huggingface-cli download model-name
```

### GGUF模型
- [TheBloke](https://huggingface.co/TheBloke) - 大量GGUF量化模型
- [Microsoft](https://huggingface.co/microsoft) - 官方模型的GGUF版本

## 配置文件管理

### 自动保存
系统会自动将配置保存到 `~/.multi_llm_comparator/config.json`

### 手动配置
```json
{
  "models": {
    "model1": {
      "type": "pytorch",
      "config": {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 512,
        "do_sample": true
      }
    },
    "model2": {
      "type": "gguf",
      "config": {
        "temperature": 0.7,
        "top_k": 40,
        "top_p": 0.9,
        "repeat_penalty": 1.1,
        "n_ctx": 2048
      }
    }
  }
}
```

## 最佳实践

1. **模型选择**
   - 选择不同类型的模型进行比较
   - 考虑模型的训练数据和用途
   - 平衡模型大小和可用内存

2. **参数调优**
   - 从默认参数开始
   - 根据任务类型调整参数
   - 记录有效的参数组合

3. **性能监控**
   - 关注内存使用情况
   - 监控生成速度
   - 注意模型加载时间

4. **结果分析**
   - 比较不同参数设置的效果
   - 分析模型在不同任务上的表现
   - 导出结果进行进一步分析