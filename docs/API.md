# API文档

本文档详细描述了多LLM模型比较器的核心API接口和使用方法。

## 核心模块

### multi_llm_comparator.core

#### 数据模型 (models.py)

##### ModelInfo
```python
@dataclass
class ModelInfo:
    """模型信息数据类。"""
    id: str                    # 模型唯一标识符
    name: str                  # 模型显示名称
    path: str                  # 模型文件路径
    model_type: ModelType      # 模型类型（PYTORCH或GGUF）
    size: int                  # 模型文件大小（字节）
    config: Dict[str, Any]     # 模型配置参数
    
    def __post_init__(self):
        """初始化后处理。"""
        if not self.id:
            self.id = self.generate_id()
    
    def generate_id(self) -> str:
        """生成模型ID。"""
        return hashlib.md5(self.path.encode()).hexdigest()[:8]
```

##### ModelType
```python
class ModelType(Enum):
    """模型类型枚举。"""
    PYTORCH = "pytorch"
    GGUF = "gguf"
```

##### InferenceResult
```python
@dataclass
class InferenceResult:
    """推理结果数据类。"""
    model_id: str              # 模型ID
    content: str               # 生成的内容
    is_complete: bool          # 是否生成完成
    error: Optional[str]       # 错误信息
    stats: InferenceStats      # 推理统计信息
```

##### InferenceStats
```python
@dataclass
class InferenceStats:
    """推理统计信息。"""
    start_time: float          # 开始时间戳
    end_time: Optional[float]  # 结束时间戳
    token_count: int           # 生成的token数量
    tokens_per_second: Optional[float]  # 生成速度
    memory_usage: Optional[int]         # 内存使用量（字节）
    
    @property
    def duration(self) -> Optional[float]:
        """计算推理持续时间。"""
        if self.end_time:
            return self.end_time - self.start_time
        return None
```

#### 配置管理 (config.py)

##### ModelConfig
```python
@dataclass
class ModelConfig:
    """模型配置类。"""
    # 通用参数
    temperature: float = 0.7
    max_tokens: int = 512
    top_p: float = 0.9
    
    # PyTorch特定参数
    do_sample: bool = True
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    
    # GGUF特定参数
    top_k: int = 40
    repeat_penalty: float = 1.1
    n_ctx: int = 2048
    n_threads: Optional[int] = None
    use_gpu: bool = True
    
    # 内存管理参数
    low_cpu_mem_usage: bool = True
    torch_dtype: str = "auto"
    
    def validate(self) -> None:
        """验证配置参数。"""
        if not 0.1 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0.1 and 2.0")
        if not 1 <= self.max_tokens <= 4096:
            raise ValueError("max_tokens must be between 1 and 4096")
        # ... 其他验证逻辑
```

##### ConfigManager
```python
class ConfigManager:
    """配置管理器。"""
    
    def __init__(self, config_path: str = "~/.multi_llm_comparator/config.json"):
        self.config_path = Path(config_path).expanduser()
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
    
    def load_config(self) -> Dict[str, ModelConfig]:
        """加载配置文件。"""
        if not self.config_path.exists():
            return {}
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return {
            model_id: ModelConfig(**config)
            for model_id, config in data.items()
        }
    
    def save_config(self, configs: Dict[str, ModelConfig]) -> None:
        """保存配置文件。"""
        data = {
            model_id: asdict(config)
            for model_id, config in configs.items()
        }
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
```

#### 异常处理 (exceptions.py)

```python
class MultiLLMError(Exception):
    """基础异常类。"""
    pass

class ModelNotFoundError(MultiLLMError):
    """模型未找到异常。"""
    pass

class ModelLoadError(MultiLLMError):
    """模型加载异常。"""
    pass

class InferenceError(MultiLLMError):
    """推理异常。"""
    pass

class ConfigurationError(MultiLLMError):
    """配置异常。"""
    pass
```

### multi_llm_comparator.services

#### 模型管理器 (model_manager.py)

```python
class ModelManager:
    """模型管理器，负责模型发现、选择和管理。"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.available_models: List[ModelInfo] = []
        self.selected_models: List[ModelInfo] = []
        self.max_models = 4
    
    def scan_models(self, directory: Optional[str] = None) -> List[ModelInfo]:
        """扫描指定目录中的模型文件。
        
        Args:
            directory: 扫描目录路径，默认使用models_dir
            
        Returns:
            发现的模型信息列表
        """
        scan_dir = Path(directory) if directory else self.models_dir
        models = []
        
        # 扫描PyTorch模型
        pytorch_dir = scan_dir / "pytorch"
        if pytorch_dir.exists():
            models.extend(self._scan_pytorch_models(pytorch_dir))
        
        # 扫描GGUF模型
        gguf_dir = scan_dir / "gguf"
        if gguf_dir.exists():
            models.extend(self._scan_gguf_models(gguf_dir))
        
        self.available_models = models
        return models
    
    def _scan_pytorch_models(self, directory: Path) -> List[ModelInfo]:
        """扫描PyTorch模型文件。"""
        models = []
        extensions = ['.bin', '.pt', '.pth', '.safetensors']
        
        for ext in extensions:
            for model_file in directory.rglob(f"*{ext}"):
                model_info = ModelInfo(
                    id="",
                    name=model_file.stem,
                    path=str(model_file),
                    model_type=ModelType.PYTORCH,
                    size=model_file.stat().st_size,
                    config={}
                )
                models.append(model_info)
        
        return models
    
    def _scan_gguf_models(self, directory: Path) -> List[ModelInfo]:
        """扫描GGUF模型文件。"""
        models = []
        
        for model_file in directory.rglob("*.gguf"):
            model_info = ModelInfo(
                id="",
                name=model_file.stem,
                path=str(model_file),
                model_type=ModelType.GGUF,
                size=model_file.stat().st_size,
                config={}
            )
            models.append(model_info)
        
        return models
    
    def select_models(self, model_ids: List[str]) -> bool:
        """选择要比较的模型。
        
        Args:
            model_ids: 模型ID列表
            
        Returns:
            选择是否成功
            
        Raises:
            ValueError: 选择的模型数量超过限制
        """
        if len(model_ids) > self.max_models:
            raise ValueError(f"Cannot select more than {self.max_models} models")
        
        selected = []
        for model_id in model_ids:
            model = self.get_model_by_id(model_id)
            if model:
                selected.append(model)
        
        self.selected_models = selected
        return len(selected) == len(model_ids)
    
    def get_model_by_id(self, model_id: str) -> Optional[ModelInfo]:
        """根据ID获取模型信息。"""
        for model in self.available_models:
            if model.id == model_id:
                return model
        return None
    
    def get_available_models(self) -> List[ModelInfo]:
        """获取可用模型列表。"""
        return self.available_models.copy()
    
    def get_selected_models(self) -> List[ModelInfo]:
        """获取已选择的模型列表。"""
        return self.selected_models.copy()
```

#### 推理引擎 (inference_engine.py)

```python
class InferenceEngine:
    """推理引擎，协调多模型推理过程。"""
    
    def __init__(self, memory_manager: Optional[MemoryManager] = None):
        self.memory_manager = memory_manager or MemoryManager()
        self.active_inferencers: Dict[str, BaseInferencer] = {}
    
    def run_inference(
        self,
        prompt: str,
        models: List[ModelInfo],
        configs: Dict[str, ModelConfig]
    ) -> Iterator[Dict[str, InferenceResult]]:
        """运行多模型推理。
        
        Args:
            prompt: 输入提示词
            models: 模型列表
            configs: 模型配置字典
            
        Yields:
            每个模型的推理结果
        """
        results = {model.id: InferenceResult(
            model_id=model.id,
            content="",
            is_complete=False,
            error=None,
            stats=InferenceStats(
                start_time=time.time(),
                end_time=None,
                token_count=0,
                tokens_per_second=None
            )
        ) for model in models}
        
        for model in models:
            try:
                # 创建推理器
                inferencer = self.create_inferencer(model)
                config = configs.get(model.id, ModelConfig())
                
                # 加载模型
                inferencer.load_model(model.path, asdict(config))
                self.active_inferencers[model.id] = inferencer
                
                # 执行推理
                for token in inferencer.generate_stream(prompt):
                    results[model.id].content += token
                    results[model.id].stats.token_count += 1
                    yield results.copy()
                
                # 标记完成
                results[model.id].is_complete = True
                results[model.id].stats.end_time = time.time()
                
                # 计算速度
                duration = results[model.id].stats.duration
                if duration and duration > 0:
                    results[model.id].stats.tokens_per_second = (
                        results[model.id].stats.token_count / duration
                    )
                
            except Exception as e:
                results[model.id].error = str(e)
                results[model.id].is_complete = True
            
            finally:
                # 清理资源
                if model.id in self.active_inferencers:
                    self.active_inferencers[model.id].unload_model()
                    del self.active_inferencers[model.id]
                
                self.memory_manager.cleanup()
        
        yield results
    
    def create_inferencer(self, model_info: ModelInfo) -> BaseInferencer:
        """创建推理器实例。"""
        if model_info.model_type == ModelType.PYTORCH:
            from multi_llm_comparator.inferencers.pytorch_inferencer import PyTorchInferencer
            return PyTorchInferencer()
        elif model_info.model_type == ModelType.GGUF:
            from multi_llm_comparator.inferencers.gguf_inferencer import GGUFInferencer
            return GGUFInferencer()
        else:
            raise ValueError(f"Unsupported model type: {model_info.model_type}")
    
    def cleanup_resources(self) -> None:
        """清理所有资源。"""
        for inferencer in self.active_inferencers.values():
            inferencer.unload_model()
        self.active_inferencers.clear()
        self.memory_manager.cleanup()
```

### multi_llm_comparator.inferencers

#### 基础推理器 (base.py)

```python
class BaseInferencer(ABC):
    """推理器基类。"""
    
    def __init__(self):
        self.model = None
        self.is_loaded = False
    
    @abstractmethod
    def load_model(self, model_path: str, config: Dict[str, Any]) -> None:
        """加载模型。
        
        Args:
            model_path: 模型文件路径
            config: 模型配置参数
            
        Raises:
            ModelLoadError: 模型加载失败
        """
        pass
    
    @abstractmethod
    def generate_stream(self, prompt: str) -> Iterator[str]:
        """生成流式文本。
        
        Args:
            prompt: 输入提示词
            
        Yields:
            生成的文本片段
            
        Raises:
            InferenceError: 推理过程出错
        """
        pass
    
    @abstractmethod
    def unload_model(self) -> None:
        """卸载模型，释放资源。"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息。
        
        Returns:
            模型信息字典
        """
        pass
    
    def is_model_loaded(self) -> bool:
        """检查模型是否已加载。"""
        return self.is_loaded
```

#### PyTorch推理器 (pytorch_inferencer.py)

```python
class PyTorchInferencer(BaseInferencer):
    """PyTorch模型推理器。"""
    
    def __init__(self):
        super().__init__()
        self.tokenizer = None
        self.device = self._get_device()
    
    def _get_device(self) -> torch.device:
        """获取推理设备。"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def load_model(self, model_path: str, config: Dict[str, Any]) -> None:
        """加载PyTorch模型。"""
        try:
            from transformers import AutoModel, AutoTokenizer
            
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            # 加载模型
            self.model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=getattr(torch, config.get("torch_dtype", "auto")),
                low_cpu_mem_usage=config.get("low_cpu_mem_usage", True),
                trust_remote_code=True
            ).to(self.device)
            
            self.is_loaded = True
            
        except Exception as e:
            raise ModelLoadError(f"Failed to load PyTorch model: {e}") from e
    
    def generate_stream(self, prompt: str) -> Iterator[str]:
        """生成流式文本。"""
        if not self.is_loaded:
            raise InferenceError("Model not loaded")
        
        try:
            from transformers import TextIteratorStreamer
            import threading
            
            # 编码输入
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # 创建流式器
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )
            
            # 生成参数
            generation_kwargs = {
                "input_ids": inputs,
                "streamer": streamer,
                "max_new_tokens": 512,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "pad_token_id": self.tokenizer.eos_token_id
            }
            
            # 在后台线程中生成
            thread = threading.Thread(
                target=self.model.generate,
                kwargs=generation_kwargs
            )
            thread.start()
            
            # 流式输出
            for text in streamer:
                yield text
            
            thread.join()
            
        except Exception as e:
            raise InferenceError(f"Inference failed: {e}") from e
    
    def unload_model(self) -> None:
        """卸载模型。"""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        self.is_loaded = False
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息。"""
        if not self.is_loaded:
            return {}
        
        return {
            "model_type": "pytorch",
            "device": str(self.device),
            "vocab_size": self.tokenizer.vocab_size if self.tokenizer else None,
            "model_size": sum(p.numel() for p in self.model.parameters()) if self.model else None
        }
```

## 使用示例

### 基本使用

```python
from multi_llm_comparator.services.model_manager import ModelManager
from multi_llm_comparator.services.inference_engine import InferenceEngine
from multi_llm_comparator.core.models import ModelConfig

# 初始化组件
model_manager = ModelManager("models")
inference_engine = InferenceEngine()

# 扫描模型
models = model_manager.scan_models()
print(f"Found {len(models)} models")

# 选择模型
selected_model_ids = [models[0].id, models[1].id]
model_manager.select_models(selected_model_ids)

# 配置参数
configs = {
    models[0].id: ModelConfig(temperature=0.7, max_tokens=512),
    models[1].id: ModelConfig(temperature=0.8, max_tokens=512)
}

# 执行推理
prompt = "请介绍一下人工智能的发展历史。"
selected_models = model_manager.get_selected_models()

for results in inference_engine.run_inference(prompt, selected_models, configs):
    for model_id, result in results.items():
        if not result.is_complete:
            print(f"Model {model_id}: {result.content}")
        else:
            print(f"Model {model_id} completed in {result.stats.duration:.2f}s")
```

### 高级使用

```python
import asyncio
from multi_llm_comparator.services.memory_manager import MemoryManager

# 自定义内存管理
memory_manager = MemoryManager(max_memory_gb=8)
inference_engine = InferenceEngine(memory_manager)

# 批量推理
prompts = [
    "解释量子计算的基本原理",
    "描述机器学习的主要算法",
    "分析区块链技术的应用前景"
]

for prompt in prompts:
    print(f"\n=== Prompt: {prompt} ===")
    
    for results in inference_engine.run_inference(prompt, selected_models, configs):
        # 实时显示结果
        for model_id, result in results.items():
            if result.error:
                print(f"Error in {model_id}: {result.error}")
            elif result.is_complete:
                stats = result.stats
                print(f"{model_id}: {stats.token_count} tokens, {stats.tokens_per_second:.2f} t/s")
```

## 错误处理

所有API调用都应该包含适当的错误处理：

```python
try:
    models = model_manager.scan_models()
except ModelNotFoundError as e:
    print(f"No models found: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")

try:
    for results in inference_engine.run_inference(prompt, models, configs):
        # 处理结果
        pass
except InferenceError as e:
    print(f"Inference failed: {e}")
except ConfigurationError as e:
    print(f"Configuration error: {e}")
```

## 扩展API

### 自定义推理器

```python
class CustomInferencer(BaseInferencer):
    """自定义推理器示例。"""
    
    def load_model(self, model_path: str, config: Dict[str, Any]) -> None:
        # 实现自定义模型加载逻辑
        pass
    
    def generate_stream(self, prompt: str) -> Iterator[str]:
        # 实现自定义推理逻辑
        pass
    
    def unload_model(self) -> None:
        # 实现资源清理逻辑
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        # 返回模型信息
        return {"model_type": "custom"}

# 注册自定义推理器
def create_custom_inferencer(model_info: ModelInfo) -> BaseInferencer:
    if model_info.model_type == ModelType.CUSTOM:
        return CustomInferencer()
    return None

# 扩展推理引擎
inference_engine.register_inferencer_factory(create_custom_inferencer)
```

这个API文档提供了完整的接口说明和使用示例，开发者可以基于这些API构建自己的应用或扩展功能。