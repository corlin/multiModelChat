"""
模型文件扫描器

负责扫描本地目录中的模型文件，识别文件类型并收集元数据。
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Optional, Set
from dataclasses import dataclass

from ..core.models import ModelInfo, ModelType


logger = logging.getLogger(__name__)


@dataclass
class ScanResult:
    """扫描结果数据类"""
    models: List[ModelInfo]
    errors: List[str]
    scanned_files: int
    valid_models: int


class ModelFileScanner:
    """模型文件扫描器类"""
    
    # 支持的文件扩展名映射到模型类型
    SUPPORTED_EXTENSIONS = {
        '.bin': ModelType.PYTORCH,
        '.pt': ModelType.PYTORCH,
        '.pth': ModelType.PYTORCH,
        '.safetensors': ModelType.PYTORCH,
        '.gguf': ModelType.GGUF,
    }
    
    # PyTorch模型目录的标识文件
    PYTORCH_MODEL_FILES = {
        'pytorch_model.bin',
        'model.safetensors', 
        'pytorch_model.safetensors',
        'model.bin',
        'model.pt',
        'model.pth'
    }
    
    # PyTorch分片模型的索引文件
    PYTORCH_SHARDED_INDEX_FILES = {
        'model.safetensors.index.json',
        'pytorch_model.bin.index.json'
    }
    
    # PyTorch模型文件的模式（用于匹配分片文件）
    PYTORCH_MODEL_PATTERNS = [
        r'pytorch_model.*\.bin$',
        r'model.*\.safetensors$',
        r'pytorch_model.*\.safetensors$',
        r'model.*\.bin$',
        r'model.*\.pt$',
        r'model.*\.pth$'
    ]
    
    # PyTorch模型配置文件
    PYTORCH_CONFIG_FILES = {
        'config.json',
        'model_config.json'
    }
    
    # PyTorch tokenizer文件
    PYTORCH_TOKENIZER_FILES = {
        'tokenizer.json',
        'tokenizer_config.json',
        'vocab.txt',
        'merges.txt',
        'special_tokens_map.json'
    }
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def scan_directory(self, directory: str, recursive: bool = True) -> ScanResult:
        """
        扫描指定目录中的模型文件
        
        Args:
            directory: 要扫描的目录路径
            recursive: 是否递归扫描子目录
            
        Returns:
            ScanResult: 扫描结果，包含发现的模型和错误信息
        """
        directory_path = Path(directory)
        
        if not directory_path.exists():
            error_msg = f"目录不存在: {directory}"
            self.logger.error(error_msg)
            return ScanResult(models=[], errors=[error_msg], scanned_files=0, valid_models=0)
        
        if not directory_path.is_dir():
            error_msg = f"路径不是目录: {directory}"
            self.logger.error(error_msg)
            return ScanResult(models=[], errors=[error_msg], scanned_files=0, valid_models=0)
        
        models = []
        errors = []
        scanned_files = 0
        scanned_directories = set()
        
        try:
            self.logger.info(f"开始扫描目录: {directory}")
            
            # 首先扫描PyTorch模型目录
            pytorch_models = self._scan_pytorch_model_directories(directory_path, recursive, scanned_directories)
            models.extend(pytorch_models)
            
            # 然后扫描GGUF模型文件
            gguf_models = self._scan_gguf_model_files(directory_path, recursive, scanned_directories)
            models.extend(gguf_models)
            
            # 最后扫描单独的PyTorch模型文件（不在模型目录中的）
            standalone_models = self._scan_standalone_pytorch_files(directory_path, recursive, scanned_directories)
            models.extend(standalone_models)
            
            scanned_files = len(models)
            self.logger.info(f"扫描完成，发现 {len(models)} 个有效模型")
            
        except Exception as e:
            error_msg = f"扫描目录时发生错误: {str(e)}"
            errors.append(error_msg)
            self.logger.error(error_msg)
        
        return ScanResult(
            models=models,
            errors=errors,
            scanned_files=scanned_files,
            valid_models=len(models)
        )
    
    def _scan_pytorch_model_directories(self, base_path: Path, recursive: bool, scanned_directories: set) -> List[ModelInfo]:
        """
        扫描PyTorch模型目录
        
        Args:
            base_path: 基础扫描路径
            recursive: 是否递归扫描
            scanned_directories: 已扫描的目录集合
            
        Returns:
            List[ModelInfo]: 发现的PyTorch模型列表
        """
        models = []
        
        # 使用队列来处理目录，避免重复扫描
        directories_to_scan = []
        
        if recursive:
            # 递归获取所有子目录，但要避免重复
            for item in base_path.rglob('*'):
                if item.is_dir() and str(item) not in scanned_directories:
                    directories_to_scan.append(item)
        else:
            # 只扫描直接子目录
            for item in base_path.iterdir():
                if item.is_dir() and str(item) not in scanned_directories:
                    directories_to_scan.append(item)
        
        # 按深度排序，优先处理较深的目录（避免父目录被误识别为模型）
        directories_to_scan.sort(key=lambda x: len(x.parts), reverse=True)
        
        for dir_path in directories_to_scan:
            # 跳过已经被标记为其他模型子目录的目录
            if str(dir_path) in scanned_directories:
                continue
                
            # 检查是否是某个已识别模型的子目录
            is_subdirectory_of_model = False
            for scanned_dir in scanned_directories:
                if str(dir_path).startswith(scanned_dir + os.sep) or str(dir_path).startswith(scanned_dir + '/'):
                    is_subdirectory_of_model = True
                    break
            
            if is_subdirectory_of_model:
                continue
                
            try:
                model_info = self._analyze_pytorch_directory(dir_path)
                if model_info:
                    models.append(model_info)
                    # 标记这个目录及其所有子目录为已扫描
                    scanned_directories.add(str(dir_path))
                    # 标记所有子目录
                    for subdir in dir_path.rglob('*'):
                        if subdir.is_dir():
                            scanned_directories.add(str(subdir))
                    
                    self.logger.debug(f"发现PyTorch模型目录: {dir_path}")
            except Exception as e:
                self.logger.warning(f"分析PyTorch目录 {dir_path} 时出错: {str(e)}")
        
        return models
    
    def _scan_gguf_model_files(self, base_path: Path, recursive: bool, scanned_directories: set) -> List[ModelInfo]:
        """
        扫描GGUF模型文件
        
        Args:
            base_path: 基础扫描路径
            recursive: 是否递归扫描
            scanned_directories: 已扫描的目录集合
            
        Returns:
            List[ModelInfo]: 发现的GGUF模型列表
        """
        models = []
        
        # 获取所有GGUF文件
        if recursive:
            gguf_files = list(base_path.rglob('*.gguf'))
        else:
            gguf_files = list(base_path.glob('*.gguf'))
        
        for file_path in gguf_files:
            # 跳过已扫描目录中的文件
            if any(str(file_path).startswith(scanned_dir) for scanned_dir in scanned_directories):
                continue
                
            try:
                model_info = self._analyze_gguf_file(file_path)
                if model_info:
                    models.append(model_info)
                    self.logger.debug(f"发现GGUF模型文件: {file_path}")
            except Exception as e:
                self.logger.warning(f"分析GGUF文件 {file_path} 时出错: {str(e)}")
        
        return models
    
    def _scan_standalone_pytorch_files(self, base_path: Path, recursive: bool, scanned_directories: set) -> List[ModelInfo]:
        """
        扫描独立的PyTorch模型文件（不在模型目录中的）
        
        Args:
            base_path: 基础扫描路径
            recursive: 是否递归扫描
            scanned_directories: 已扫描的目录集合
            
        Returns:
            List[ModelInfo]: 发现的独立PyTorch模型列表
        """
        models = []
        
        # 获取所有PyTorch模型文件
        pytorch_extensions = ['.bin', '.pt', '.pth', '.safetensors']
        
        for ext in pytorch_extensions:
            if recursive:
                files = list(base_path.rglob(f'*{ext}'))
            else:
                files = list(base_path.glob(f'*{ext}'))
            
            for file_path in files:
                # 检查文件是否在已扫描的模型目录中
                file_parent = str(file_path.parent)
                is_in_model_directory = False
                
                for scanned_dir in scanned_directories:
                    if (file_parent == scanned_dir or 
                        file_parent.startswith(scanned_dir + os.sep) or 
                        file_parent.startswith(scanned_dir + '/')):
                        is_in_model_directory = True
                        break
                
                if is_in_model_directory:
                    continue
                
                # 检查文件所在目录是否包含其他模型相关文件
                # 如果包含，说明这是一个模型目录，应该被目录扫描处理
                parent_files = {f.name.lower() for f in file_path.parent.iterdir() if f.is_file()}
                has_config = bool(parent_files & {f.lower() for f in self.PYTORCH_CONFIG_FILES})
                has_tokenizer = bool(parent_files & {f.lower() for f in self.PYTORCH_TOKENIZER_FILES})
                
                # 如果包含配置文件或tokenizer文件，跳过（应该作为目录处理）
                if has_config or has_tokenizer:
                    self.logger.debug(f"跳过文件 {file_path}，因为其目录包含配置文件，应作为模型目录处理")
                    continue
                
                try:
                    model_info = self._analyze_standalone_pytorch_file(file_path)
                    if model_info:
                        models.append(model_info)
                        self.logger.debug(f"发现独立PyTorch模型文件: {file_path}")
                except Exception as e:
                    self.logger.warning(f"分析独立PyTorch文件 {file_path} 时出错: {str(e)}")
        
        return models
    
    def _analyze_pytorch_directory(self, dir_path: Path) -> Optional[ModelInfo]:
        """
        分析PyTorch模型目录
        
        Args:
            dir_path: 目录路径
            
        Returns:
            ModelInfo: 如果是有效的PyTorch模型目录则返回模型信息，否则返回None
        """
        try:
            # 获取目录中的所有文件（只检查直接子文件，不递归）
            direct_files = []
            for item in dir_path.iterdir():
                if item.is_file():
                    direct_files.append(item)
            
            if not direct_files:
                return None
            
            files_in_dir = {f.name.lower() for f in direct_files}
            
            # 检查是否包含PyTorch模型文件（精确匹配）
            has_exact_model_file = bool(files_in_dir & {f.lower() for f in self.PYTORCH_MODEL_FILES})
            
            # 检查是否包含分片索引文件
            has_sharded_index = bool(files_in_dir & {f.lower() for f in self.PYTORCH_SHARDED_INDEX_FILES})
            
            # 检查是否包含PyTorch模型文件（模式匹配）
            import re
            has_pattern_model_file = False
            pattern_matched_files = []
            
            for pattern in self.PYTORCH_MODEL_PATTERNS:
                for file_name in files_in_dir:
                    if re.match(pattern, file_name, re.IGNORECASE):
                        has_pattern_model_file = True
                        pattern_matched_files.append(file_name)
            
            if not has_exact_model_file and not has_pattern_model_file and not has_sharded_index:
                return None
            
            # 查找主要的模型文件
            main_model_file = None
            found_model_files = []
            
            # 优先查找精确匹配的文件
            for model_file in self.PYTORCH_MODEL_FILES:
                model_file_path = dir_path / model_file
                if model_file_path.exists():
                    found_model_files.append(model_file)
                    if main_model_file is None:
                        main_model_file = model_file_path
            
            # 如果没有精确匹配，使用模式匹配的文件
            if not found_model_files and pattern_matched_files:
                # 按文件名排序，选择第一个作为主文件
                pattern_matched_files.sort()
                for file_name in pattern_matched_files:
                    file_path = dir_path / file_name
                    if file_path.exists():
                        found_model_files.append(file_name)
                        if main_model_file is None:
                            main_model_file = file_path
            
            if not main_model_file or not found_model_files:
                return None
            
            # 计算目录总大小（包括所有子文件和子目录）
            total_size = 0
            file_count = 0
            for item in dir_path.rglob('*'):
                if item.is_file():
                    total_size += item.stat().st_size
                    file_count += 1
            
            # 生成模型ID（使用相对路径，确保唯一性）
            model_id = str(dir_path).replace('\\', '/').replace(os.getcwd().replace('\\', '/'), '').lstrip('/')
            
            # 生成模型名称（使用目录名）
            model_name = dir_path.name
            
            # 收集模型元数据
            metadata = self._collect_pytorch_directory_metadata(dir_path, files_in_dir)
            metadata['found_model_files'] = found_model_files
            metadata['main_model_file'] = main_model_file.name
            metadata['total_file_count'] = file_count
            
            self.logger.info(f"识别PyTorch模型目录: {dir_path.name}, 文件数: {file_count}, 大小: {total_size} bytes")
            
            return ModelInfo(
                id=model_id,
                name=model_name,
                path=str(dir_path),
                model_type=ModelType.PYTORCH,
                size=total_size,
                config=metadata
            )
            
        except Exception as e:
            self.logger.warning(f"分析PyTorch目录 {dir_path} 时出错: {str(e)}")
            return None
    
    def _analyze_gguf_file(self, file_path: Path) -> Optional[ModelInfo]:
        """
        分析GGUF模型文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            ModelInfo: 如果是有效的GGUF模型文件则返回模型信息，否则返回None
        """
        try:
            # 获取文件大小
            file_size = file_path.stat().st_size
            
            # 生成模型ID（使用文件路径）
            model_id = str(file_path).replace('\\', '/')
            
            # 生成模型名称（使用文件名，不包含扩展名）
            model_name = file_path.stem
            
            # 收集模型元数据
            metadata = self._get_gguf_metadata(file_path)
            
            return ModelInfo(
                id=model_id,
                name=model_name,
                path=str(file_path),
                model_type=ModelType.GGUF,
                size=file_size,
                config=metadata
            )
            
        except Exception as e:
            self.logger.warning(f"分析GGUF文件 {file_path} 时出错: {str(e)}")
            return None
    
    def _analyze_standalone_pytorch_file(self, file_path: Path) -> Optional[ModelInfo]:
        """
        分析独立的PyTorch模型文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            ModelInfo: 如果是有效的PyTorch模型文件则返回模型信息，否则返回None
        """
        # 检查文件扩展名
        extension = file_path.suffix.lower()
        if extension not in ['.bin', '.pt', '.pth', '.safetensors']:
            return None
        
        try:
            # 获取文件大小
            file_size = file_path.stat().st_size
            
            # 生成模型ID（使用文件路径）
            model_id = str(file_path).replace('\\', '/')
            
            # 生成模型名称（使用文件名，不包含扩展名）
            model_name = file_path.stem
            
            # 收集模型元数据
            metadata = self._get_pytorch_metadata(file_path)
            metadata['is_standalone'] = True
            
            return ModelInfo(
                id=model_id,
                name=model_name,
                path=str(file_path),
                model_type=ModelType.PYTORCH,
                size=file_size,
                config=metadata
            )
            
        except Exception as e:
            self.logger.warning(f"分析独立PyTorch文件 {file_path} 时出错: {str(e)}")
            return None

    def _analyze_file(self, file_path: Path) -> Optional[ModelInfo]:
        """
        分析单个文件，判断是否为支持的模型文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            ModelInfo: 如果是支持的模型文件则返回模型信息，否则返回None
        """
        # 检查文件扩展名
        extension = file_path.suffix.lower()
        if extension not in self.SUPPORTED_EXTENSIONS:
            return None
        
        try:
            # 获取文件大小
            file_size = file_path.stat().st_size
            
            # 确定模型类型
            model_type = self.SUPPORTED_EXTENSIONS[extension]
            
            # 生成模型ID（使用相对路径作为唯一标识）
            model_id = str(file_path).replace('\\', '/')
            
            # 生成模型名称（使用文件名，不包含扩展名）
            model_name = file_path.stem
            
            # 收集模型元数据
            metadata = self._collect_metadata(file_path, model_type)
            
            return ModelInfo(
                id=model_id,
                name=model_name,
                path=str(file_path),
                model_type=model_type,
                size=file_size,
                config=metadata
            )
            
        except Exception as e:
            self.logger.warning(f"分析文件 {file_path} 时出错: {str(e)}")
            return None
    
    def _collect_pytorch_directory_metadata(self, dir_path: Path, files_in_dir: set) -> Dict[str, any]:
        """
        收集PyTorch模型目录的元数据
        
        Args:
            dir_path: 目录路径
            files_in_dir: 目录中的文件名集合（小写）
            
        Returns:
            Dict: 包含元数据的字典
        """
        metadata = {
            'framework': 'pytorch',
            'model_type': 'directory',
            'directory_name': dir_path.name,
            'absolute_path': str(dir_path.absolute()),
            'relative_path': str(dir_path),
            'is_standalone': False,
            'supports_streaming': True,
            'requires_tokenizer': True,
        }
        
        # 检查配置文件
        has_config = bool(files_in_dir & {f.lower() for f in self.PYTORCH_CONFIG_FILES})
        metadata['has_config'] = has_config
        
        # 检查tokenizer文件
        has_tokenizer = bool(files_in_dir & {f.lower() for f in self.PYTORCH_TOKENIZER_FILES})
        metadata['has_tokenizer'] = has_tokenizer
        
        # 检查分片索引文件
        sharded_index_files = []
        for index_file in self.PYTORCH_SHARDED_INDEX_FILES:
            if index_file.lower() in files_in_dir:
                sharded_index_files.append(index_file)
        
        metadata['sharded_index_files'] = sharded_index_files
        metadata['is_sharded'] = len(sharded_index_files) > 0
        
        # 检查模型文件格式
        model_files = []
        for model_file in self.PYTORCH_MODEL_FILES:
            if model_file.lower() in files_in_dir:
                model_files.append(model_file)
        
        metadata['model_files'] = model_files
        
        # 如果是分片模型，分析索引文件获取分片信息
        if metadata['is_sharded'] and sharded_index_files:
            sharded_info = self._analyze_sharded_model_index(dir_path, sharded_index_files[0])
            metadata.update(sharded_info)
        
        # 确定主要格式
        if any('safetensors' in f for f in model_files) or any('safetensors' in f for f in sharded_index_files):
            metadata['format'] = 'safetensors'
            metadata['memory_efficient'] = True
        else:
            metadata['format'] = 'pytorch'
            metadata['memory_efficient'] = False
        
        # 尝试读取config.json获取更多信息
        config_path = dir_path / 'config.json'
        if config_path.exists():
            try:
                import json
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                # 提取有用的配置信息
                if 'model_type' in config_data:
                    metadata['model_architecture'] = config_data['model_type']
                if 'vocab_size' in config_data:
                    metadata['vocab_size'] = config_data['vocab_size']
                if 'hidden_size' in config_data:
                    metadata['hidden_size'] = config_data['hidden_size']
                if 'num_attention_heads' in config_data:
                    metadata['num_attention_heads'] = config_data['num_attention_heads']
                if 'num_hidden_layers' in config_data:
                    metadata['num_hidden_layers'] = config_data['num_hidden_layers']
                if 'max_position_embeddings' in config_data:
                    metadata['max_position_embeddings'] = config_data['max_position_embeddings']
                    
            except Exception as e:
                self.logger.debug(f"读取config.json失败: {e}")
        
        # 统计文件数量和类型
        all_files = list(dir_path.rglob('*'))
        file_count = len([f for f in all_files if f.is_file()])
        metadata['file_count'] = file_count
        
        return metadata

    def _analyze_sharded_model_index(self, dir_path: Path, index_file: str) -> Dict[str, any]:
        """
        分析分片模型的索引文件
        
        Args:
            dir_path: 模型目录路径
            index_file: 索引文件名
            
        Returns:
            Dict: 分片模型的元数据
        """
        sharded_info = {
            'shard_count': 0,
            'shard_files': [],
            'total_parameters': 0,
            'total_size': 0,
            'weight_map_entries': 0
        }
        
        try:
            index_path = dir_path / index_file
            if not index_path.exists():
                return sharded_info
            
            import json
            with open(index_path, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            
            # 提取元数据信息
            if 'metadata' in index_data:
                metadata = index_data['metadata']
                if 'total_parameters' in metadata:
                    sharded_info['total_parameters'] = metadata['total_parameters']
                if 'total_size' in metadata:
                    sharded_info['total_size'] = metadata['total_size']
            
            # 提取权重映射信息
            if 'weight_map' in index_data:
                weight_map = index_data['weight_map']
                sharded_info['weight_map_entries'] = len(weight_map)
                
                # 获取所有分片文件
                shard_files = set(weight_map.values())
                sharded_info['shard_files'] = sorted(list(shard_files))
                sharded_info['shard_count'] = len(shard_files)
                
                # 验证分片文件是否存在
                existing_shards = []
                missing_shards = []
                for shard_file in shard_files:
                    shard_path = dir_path / shard_file
                    if shard_path.exists():
                        existing_shards.append(shard_file)
                    else:
                        missing_shards.append(shard_file)
                
                sharded_info['existing_shards'] = existing_shards
                sharded_info['missing_shards'] = missing_shards
                sharded_info['shards_complete'] = len(missing_shards) == 0
            
            self.logger.debug(f"分析分片模型索引: {index_file}, 分片数: {sharded_info['shard_count']}")
            
        except Exception as e:
            self.logger.warning(f"分析分片模型索引文件 {index_file} 失败: {e}")
        
        return sharded_info

    def _collect_metadata(self, file_path: Path, model_type: ModelType) -> Dict[str, any]:
        """
        收集模型文件的元数据
        
        Args:
            file_path: 文件路径
            model_type: 模型类型
            
        Returns:
            Dict: 包含元数据的字典
        """
        metadata = {
            'file_extension': file_path.suffix.lower(),
            'file_name': file_path.name,
            'parent_directory': file_path.parent.name,
            'absolute_path': str(file_path.absolute()),
            'relative_path': str(file_path),
        }
        
        # 根据模型类型添加特定的元数据
        if model_type == ModelType.PYTORCH:
            metadata.update(self._get_pytorch_metadata(file_path))
        elif model_type == ModelType.GGUF:
            metadata.update(self._get_gguf_metadata(file_path))
        
        return metadata
    
    def _get_pytorch_metadata(self, file_path: Path) -> Dict[str, any]:
        """获取PyTorch模型的特定元数据"""
        metadata = {
            'framework': 'pytorch',
            'supports_streaming': True,
            'requires_tokenizer': True,
        }
        
        # 检查是否为safetensors格式
        if file_path.suffix.lower() == '.safetensors':
            metadata['format'] = 'safetensors'
            metadata['memory_efficient'] = True
        else:
            metadata['format'] = 'pytorch'
            metadata['memory_efficient'] = False
        
        return metadata
    
    def _get_gguf_metadata(self, file_path: Path) -> Dict[str, any]:
        """获取GGUF模型的特定元数据"""
        metadata = {
            'framework': 'gguf',
            'format': 'gguf',
            'supports_streaming': True,
            'requires_tokenizer': False,
            'cpu_optimized': True,
        }
        
        # 尝试从文件名推断量化信息
        filename_lower = file_path.name.lower()
        if 'q4_0' in filename_lower:
            metadata['quantization'] = 'Q4_0'
        elif 'q4_1' in filename_lower:
            metadata['quantization'] = 'Q4_1'
        elif 'q5_0' in filename_lower:
            metadata['quantization'] = 'Q5_0'
        elif 'q5_1' in filename_lower:
            metadata['quantization'] = 'Q5_1'
        elif 'q8_0' in filename_lower:
            metadata['quantization'] = 'Q8_0'
        elif 'f16' in filename_lower:
            metadata['quantization'] = 'F16'
        elif 'f32' in filename_lower:
            metadata['quantization'] = 'F32'
        else:
            metadata['quantization'] = 'unknown'
        
        return metadata
    
    def get_supported_extensions(self) -> Set[str]:
        """获取支持的文件扩展名列表"""
        return set(self.SUPPORTED_EXTENSIONS.keys())
    
    def is_supported_file(self, file_path: str) -> bool:
        """检查文件是否为支持的模型格式"""
        path = Path(file_path)
        return path.suffix.lower() in self.SUPPORTED_EXTENSIONS