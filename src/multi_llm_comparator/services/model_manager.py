"""
模型管理器

负责管理本地模型的发现、选择和配置。
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Set
from dataclasses import asdict

from ..core.models import ModelInfo, ModelType
from ..core.exceptions import ModelSelectionError, ModelNotFoundError
from .model_scanner import ModelFileScanner, ScanResult
from .api_model_manager import APIModelManager


logger = logging.getLogger(__name__)


class ModelManager:
    """模型管理器类"""
    
    MAX_SELECTED_MODELS = 4  # 最多同时选择4个模型
    
    def __init__(self, cache_file: Optional[str] = None):
        """
        初始化模型管理器
        
        Args:
            cache_file: 模型缓存文件路径，如果为None则使用默认路径
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.scanner = ModelFileScanner()
        self.api_manager = APIModelManager()
        
        # 设置缓存文件路径
        if cache_file is None:
            cache_file = "models_cache.json"
        self.cache_file = Path(cache_file)
        
        # 内部状态
        self._available_models: Dict[str, ModelInfo] = {}
        self._selected_model_ids: Set[str] = set()
        self._last_scan_directories: List[str] = []
        
        # 尝试加载缓存
        self._load_cache()
        
        # 加载API模型
        self._load_api_models()
    
    def scan_models(self, directories: List[str], recursive: bool = True, 
                   force_rescan: bool = False) -> ScanResult:
        """
        扫描指定目录中的模型文件
        
        Args:
            directories: 要扫描的目录列表
            recursive: 是否递归扫描子目录
            force_rescan: 是否强制重新扫描（忽略缓存）
            
        Returns:
            ScanResult: 扫描结果
        """
        # 检查是否需要重新扫描
        if not force_rescan and directories == self._last_scan_directories:
            self.logger.info("使用缓存的模型列表，跳过扫描")
            return ScanResult(
                models=list(self._available_models.values()),
                errors=[],
                scanned_files=len(self._available_models),
                valid_models=len(self._available_models)
            )
        
        self.logger.info(f"开始扫描模型目录: {directories}")
        
        all_models = []
        all_errors = []
        total_scanned_files = 0
        
        # 扫描每个目录
        for directory in directories:
            try:
                result = self.scanner.scan_directory(directory, recursive)
                all_models.extend(result.models)
                all_errors.extend(result.errors)
                total_scanned_files += result.scanned_files
                
                self.logger.info(f"目录 {directory} 扫描完成，发现 {result.valid_models} 个模型")
                
            except Exception as e:
                error_msg = f"扫描目录 {directory} 时发生错误: {str(e)}"
                all_errors.append(error_msg)
                self.logger.error(error_msg)
        
        # 更新内部状态
        self._available_models.clear()
        for model in all_models:
            self._available_models[model.id] = model
        
        self._last_scan_directories = directories.copy()
        
        # 清理无效的选中模型
        self._cleanup_invalid_selections()
        
        # 保存缓存
        self._save_cache()
        
        scan_result = ScanResult(
            models=all_models,
            errors=all_errors,
            scanned_files=total_scanned_files,
            valid_models=len(all_models)
        )
        
        self.logger.info(f"模型扫描完成，总共发现 {len(all_models)} 个有效模型")
        return scan_result
    
    def get_available_models(self) -> List[ModelInfo]:
        """
        获取所有可用的模型列表（包括本地模型和API模型）
        
        Returns:
            List[ModelInfo]: 可用模型列表
        """
        all_models = list(self._available_models.values())
        api_models = self.api_manager.get_api_models()
        all_models.extend(api_models)
        return all_models
    
    def get_model_by_id(self, model_id: str) -> Optional[ModelInfo]:
        """
        根据ID获取模型信息（包括本地模型和API模型）
        
        Args:
            model_id: 模型ID
            
        Returns:
            ModelInfo: 模型信息，如果不存在则返回None
        """
        # 先查找本地模型
        model = self._available_models.get(model_id)
        if model:
            return model
        
        # 再查找API模型
        api_models = self.api_manager.get_api_models()
        for api_model in api_models:
            if api_model.id == model_id:
                return api_model
        
        return None
    
    def get_models_by_type(self, model_type: ModelType) -> List[ModelInfo]:
        """
        根据类型获取模型列表
        
        Args:
            model_type: 模型类型
            
        Returns:
            List[ModelInfo]: 指定类型的模型列表
        """
        return [model for model in self._available_models.values() 
                if model.model_type == model_type]
    
    def select_models(self, model_ids: List[str]) -> None:
        """
        选择要比较的模型
        
        Args:
            model_ids: 要选择的模型ID列表
            
        Raises:
            ModelSelectionError: 当选择的模型数量超过限制或模型不存在时
        """
        # 验证模型数量限制
        if len(model_ids) > self.MAX_SELECTED_MODELS:
            raise ModelSelectionError(
                f"最多只能选择 {self.MAX_SELECTED_MODELS} 个模型，当前选择了 {len(model_ids)} 个"
            )
        
        # 验证所有模型都存在（包括本地模型和API模型）
        missing_models = []
        for model_id in model_ids:
            if self.get_model_by_id(model_id) is None:
                missing_models.append(model_id)
        
        if missing_models:
            raise ModelNotFoundError(f"以下模型不存在: {missing_models}")
        
        # 检查选择是否发生了变化
        old_selection = self._selected_model_ids.copy()
        new_selection = set(model_ids)
        
        # 更新选中的模型
        self._selected_model_ids = new_selection
        
        # 只在选择发生变化时记录日志
        if old_selection != new_selection:
            self.logger.info(f"已选择 {len(model_ids)} 个模型进行比较")
        
        # 保存缓存
        self._save_cache()
    
    def add_selected_model(self, model_id: str) -> None:
        """
        添加一个模型到选中列表
        
        Args:
            model_id: 要添加的模型ID
            
        Raises:
            ModelSelectionError: 当选择的模型数量超过限制或模型不存在时
        """
        if len(self._selected_model_ids) >= self.MAX_SELECTED_MODELS:
            raise ModelSelectionError(
                f"已达到最大选择数量限制 ({self.MAX_SELECTED_MODELS})"
            )
        
        if self.get_model_by_id(model_id) is None:
            raise ModelNotFoundError(f"模型不存在: {model_id}")
        
        self._selected_model_ids.add(model_id)
        self.logger.info(f"已添加模型到选中列表: {model_id}")
        
        # 保存缓存
        self._save_cache()
    
    def remove_selected_model(self, model_id: str) -> None:
        """
        从选中列表中移除一个模型
        
        Args:
            model_id: 要移除的模型ID
        """
        self._selected_model_ids.discard(model_id)
        self.logger.info(f"已从选中列表移除模型: {model_id}")
        
        # 保存缓存
        self._save_cache()
    
    def clear_selected_models(self) -> None:
        """清空选中的模型列表"""
        self._selected_model_ids.clear()
        self.logger.info("已清空选中的模型列表")
        
        # 保存缓存
        self._save_cache()
    
    def get_selected_models(self) -> List[ModelInfo]:
        """
        获取当前选中的模型列表
        
        Returns:
            List[ModelInfo]: 选中的模型列表
        """
        selected_models = []
        for model_id in self._selected_model_ids:
            # 使用get_model_by_id来同时查找本地模型和API模型
            model = self.get_model_by_id(model_id)
            if model:
                # 确保模型有配置，如果没有则使用默认配置
                if not model.config:
                    from ..core.config import ConfigManager
                    config_manager = ConfigManager()
                    model.config = config_manager.get_model_config(model_id).__dict__
                selected_models.append(model)
        
        return selected_models
    
    def get_selected_model_ids(self) -> List[str]:
        """
        获取当前选中的模型ID列表
        
        Returns:
            List[str]: 选中的模型ID列表
        """
        return list(self._selected_model_ids)
    
    def is_model_selected(self, model_id: str) -> bool:
        """
        检查模型是否已被选中
        
        Args:
            model_id: 模型ID
            
        Returns:
            bool: 如果模型已被选中则返回True
        """
        return model_id in self._selected_model_ids
    
    def get_selection_count(self) -> int:
        """
        获取当前选中的模型数量
        
        Returns:
            int: 选中的模型数量
        """
        return len(self._selected_model_ids)
    
    def can_select_more_models(self) -> bool:
        """
        检查是否还可以选择更多模型
        
        Returns:
            bool: 如果还可以选择更多模型则返回True
        """
        return len(self._selected_model_ids) < self.MAX_SELECTED_MODELS
    
    def get_model_statistics(self) -> Dict[str, any]:
        """
        获取模型统计信息
        
        Returns:
            Dict: 包含统计信息的字典
        """
        all_models = self.get_available_models()
        pytorch_count = len([m for m in all_models if m.model_type == ModelType.PYTORCH])
        gguf_count = len([m for m in all_models if m.model_type == ModelType.GGUF])
        api_count = len([m for m in all_models if m.model_type == ModelType.OPENAI_API])
        
        # 只计算本地模型的大小
        total_size = sum(model.size for model in self._available_models.values())
        
        return {
            'total_models': len(all_models),
            'pytorch_models': pytorch_count,
            'gguf_models': gguf_count,
            'api_models': api_count,
            'selected_models': len(self._selected_model_ids),
            'max_selection': self.MAX_SELECTED_MODELS,
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'total_size_gb': round(total_size / (1024 * 1024 * 1024), 2),
        }
    
    def _load_api_models(self) -> None:
        """加载API模型"""
        try:
            api_models = self.api_manager.get_api_models()
            self.logger.info(f"加载了 {len(api_models)} 个API模型")
        except Exception as e:
            self.logger.error(f"加载API模型失败: {str(e)}")
    
    def add_doubao_model(self, model_id: str, model_name: str, display_name: Optional[str] = None, 
                        api_key: Optional[str] = None, base_url: Optional[str] = None) -> ModelInfo:
        """
        添加Doubao模型
        
        Args:
            model_id: 模型ID
            model_name: 模型名称
            display_name: 显示名称
            api_key: API密钥（可选，留空则使用环境变量）
            base_url: API基础URL（可选）
            
        Returns:
            ModelInfo: 创建的模型信息
        """
        return self.api_manager.add_doubao_model(model_id, model_name, display_name, api_key, base_url)
    
    def add_openai_compatible_model(self, model_id: str, model_name: str, display_name: Optional[str] = None, 
                                  api_key: Optional[str] = None, base_url: Optional[str] = None, 
                                  provider: str = "OpenAI Compatible") -> ModelInfo:
        """
        添加OpenAI兼容模型
        
        Args:
            model_id: 模型ID
            model_name: 模型名称
            display_name: 显示名称
            api_key: API密钥（可选，留空则使用环境变量）
            base_url: API基础URL（可选）
            provider: 提供商名称
            
        Returns:
            ModelInfo: 创建的模型信息
        """
        return self.api_manager.add_openai_compatible_model(model_id, model_name, display_name, api_key, base_url, provider)
    
    def _cleanup_invalid_selections(self) -> None:
        """清理无效的选中模型（不存在的模型）"""
        valid_selections = set()
        for model_id in self._selected_model_ids:
            # 使用get_model_by_id来检查本地模型和API模型
            if self.get_model_by_id(model_id) is not None:
                valid_selections.add(model_id)
            else:
                self.logger.warning(f"清理无效的选中模型: {model_id}")
        
        self._selected_model_ids = valid_selections
    
    def _save_cache(self) -> None:
        """保存模型缓存到文件"""
        try:
            # 转换模型数据，处理枚举类型
            available_models_data = {}
            for model_id, model in self._available_models.items():
                model_dict = asdict(model)
                model_dict['model_type'] = model.model_type.value  # 转换枚举为字符串
                available_models_data[model_id] = model_dict
            
            cache_data = {
                'available_models': available_models_data,
                'selected_model_ids': list(self._selected_model_ids),
                'last_scan_directories': self._last_scan_directories,
            }
            
            # 确保目录存在
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            self.logger.debug(f"模型缓存已保存到: {self.cache_file}")
            
        except Exception as e:
            self.logger.error(f"保存模型缓存失败: {str(e)}")
    
    def _load_cache(self) -> None:
        """从文件加载模型缓存"""
        try:
            if not self.cache_file.exists():
                self.logger.debug("缓存文件不存在，使用空缓存")
                return
            
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # 恢复可用模型
            self._available_models.clear()
            for model_id, model_data in cache_data.get('available_models', {}).items():
                # 重建ModelInfo对象
                model_data['model_type'] = ModelType(model_data['model_type'])
                model_info = ModelInfo(**model_data)
                self._available_models[model_id] = model_info
            
            # 恢复选中的模型
            self._selected_model_ids = set(cache_data.get('selected_model_ids', []))
            
            # 恢复扫描目录
            self._last_scan_directories = cache_data.get('last_scan_directories', [])
            
            # 清理无效的选中模型
            self._cleanup_invalid_selections()
            
            self.logger.info(f"从缓存加载了 {len(self._available_models)} 个模型")
            
        except Exception as e:
            self.logger.error(f"加载模型缓存失败: {str(e)}")
            # 重置为空状态
            self._available_models.clear()
            self._selected_model_ids.clear()
            self._last_scan_directories.clear()
    
    def clear_cache(self) -> None:
        """清空缓存文件"""
        try:
            if self.cache_file.exists():
                self.cache_file.unlink()
                self.logger.info("模型缓存文件已删除")
        except Exception as e:
            self.logger.error(f"删除缓存文件失败: {str(e)}")
    
    def refresh_models(self, directories: List[str], recursive: bool = True) -> ScanResult:
        """
        刷新模型列表（强制重新扫描）
        
        Args:
            directories: 要扫描的目录列表
            recursive: 是否递归扫描子目录
            
        Returns:
            ScanResult: 扫描结果
        """
        return self.scan_models(directories, recursive, force_rescan=True)