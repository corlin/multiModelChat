"""
内存管理服务

优化内存使用和模型加载策略。
"""

import gc
import psutil
from typing import Dict, Any


class MemoryManager:
    """内存管理器"""
    
    def __init__(self):
        """初始化内存管理器"""
        self.process = psutil.Process()
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """获取当前内存使用情况"""
        memory_info = self.process.memory_info()
        return {
            "rss": memory_info.rss,  # 物理内存
            "vms": memory_info.vms,  # 虚拟内存
            "percent": self.process.memory_percent(),
            "available": psutil.virtual_memory().available
        }
    
    def force_cleanup(self) -> None:
        """强制清理内存"""
        gc.collect()
        
        # 如果使用PyTorch，清理GPU缓存
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    
    def check_memory_availability(self, required_mb: int) -> bool:
        """检查是否有足够的可用内存"""
        available_mb = psutil.virtual_memory().available / (1024 * 1024)
        return available_mb > required_mb