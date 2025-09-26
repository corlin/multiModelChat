"""
会话管理服务

提供比较历史的本地存储和管理功能。
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class SessionService:
    """会话管理服务类"""
    
    def __init__(self, sessions_dir: str = ".sessions"):
        """
        初始化会话管理服务
        
        Args:
            sessions_dir: 会话存储目录
        """
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(exist_ok=True)
        
        # 会话文件名格式
        self.session_file_pattern = "session_{timestamp}.json"
        self.backup_dir = self.sessions_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        
        # 最大会话数量（防止磁盘空间耗尽）
        self.max_sessions = 100
    
    def save_session(
        self,
        results: Dict[str, Dict[str, Any]],
        prompt: str,
        models_info: List[Dict[str, Any]],
        session_name: Optional[str] = None
    ) -> str:
        """
        保存比较会话
        
        Args:
            results: 比较结果
            prompt: 使用的提示词
            models_info: 模型信息列表
            session_name: 自定义会话名称
            
        Returns:
            会话ID
            
        Raises:
            IOError: 保存失败时抛出
        """
        try:
            timestamp = datetime.now()
            session_id = timestamp.strftime("%Y%m%d_%H%M%S_%f")  # Add microseconds for uniqueness
            
            # 创建会话数据
            session_data = {
                "session_info": {
                    "id": session_id,
                    "name": session_name or f"比较会话 {timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
                    "created_at": timestamp.isoformat(),
                    "prompt": prompt,
                    "total_models": len(results),
                    "version": "1.0"
                },
                "models_info": models_info,
                "comparison_results": results,
                "statistics": self._calculate_session_statistics(results)
            }
            
            # 保存到文件
            session_file = self.sessions_dir / f"session_{session_id}.json"
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
            
            # 清理旧会话（如果超过最大数量）
            self._cleanup_old_sessions()
            
            logger.info(f"会话已保存: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"保存会话失败: {str(e)}")
            raise IOError(f"保存会话失败: {str(e)}")
    
    def load_session(self, session_id: str) -> Dict[str, Any]:
        """
        加载比较会话
        
        Args:
            session_id: 会话ID
            
        Returns:
            会话数据
            
        Raises:
            FileNotFoundError: 会话不存在时抛出
            ValueError: 会话数据无效时抛出
        """
        try:
            session_file = self.sessions_dir / f"session_{session_id}.json"
            
            if not session_file.exists():
                raise FileNotFoundError(f"会话不存在: {session_id}")
            
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            # 验证会话数据结构
            required_keys = ['session_info', 'models_info', 'comparison_results']
            for key in required_keys:
                if key not in session_data:
                    raise ValueError(f"会话数据缺少必要字段: {key}")
            
            logger.info(f"会话已加载: {session_id}")
            return session_data
            
        except json.JSONDecodeError as e:
            logger.error(f"会话数据格式错误: {str(e)}")
            raise ValueError(f"会话数据格式错误: {str(e)}")
        except Exception as e:
            logger.error(f"加载会话失败: {str(e)}")
            raise
    
    def list_sessions(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        列出所有会话
        
        Args:
            limit: 限制返回的会话数量
            
        Returns:
            会话信息列表
        """
        try:
            sessions = []
            
            # 获取所有会话文件
            session_files = list(self.sessions_dir.glob("session_*.json"))
            
            # 按修改时间排序（最新的在前）
            session_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # 限制数量
            if limit:
                session_files = session_files[:limit]
            
            for session_file in session_files:
                try:
                    with open(session_file, 'r', encoding='utf-8') as f:
                        session_data = json.load(f)
                    
                    # 提取会话摘要信息
                    session_info = session_data.get('session_info', {})
                    statistics = session_data.get('statistics', {})
                    
                    sessions.append({
                        "id": session_info.get('id', session_file.stem.replace('session_', '')),
                        "name": session_info.get('name', '未命名会话'),
                        "created_at": session_info.get('created_at'),
                        "prompt_preview": self._truncate_text(session_info.get('prompt', ''), 100),
                        "total_models": session_info.get('total_models', 0),
                        "completed_models": statistics.get('completed_models', 0),
                        "error_models": statistics.get('error_models', 0),
                        "file_size": session_file.stat().st_size
                    })
                    
                except Exception as e:
                    logger.warning(f"跳过损坏的会话文件 {session_file}: {str(e)}")
                    continue
            
            return sessions
            
        except Exception as e:
            logger.error(f"列出会话失败: {str(e)}")
            return []
    
    def delete_session(self, session_id: str) -> bool:
        """
        删除会话
        
        Args:
            session_id: 会话ID
            
        Returns:
            是否删除成功
        """
        try:
            session_file = self.sessions_dir / f"session_{session_id}.json"
            
            if not session_file.exists():
                logger.warning(f"要删除的会话不存在: {session_id}")
                return False
            
            # 移动到备份目录而不是直接删除
            backup_file = self.backup_dir / f"deleted_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            shutil.move(str(session_file), str(backup_file))
            
            logger.info(f"会话已删除并备份: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"删除会话失败: {str(e)}")
            return False
    
    def clear_all_sessions(self, create_backup: bool = True) -> bool:
        """
        清除所有会话
        
        Args:
            create_backup: 是否创建备份
            
        Returns:
            是否清除成功
        """
        try:
            session_files = list(self.sessions_dir.glob("session_*.json"))
            
            if not session_files:
                logger.info("没有会话需要清除")
                return True
            
            if create_backup:
                # 创建批量备份
                backup_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                batch_backup_dir = self.backup_dir / f"batch_clear_{backup_timestamp}"
                batch_backup_dir.mkdir(exist_ok=True)
                
                for session_file in session_files:
                    backup_file = batch_backup_dir / session_file.name
                    shutil.copy2(str(session_file), str(backup_file))
            
            # 删除所有会话文件
            for session_file in session_files:
                session_file.unlink()
            
            logger.info(f"已清除 {len(session_files)} 个会话")
            return True
            
        except Exception as e:
            logger.error(f"清除会话失败: {str(e)}")
            return False
    
    def backup_sessions(self, backup_name: Optional[str] = None) -> str:
        """
        备份所有会话
        
        Args:
            backup_name: 备份名称
            
        Returns:
            备份目录路径
            
        Raises:
            IOError: 备份失败时抛出
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = backup_name or f"sessions_backup_{timestamp}"
            backup_path = self.backup_dir / backup_name
            
            # 创建备份目录
            backup_path.mkdir(exist_ok=True)
            
            # 复制所有会话文件
            session_files = list(self.sessions_dir.glob("session_*.json"))
            
            if not session_files:
                logger.warning("没有会话需要备份")
                return str(backup_path)
            
            for session_file in session_files:
                backup_file = backup_path / session_file.name
                shutil.copy2(str(session_file), str(backup_file))
            
            # 创建备份信息文件
            backup_info = {
                "backup_name": backup_name,
                "created_at": datetime.now().isoformat(),
                "total_sessions": len(session_files),
                "sessions": [f.name for f in session_files]
            }
            
            with open(backup_path / "backup_info.json", 'w', encoding='utf-8') as f:
                json.dump(backup_info, f, ensure_ascii=False, indent=2)
            
            logger.info(f"会话备份完成: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"备份会话失败: {str(e)}")
            raise IOError(f"备份会话失败: {str(e)}")
    
    def restore_sessions(self, backup_path: str, overwrite: bool = False) -> bool:
        """
        恢复会话备份
        
        Args:
            backup_path: 备份目录路径
            overwrite: 是否覆盖现有会话
            
        Returns:
            是否恢复成功
        """
        try:
            backup_dir = Path(backup_path)
            
            if not backup_dir.exists():
                raise FileNotFoundError(f"备份目录不存在: {backup_path}")
            
            # 获取备份中的会话文件
            backup_sessions = list(backup_dir.glob("session_*.json"))
            
            if not backup_sessions:
                logger.warning("备份目录中没有会话文件")
                return True
            
            restored_count = 0
            skipped_count = 0
            
            for backup_session in backup_sessions:
                target_file = self.sessions_dir / backup_session.name
                
                # 检查是否已存在
                if target_file.exists() and not overwrite:
                    skipped_count += 1
                    continue
                
                # 复制会话文件
                shutil.copy2(str(backup_session), str(target_file))
                restored_count += 1
            
            logger.info(f"会话恢复完成: 恢复 {restored_count} 个，跳过 {skipped_count} 个")
            return True
            
        except FileNotFoundError:
            # Re-raise FileNotFoundError so tests can catch it
            raise
        except Exception as e:
            logger.error(f"恢复会话失败: {str(e)}")
            return False
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """
        获取会话统计信息
        
        Returns:
            统计信息字典
        """
        try:
            sessions = self.list_sessions()
            
            if not sessions:
                return {
                    "total_sessions": 0,
                    "total_size_mb": 0,
                    "latest_session": None,
                    "oldest_session": None
                }
            
            # 计算总大小
            total_size = sum(session['file_size'] for session in sessions)
            
            # 按创建时间排序
            sessions_by_time = sorted(sessions, key=lambda x: x['created_at'] or '')
            
            return {
                "total_sessions": len(sessions),
                "total_size_mb": total_size / (1024 * 1024),
                "latest_session": sessions_by_time[-1] if sessions_by_time else None,
                "oldest_session": sessions_by_time[0] if sessions_by_time else None,
                "average_models_per_session": sum(s['total_models'] for s in sessions) / len(sessions),
                "sessions_with_errors": len([s for s in sessions if s['error_models'] > 0])
            }
            
        except Exception as e:
            logger.error(f"获取会话统计失败: {str(e)}")
            return {}
    
    def _calculate_session_statistics(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """计算会话统计信息"""
        if not results:
            return {
                "total_models": 0,
                "completed_models": 0,
                "error_models": 0,
                "success_rate": 0.0
            }
        
        completed_models = len([r for r in results.values() if r.get('status') == 'completed'])
        error_models = len([r for r in results.values() if r.get('error')])
        
        # 计算总token数
        total_tokens = 0
        for result in results.values():
            stats = result.get('stats') or {}
            if isinstance(stats, dict):
                token_count = stats.get('token_count', 0)
                if token_count is not None:
                    total_tokens += int(token_count)
        
        return {
            "total_models": len(results),
            "completed_models": completed_models,
            "error_models": error_models,
            "success_rate": (completed_models / len(results)) * 100 if results else 0.0,
            "average_duration": self._calculate_average_duration(results),
            "total_tokens": total_tokens
        }
    
    def _calculate_average_duration(self, results: Dict[str, Dict[str, Any]]) -> float:
        """计算平均用时"""
        durations = []
        for result in results.values():
            stats = result.get('stats') or {}
            if isinstance(stats, dict):
                duration = stats.get('duration')
                if duration is not None and duration > 0:
                    durations.append(float(duration))
        
        return sum(durations) / len(durations) if durations else 0.0
    
    def _cleanup_old_sessions(self):
        """清理旧会话（保持最大数量限制）"""
        try:
            session_files = list(self.sessions_dir.glob("session_*.json"))
            
            if len(session_files) <= self.max_sessions:
                return
            
            # 按修改时间排序，删除最旧的
            session_files.sort(key=lambda x: x.stat().st_mtime)
            
            files_to_delete = session_files[:-self.max_sessions]
            
            for file_to_delete in files_to_delete:
                # 移动到备份目录
                backup_file = self.backup_dir / f"auto_cleanup_{file_to_delete.name}"
                shutil.move(str(file_to_delete), str(backup_file))
            
            logger.info(f"自动清理了 {len(files_to_delete)} 个旧会话")
            
        except Exception as e:
            logger.warning(f"清理旧会话失败: {str(e)}")
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        """截断文本"""
        if not text:
            return ""
        
        if len(text) <= max_length:
            return text
        
        return text[:max_length - 3] + "..."