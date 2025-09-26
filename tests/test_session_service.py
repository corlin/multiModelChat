"""
会话管理服务测试

测试会话保存、加载、管理等功能。
"""

import pytest
import json
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

from src.multi_llm_comparator.services.session_service import SessionService


class TestSessionService:
    """会话管理服务测试类"""
    
    def setup_method(self):
        """测试前设置"""
        # 创建临时目录用于测试
        self.temp_dir = tempfile.mkdtemp()
        self.sessions_dir = Path(self.temp_dir) / "test_sessions"
        
        self.session_service = SessionService(str(self.sessions_dir))
        
        # 创建测试数据
        self.sample_results = {
            'model_1': {
                'model_name': 'Test Model 1',
                'model_type': 'pytorch',
                'status': 'completed',
                'content': 'This is a test response from model 1.',
                'error': None,
                'stats': {
                    'start_time': 1640995200.0,
                    'end_time': 1640995205.5,
                    'duration': 5.5,
                    'token_count': 15
                }
            },
            'model_2': {
                'model_name': 'Test Model 2',
                'model_type': 'gguf',
                'status': 'error',
                'content': '',
                'error': 'Model loading failed',
                'stats': None
            }
        }
        
        self.sample_prompt = "What is artificial intelligence?"
        self.sample_models_info = [
            {
                'id': 'model_1',
                'name': 'Test Model 1',
                'type': 'pytorch',
                'path': '/path/to/model1',
                'size': 1024000
            },
            {
                'id': 'model_2',
                'name': 'Test Model 2',
                'type': 'gguf',
                'path': '/path/to/model2',
                'size': 2048000
            }
        ]
    
    def teardown_method(self):
        """测试后清理"""
        if Path(self.temp_dir).exists():
            # 在Windows上，需要先修改只读目录的权限
            try:
                # 递归修改权限
                for root, dirs, files in os.walk(self.temp_dir):
                    for d in dirs:
                        os.chmod(os.path.join(root, d), 0o777)
                    for f in files:
                        os.chmod(os.path.join(root, f), 0o777)
                shutil.rmtree(self.temp_dir)
            except Exception:
                # 如果仍然失败，尝试强制删除
                try:
                    import subprocess
                    subprocess.run(['rmdir', '/s', '/q', self.temp_dir], shell=True, check=False)
                except Exception:
                    pass  # 忽略清理失败
    
    def test_init(self):
        """测试初始化"""
        assert self.session_service.sessions_dir.exists()
        assert self.session_service.backup_dir.exists()
        assert self.session_service.max_sessions == 100
    
    def test_save_session_success(self):
        """测试成功保存会话"""
        with patch('src.multi_llm_comparator.services.session_service.datetime') as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = '20220101_120000_000000'
            mock_datetime.now.return_value.isoformat.return_value = '2022-01-01T12:00:00'
            
            session_id = self.session_service.save_session(
                results=self.sample_results,
                prompt=self.sample_prompt,
                models_info=self.sample_models_info,
                session_name="Test Session"
            )
        
        assert session_id == '20220101_120000_000000'
        
        # 验证文件是否创建
        session_file = self.sessions_dir / f"session_{session_id}.json"
        assert session_file.exists()
        
        # 验证文件内容
        with open(session_file, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
        
        assert session_data['session_info']['id'] == session_id
        assert session_data['session_info']['name'] == "Test Session"
        assert session_data['session_info']['prompt'] == self.sample_prompt
        assert session_data['comparison_results'] == self.sample_results
        assert session_data['models_info'] == self.sample_models_info
        
        # 验证统计信息
        stats = session_data['statistics']
        assert stats['total_models'] == 2
        assert stats['completed_models'] == 1
        assert stats['error_models'] == 1
        assert stats['success_rate'] == 50.0
    
    def test_save_session_auto_name(self):
        """测试自动生成会话名称"""
        with patch('src.multi_llm_comparator.services.session_service.datetime') as mock_datetime:
            mock_datetime.now.return_value.strftime.side_effect = ['20220101_120000_000000', '2022-01-01 12:00:00']
            mock_datetime.now.return_value.isoformat.return_value = '2022-01-01T12:00:00'
            
            session_id = self.session_service.save_session(
                results=self.sample_results,
                prompt=self.sample_prompt,
                models_info=self.sample_models_info
            )
        
        # 验证自动生成的名称
        session_file = self.sessions_dir / f"session_{session_id}.json"
        with open(session_file, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
        
        assert session_data['session_info']['name'] == "比较会话 2022-01-01 12:00:00"
    
    def test_save_session_io_error(self):
        """测试保存会话时的IO错误"""
        # 模拟文件写入失败的情况
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            with pytest.raises(IOError, match="保存会话失败"):
                self.session_service.save_session(
                    results=self.sample_results,
                    prompt=self.sample_prompt,
                    models_info=self.sample_models_info
                )
    
    def test_load_session_success(self):
        """测试成功加载会话"""
        # 先保存一个会话
        session_id = self.session_service.save_session(
            results=self.sample_results,
            prompt=self.sample_prompt,
            models_info=self.sample_models_info,
            session_name="Test Session"
        )
        
        # 加载会话
        session_data = self.session_service.load_session(session_id)
        
        # 验证加载的数据
        assert session_data['session_info']['name'] == "Test Session"
        assert session_data['session_info']['prompt'] == self.sample_prompt
        assert session_data['comparison_results'] == self.sample_results
        assert session_data['models_info'] == self.sample_models_info
    
    def test_load_session_not_found(self):
        """测试加载不存在的会话"""
        with pytest.raises(FileNotFoundError, match="会话不存在"):
            self.session_service.load_session("nonexistent_session")
    
    def test_load_session_invalid_json(self):
        """测试加载无效JSON格式的会话"""
        # 创建一个无效的JSON文件
        invalid_session_file = self.sessions_dir / "session_invalid.json"
        with open(invalid_session_file, 'w') as f:
            f.write("invalid json content")
        
        with pytest.raises(ValueError, match="会话数据格式错误"):
            self.session_service.load_session("invalid")
    
    def test_load_session_missing_fields(self):
        """测试加载缺少必要字段的会话"""
        # 创建一个缺少字段的会话文件
        incomplete_data = {"session_info": {}}  # 缺少其他必要字段
        
        incomplete_session_file = self.sessions_dir / "session_incomplete.json"
        with open(incomplete_session_file, 'w', encoding='utf-8') as f:
            json.dump(incomplete_data, f)
        
        with pytest.raises(ValueError, match="会话数据缺少必要字段"):
            self.session_service.load_session("incomplete")
    
    def test_list_sessions(self):
        """测试列出会话"""
        # 保存几个会话
        session_ids = []
        for i in range(3):
            with patch('src.multi_llm_comparator.services.session_service.datetime') as mock_datetime:
                mock_datetime.now.return_value.strftime.return_value = f'2022010{i+1}_120000'
                mock_datetime.now.return_value.isoformat.return_value = f'2022-01-0{i+1}T12:00:00'
                
                session_id = self.session_service.save_session(
                    results=self.sample_results,
                    prompt=f"Test prompt {i+1}",
                    models_info=self.sample_models_info,
                    session_name=f"Test Session {i+1}"
                )
                session_ids.append(session_id)
        
        # 列出会话
        sessions = self.session_service.list_sessions()
        
        assert len(sessions) == 3
        
        # 验证会话信息
        for i, session in enumerate(sessions):
            assert 'id' in session
            assert 'name' in session
            assert 'created_at' in session
            assert 'prompt_preview' in session
            assert session['total_models'] == 2
            assert session['completed_models'] == 1
            assert session['error_models'] == 1
    
    def test_list_sessions_with_limit(self):
        """测试限制会话列表数量"""
        # 保存5个会话
        for i in range(5):
            with patch('src.multi_llm_comparator.services.session_service.datetime') as mock_datetime:
                mock_datetime.now.return_value.strftime.return_value = f'2022010{i+1}_120000'
                mock_datetime.now.return_value.isoformat.return_value = f'2022-01-0{i+1}T12:00:00'
                
                self.session_service.save_session(
                    results=self.sample_results,
                    prompt=f"Test prompt {i+1}",
                    models_info=self.sample_models_info
                )
        
        # 限制返回3个
        sessions = self.session_service.list_sessions(limit=3)
        assert len(sessions) == 3
    
    def test_list_sessions_empty(self):
        """测试列出空会话列表"""
        sessions = self.session_service.list_sessions()
        assert sessions == []
    
    def test_delete_session_success(self):
        """测试成功删除会话"""
        # 保存一个会话
        session_id = self.session_service.save_session(
            results=self.sample_results,
            prompt=self.sample_prompt,
            models_info=self.sample_models_info
        )
        
        # 验证文件存在
        session_file = self.sessions_dir / f"session_{session_id}.json"
        assert session_file.exists()
        
        # 删除会话
        result = self.session_service.delete_session(session_id)
        assert result is True
        
        # 验证文件已移动到备份目录
        assert not session_file.exists()
        backup_files = list(self.session_service.backup_dir.glob(f"deleted_{session_id}_*.json"))
        assert len(backup_files) == 1
    
    def test_delete_session_not_found(self):
        """测试删除不存在的会话"""
        result = self.session_service.delete_session("nonexistent")
        assert result is False
    
    def test_clear_all_sessions(self):
        """测试清空所有会话"""
        # 保存几个会话
        session_ids = []
        for i in range(3):
            # Add small delay to ensure unique timestamps
            import time
            time.sleep(0.001)
            session_id = self.session_service.save_session(
                results=self.sample_results,
                prompt=f"Test prompt {i+1}",
                models_info=self.sample_models_info
            )
            session_ids.append(session_id)
        
        # 验证会话存在
        assert len(list(self.sessions_dir.glob("session_*.json"))) == 3
        
        # 清空所有会话
        result = self.session_service.clear_all_sessions(create_backup=True)
        assert result is True
        
        # 验证会话已清空
        assert len(list(self.sessions_dir.glob("session_*.json"))) == 0
        
        # 验证备份已创建
        backup_dirs = list(self.session_service.backup_dir.glob("batch_clear_*"))
        assert len(backup_dirs) == 1
        
        backup_files = list(backup_dirs[0].glob("session_*.json"))
        assert len(backup_files) == 3
    
    def test_clear_all_sessions_empty(self):
        """测试清空空会话列表"""
        result = self.session_service.clear_all_sessions()
        assert result is True
    
    def test_backup_sessions(self):
        """测试备份会话"""
        # 保存几个会话
        for i in range(2):
            # Add small delay to ensure unique timestamps
            import time
            time.sleep(0.001)
            self.session_service.save_session(
                results=self.sample_results,
                prompt=f"Test prompt {i+1}",
                models_info=self.sample_models_info
            )
        
        # 创建备份
        with patch('src.multi_llm_comparator.services.session_service.datetime') as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = '20220101_120000'
            mock_datetime.now.return_value.isoformat.return_value = '2022-01-01T12:00:00'
            
            backup_path = self.session_service.backup_sessions("test_backup")
        
        # 验证备份目录
        backup_dir = Path(backup_path)
        assert backup_dir.exists()
        assert backup_dir.name == "test_backup"
        
        # 验证备份文件
        backup_files = list(backup_dir.glob("session_*.json"))
        assert len(backup_files) == 2
        
        # 验证备份信息文件
        backup_info_file = backup_dir / "backup_info.json"
        assert backup_info_file.exists()
        
        with open(backup_info_file, 'r', encoding='utf-8') as f:
            backup_info = json.load(f)
        
        assert backup_info['backup_name'] == "test_backup"
        assert backup_info['total_sessions'] == 2
    
    def test_backup_sessions_empty(self):
        """测试备份空会话"""
        backup_path = self.session_service.backup_sessions()
        
        backup_dir = Path(backup_path)
        assert backup_dir.exists()
        
        # 应该没有会话文件，但有备份信息文件
        backup_files = list(backup_dir.glob("session_*.json"))
        assert len(backup_files) == 0
    
    def test_restore_sessions(self):
        """测试恢复会话"""
        # 创建备份数据
        backup_dir = self.session_service.backup_dir / "test_restore"
        backup_dir.mkdir()
        
        # 在备份目录中创建会话文件
        test_session_data = {
            "session_info": {"id": "test_restore", "name": "Restored Session"},
            "models_info": [],
            "comparison_results": {}
        }
        
        backup_session_file = backup_dir / "session_test_restore.json"
        with open(backup_session_file, 'w', encoding='utf-8') as f:
            json.dump(test_session_data, f)
        
        # 恢复会话
        result = self.session_service.restore_sessions(str(backup_dir))
        assert result is True
        
        # 验证会话已恢复
        restored_file = self.sessions_dir / "session_test_restore.json"
        assert restored_file.exists()
        
        # 验证恢复的数据
        with open(restored_file, 'r', encoding='utf-8') as f:
            restored_data = json.load(f)
        
        assert restored_data['session_info']['name'] == "Restored Session"
    
    def test_restore_sessions_not_found(self):
        """测试恢复不存在的备份"""
        with pytest.raises(FileNotFoundError, match="备份目录不存在"):
            self.session_service.restore_sessions("/nonexistent/path")
    
    def test_restore_sessions_overwrite(self):
        """测试恢复会话时覆盖现有文件"""
        # 先保存一个会话
        original_session_id = self.session_service.save_session(
            results=self.sample_results,
            prompt="Original prompt",
            models_info=self.sample_models_info,
            session_name="Original Session"
        )
        
        # 创建备份数据（相同的session_id但不同内容）
        backup_dir = self.session_service.backup_dir / "test_overwrite"
        backup_dir.mkdir()
        
        modified_session_data = {
            "session_info": {"id": original_session_id, "name": "Modified Session"},
            "models_info": [],
            "comparison_results": {}
        }
        
        backup_session_file = backup_dir / f"session_{original_session_id}.json"
        with open(backup_session_file, 'w', encoding='utf-8') as f:
            json.dump(modified_session_data, f)
        
        # 不覆盖恢复
        result = self.session_service.restore_sessions(str(backup_dir), overwrite=False)
        assert result is True
        
        # 验证原文件未被覆盖
        session_data = self.session_service.load_session(original_session_id)
        assert session_data['session_info']['name'] == "Original Session"
        
        # 覆盖恢复
        result = self.session_service.restore_sessions(str(backup_dir), overwrite=True)
        assert result is True
        
        # 验证文件已被覆盖
        session_data = self.session_service.load_session(original_session_id)
        assert session_data['session_info']['name'] == "Modified Session"
    
    def test_get_session_statistics(self):
        """测试获取会话统计信息"""
        # 保存几个会话
        for i in range(3):
            # Add small delay to ensure unique timestamps
            import time
            time.sleep(0.001)
            self.session_service.save_session(
                results=self.sample_results,
                prompt=f"Test prompt {i+1}",
                models_info=self.sample_models_info
            )
        
        stats = self.session_service.get_session_statistics()
        
        assert stats['total_sessions'] == 3
        assert stats['total_size_mb'] > 0
        assert stats['average_models_per_session'] == 2.0
        assert stats['sessions_with_errors'] == 3  # 所有会话都有错误模型
        assert 'latest_session' in stats
        assert 'oldest_session' in stats
    
    def test_get_session_statistics_empty(self):
        """测试获取空会话统计"""
        stats = self.session_service.get_session_statistics()
        
        assert stats['total_sessions'] == 0
        assert stats['total_size_mb'] == 0
        assert stats['latest_session'] is None
        assert stats['oldest_session'] is None
    
    def test_cleanup_old_sessions(self):
        """测试清理旧会话"""
        # 设置较小的最大会话数
        self.session_service.max_sessions = 2
        
        # 保存3个会话
        session_ids = []
        for i in range(3):
            # Add small delay to ensure unique timestamps
            import time
            time.sleep(0.001)
            session_id = self.session_service.save_session(
                results=self.sample_results,
                prompt=f"Test prompt {i+1}",
                models_info=self.sample_models_info
            )
            session_ids.append(session_id)
        
        # 验证只保留了2个会话
        remaining_sessions = list(self.sessions_dir.glob("session_*.json"))
        assert len(remaining_sessions) == 2
        
        # 验证最旧的会话被移动到备份目录
        backup_files = list(self.session_service.backup_dir.glob("auto_cleanup_*.json"))
        assert len(backup_files) == 1
    
    def test_truncate_text(self):
        """测试文本截断"""
        # 正常文本
        short_text = "Short text"
        result = self.session_service._truncate_text(short_text, 20)
        assert result == short_text
        
        # 长文本
        long_text = "This is a very long text that should be truncated"
        result = self.session_service._truncate_text(long_text, 20)
        assert result == "This is a very lo..."
        assert len(result) == 20
        
        # 空文本
        result = self.session_service._truncate_text("", 10)
        assert result == ""
        
        # None文本
        result = self.session_service._truncate_text(None, 10)
        assert result == ""
    
    def test_calculate_session_statistics(self):
        """测试计算会话统计信息"""
        # 测试正常结果
        stats = self.session_service._calculate_session_statistics(self.sample_results)
        
        assert stats['total_models'] == 2
        assert stats['completed_models'] == 1
        assert stats['error_models'] == 1
        assert stats['success_rate'] == 50.0
        assert stats['total_tokens'] == 15
        assert stats['average_duration'] == 5.5
        
        # 测试空结果
        empty_stats = self.session_service._calculate_session_statistics({})
        
        assert empty_stats['total_models'] == 0
        assert empty_stats['completed_models'] == 0
        assert empty_stats['error_models'] == 0
        assert empty_stats['success_rate'] == 0.0
    
    def test_calculate_average_duration(self):
        """测试计算平均用时"""
        # 正常情况
        results_with_duration = {
            'model_1': {'stats': {'duration': 5.0}},
            'model_2': {'stats': {'duration': 3.0}},
            'model_3': {'stats': {'duration': 7.0}}
        }
        
        avg_duration = self.session_service._calculate_average_duration(results_with_duration)
        assert avg_duration == 5.0
        
        # 包含None值
        results_with_none = {
            'model_1': {'stats': {'duration': 5.0}},
            'model_2': {'stats': {'duration': None}},
            'model_3': {'stats': {}}
        }
        
        avg_duration = self.session_service._calculate_average_duration(results_with_none)
        assert avg_duration == 5.0
        
        # 全部为None或无效
        results_invalid = {
            'model_1': {'stats': {'duration': None}},
            'model_2': {'stats': {}}
        }
        
        avg_duration = self.session_service._calculate_average_duration(results_invalid)
        assert avg_duration == 0.0


if __name__ == '__main__':
    pytest.main([__file__])