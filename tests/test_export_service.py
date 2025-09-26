"""
导出服务测试

测试结果导出功能的各种格式和场景。
"""

import pytest
import json
import csv
from datetime import datetime
from io import StringIO
from unittest.mock import patch, MagicMock

from src.multi_llm_comparator.services.export_service import ExportService


class TestExportService:
    """导出服务测试类"""
    
    def setup_method(self):
        """测试前设置"""
        self.export_service = ExportService()
        
        # 创建测试数据
        self.sample_results = {
            'model_1': {
                'model_name': 'Test Model 1',
                'model_type': 'pytorch',
                'status': 'completed',
                'content': 'This is a test response from model 1.\n\nIt has multiple lines.',
                'error': None,
                'stats': {
                    'start_time': 1640995200.0,  # 2022-01-01 00:00:00
                    'end_time': 1640995205.5,    # 2022-01-01 00:00:05.5
                    'duration': 5.5,
                    'token_count': 15
                }
            },
            'model_2': {
                'model_name': 'Test Model 2',
                'model_type': 'gguf',
                'status': 'completed',
                'content': 'This is another test response.\n\n```python\nprint("Hello World")\n```',
                'error': None,
                'stats': {
                    'start_time': 1640995210.0,
                    'end_time': 1640995213.2,
                    'duration': 3.2,
                    'token_count': 12
                }
            },
            'model_3': {
                'model_name': 'Test Model 3',
                'model_type': 'pytorch',
                'status': 'error',
                'content': '',
                'error': 'Model loading failed',
                'stats': None
            }
        }
        
        self.empty_results = {}
    
    def test_init(self):
        """测试初始化"""
        assert self.export_service.supported_formats == ['json', 'csv', 'markdown', 'txt']
    
    def test_export_comparison_results_invalid_format(self):
        """测试不支持的导出格式"""
        with pytest.raises(ValueError, match="不支持的导出格式"):
            self.export_service.export_comparison_results(
                self.sample_results, 'xml'
            )
    
    def test_export_to_json_with_metadata_and_stats(self):
        """测试JSON导出（包含元数据和统计信息）"""
        with patch('src.multi_llm_comparator.services.export_service.datetime') as mock_datetime:
            mock_datetime.now.return_value.isoformat.return_value = '2022-01-01T12:00:00'
            
            result = self.export_service.export_comparison_results(
                self.sample_results, 'json', True, True
            )
        
        # 验证JSON格式
        data = json.loads(result)
        
        # 验证元数据
        assert data['export_info']['format'] == 'json'
        assert data['export_info']['timestamp'] == '2022-01-01T12:00:00'
        assert data['export_info']['total_models'] == 3
        assert data['export_info']['version'] == '1.0'
        
        # 验证结果数据
        assert len(data['comparison_results']) == 3
        
        # 验证第一个模型
        model_1 = data['comparison_results']['model_1']
        assert model_1['model_name'] == 'Test Model 1'
        assert model_1['model_type'] == 'pytorch'
        assert model_1['status'] == 'completed'
        assert 'statistics' in model_1
        assert model_1['statistics']['duration'] == 5.5
        
        # 验证错误模型
        model_3 = data['comparison_results']['model_3']
        assert model_3['error'] == 'Model loading failed'
        assert 'statistics' not in model_3  # 没有统计信息
    
    def test_export_to_json_without_metadata_and_stats(self):
        """测试JSON导出（不包含元数据和统计信息）"""
        result = self.export_service.export_comparison_results(
            self.sample_results, 'json', False, False
        )
        
        data = json.loads(result)
        
        # 验证没有元数据
        assert data['export_info'] == {}
        
        # 验证没有统计信息
        model_1 = data['comparison_results']['model_1']
        assert 'statistics' not in model_1
    
    def test_export_to_csv_with_metadata_and_stats(self):
        """测试CSV导出（包含元数据和统计信息）"""
        with patch('src.multi_llm_comparator.services.export_service.datetime') as mock_datetime:
            mock_datetime.now.return_value.isoformat.return_value = '2022-01-01T12:00:00'
            mock_datetime.now.return_value.strftime.return_value = '2022-01-01 12:00:00'
            
            result = self.export_service.export_comparison_results(
                self.sample_results, 'csv', True, True
            )
        
        lines = result.strip().split('\n')
        
        # 验证元数据注释
        assert lines[0].startswith('# 多LLM模型比较结果导出')
        assert '# 导出时间: 2022-01-01T12:00:00' in lines[1]
        assert '# 总模型数: 3' in lines[2]
        
        # 验证CSV内容
        csv_content = '\n'.join([line for line in lines if not line.startswith('#') and line.strip()])
        reader = csv.DictReader(StringIO(csv_content))
        rows = list(reader)
        
        assert len(rows) == 3
        
        # 验证第一行数据
        row1 = rows[0]
        assert row1['model_id'] == 'model_1'
        assert row1['model_name'] == 'Test Model 1'
        assert row1['model_type'] == 'pytorch'
        assert row1['status'] == 'completed'
        assert row1['duration'] == '5.5'
        assert row1['token_count'] == '15'
        assert float(row1['tokens_per_second']) == pytest.approx(15/5.5, rel=1e-2)
    
    def test_export_to_csv_without_stats(self):
        """测试CSV导出（不包含统计信息）"""
        result = self.export_service.export_comparison_results(
            self.sample_results, 'csv', False, False
        )
        
        # 解析CSV
        lines = [line for line in result.split('\n') if not line.startswith('#') and line.strip()]
        reader = csv.DictReader(StringIO('\n'.join(lines)))
        rows = list(reader)
        
        # 验证字段不包含统计信息
        fieldnames = reader.fieldnames
        stats_fields = ['start_time', 'end_time', 'duration', 'token_count', 'tokens_per_second']
        for field in stats_fields:
            assert field not in fieldnames
    
    def test_export_to_markdown_with_metadata_and_stats(self):
        """测试Markdown导出（包含元数据和统计信息）"""
        with patch('src.multi_llm_comparator.services.export_service.datetime') as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = '2022-01-01 12:00:00'
            
            result = self.export_service.export_comparison_results(
                self.sample_results, 'markdown', True, True
            )
        
        lines = result.split('\n')
        
        # 验证标题
        assert '# 多LLM模型比较结果报告' in lines
        
        # 验证元数据
        assert '## 导出信息' in result
        assert '- **导出时间**: 2022-01-01 12:00:00' in result
        assert '- **总模型数**: 3' in result
        
        # 验证摘要表格
        assert '## 结果摘要' in result
        assert '| 总模型数 | 3 |' in result
        assert '| 成功完成 | 2 |' in result
        assert '| 出现错误 | 1 |' in result
        
        # 验证性能统计表格
        assert '## 性能统计' in result
        assert '| Test Model 1 | PYTORCH | completed | 5.5 | 15 |' in result
        
        # 验证详细结果
        assert '## 详细结果' in result
        assert '### 1. Test Model 1 (PYTORCH)' in result
        assert '**错误**: Model loading failed' in result  # 错误模型
    
    def test_export_to_txt_format(self):
        """测试TXT格式导出"""
        with patch('src.multi_llm_comparator.services.export_service.datetime') as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = '2022-01-01 12:00:00'
            
            result = self.export_service.export_comparison_results(
                self.sample_results, 'txt', True, True
            )
        
        lines = result.split('\n')
        
        # 验证标题
        assert '多LLM模型比较结果报告' in lines
        assert '=' * 60 in lines
        
        # 验证导出信息
        assert '导出时间: 2022-01-01 12:00:00' in result
        assert '总模型数: 3' in result
        
        # 验证结果摘要
        assert '成功完成: 2' in result
        assert '出现错误: 1' in result
        assert '成功率: 66.7%' in result
        
        # 验证详细结果
        assert '1. Test Model 1 (PYTORCH)' in result
        assert '   状态: completed' in result
        assert '   用时: 5.5秒' in result
        assert '   错误: Model loading failed' in result  # 错误模型
    
    def test_export_empty_results(self):
        """测试导出空结果"""
        formats = ['json', 'csv', 'markdown', 'txt']
        
        for format_type in formats:
            result = self.export_service.export_comparison_results(
                self.empty_results, format_type, True, True
            )
            
            # 验证不为空
            assert result.strip() != ''
            
            # 验证包含相关信息
            if format_type == 'json':
                data = json.loads(result)
                assert data['export_info']['total_models'] == 0
                assert data['message'] == '暂无比较结果'
            else:
                assert '暂无比较结果' in result or '总模型数: 0' in result
    
    def test_calculate_tokens_per_second(self):
        """测试每秒token数计算"""
        # 正常情况
        stats = {'duration': 5.0, 'token_count': 20}
        result = self.export_service._calculate_tokens_per_second(stats)
        assert result == 4.0
        
        # 零除法
        stats = {'duration': 0, 'token_count': 20}
        result = self.export_service._calculate_tokens_per_second(stats)
        assert result == 0.0
        
        # 无效数据
        stats = {'duration': 'invalid', 'token_count': 20}
        result = self.export_service._calculate_tokens_per_second(stats)
        assert result == 0.0
        
        # 缺少字段
        stats = {}
        result = self.export_service._calculate_tokens_per_second(stats)
        assert result == 0.0
    
    def test_format_timestamp(self):
        """测试时间戳格式化"""
        # 正常时间戳
        timestamp = 1640995200.0  # 2022-01-01 00:00:00 UTC
        result = self.export_service._format_timestamp(timestamp)
        # 注意：结果会根据本地时区而变化，这里只验证格式
        assert len(result) == 19  # YYYY-MM-DD HH:MM:SS
        assert '-' in result and ':' in result
        
        # None值
        result = self.export_service._format_timestamp(None)
        assert result == 'N/A'
        
        # 无效时间戳
        result = self.export_service._format_timestamp('invalid')
        assert result == 'N/A'
    
    def test_get_filename(self):
        """测试文件名生成"""
        with patch('src.multi_llm_comparator.services.export_service.datetime') as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = '20220101_120000'
            
            # 默认前缀
            filename = self.export_service.get_filename('json')
            assert filename == 'llm_comparison_20220101_120000.json'
            
            # 自定义前缀
            filename = self.export_service.get_filename('csv', 'custom_export')
            assert filename == 'custom_export_20220101_120000.csv'
    
    def test_get_mime_type(self):
        """测试MIME类型获取"""
        assert self.export_service.get_mime_type('json') == 'application/json'
        assert self.export_service.get_mime_type('csv') == 'text/csv'
        assert self.export_service.get_mime_type('markdown') == 'text/markdown'
        assert self.export_service.get_mime_type('txt') == 'text/plain'
        assert self.export_service.get_mime_type('unknown') == 'text/plain'
    
    def test_export_with_special_characters(self):
        """测试包含特殊字符的导出"""
        special_results = {
            'model_1': {
                'model_name': 'Model with 中文 and émojis 🤖',
                'model_type': 'pytorch',
                'status': 'completed',
                'content': 'Response with special chars: "quotes", <tags>, & symbols',
                'error': None,
                'stats': {
                    'start_time': 1640995200.0,
                    'end_time': 1640995205.0,
                    'duration': 5.0,
                    'token_count': 10
                }
            }
        }
        
        # 测试JSON导出
        json_result = self.export_service.export_comparison_results(
            special_results, 'json', True, True
        )
        data = json.loads(json_result)
        assert data['comparison_results']['model_1']['model_name'] == 'Model with 中文 and émojis 🤖'
        
        # 测试CSV导出
        csv_result = self.export_service.export_comparison_results(
            special_results, 'csv', True, True
        )
        assert 'Model with 中文 and émojis 🤖' in csv_result
        
        # 测试Markdown导出
        md_result = self.export_service.export_comparison_results(
            special_results, 'markdown', True, True
        )
        assert 'Model with 中文 and émojis 🤖' in md_result
    
    def test_export_large_content(self):
        """测试大内容导出"""
        large_content = "This is a very long response. " * 1000  # 约30KB内容
        
        large_results = {
            'model_1': {
                'model_name': 'Large Content Model',
                'model_type': 'pytorch',
                'status': 'completed',
                'content': large_content,
                'error': None,
                'stats': {
                    'start_time': 1640995200.0,
                    'end_time': 1640995300.0,
                    'duration': 100.0,
                    'token_count': 5000
                }
            }
        }
        
        # 测试各种格式都能处理大内容
        for format_type in ['json', 'csv', 'markdown', 'txt']:
            result = self.export_service.export_comparison_results(
                large_results, format_type, True, True
            )
            # CSV格式只包含元数据，不包含完整内容，所以长度较小
            if format_type == 'csv':
                assert len(result) > 200  # CSV只包含统计信息
                assert '5000' in result  # token数量应该在CSV中
            else:
                assert len(result) > 30000  # 其他格式包含完整内容
            assert 'Large Content Model' in result
    
    def test_export_error_handling(self):
        """测试导出过程中的错误处理"""
        # 创建包含None值的异常数据
        problematic_results = {
            'model_1': {
                'model_name': None,
                'model_type': None,
                'status': None,
                'content': None,
                'error': None,
                'stats': {
                    'start_time': None,
                    'end_time': None,
                    'duration': None,
                    'token_count': None
                }
            }
        }
        
        # 测试各种格式的错误处理
        for format_type in ['json', 'csv', 'markdown', 'txt']:
            try:
                result = self.export_service.export_comparison_results(
                    problematic_results, format_type, True, True
                )
                # 应该能够处理None值而不崩溃
                assert result is not None
                assert len(result) > 0
            except Exception as e:
                pytest.fail(f"导出格式 {format_type} 处理异常数据时失败: {str(e)}")


if __name__ == '__main__':
    pytest.main([__file__])