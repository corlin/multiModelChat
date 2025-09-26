"""
结果导出服务

提供多种格式的比较结果导出功能。
"""

import json
import csv
from datetime import datetime
from typing import Dict, Any, List, Optional
from io import StringIO
import logging

from ..core.models import InferenceResult, ModelInfo, ModelType

logger = logging.getLogger(__name__)


class ExportService:
    """结果导出服务类"""
    
    def __init__(self):
        """初始化导出服务"""
        self.supported_formats = ['json', 'csv', 'markdown', 'txt']
    
    def export_comparison_results(
        self,
        results: Dict[str, Dict[str, Any]],
        format_type: str,
        include_metadata: bool = True,
        include_stats: bool = True
    ) -> str:
        """
        导出比较结果
        
        Args:
            results: 比较结果字典
            format_type: 导出格式 ('json', 'csv', 'markdown', 'txt')
            include_metadata: 是否包含元数据
            include_stats: 是否包含统计信息
            
        Returns:
            导出的内容字符串
            
        Raises:
            ValueError: 不支持的导出格式
        """
        if format_type not in self.supported_formats:
            raise ValueError(f"不支持的导出格式: {format_type}")
        
        if not results:
            return self._generate_empty_export(format_type)
        
        try:
            if format_type == 'json':
                return self._export_to_json(results, include_metadata, include_stats)
            elif format_type == 'csv':
                return self._export_to_csv(results, include_metadata, include_stats)
            elif format_type == 'markdown':
                return self._export_to_markdown(results, include_metadata, include_stats)
            elif format_type == 'txt':
                return self._export_to_txt(results, include_metadata, include_stats)
        except Exception as e:
            logger.error(f"导出失败 ({format_type}): {str(e)}")
            raise
    
    def _export_to_json(
        self,
        results: Dict[str, Dict[str, Any]],
        include_metadata: bool,
        include_stats: bool
    ) -> str:
        """导出为JSON格式"""
        export_data = {
            "export_info": {
                "format": "json",
                "timestamp": datetime.now().isoformat(),
                "total_models": len(results),
                "version": "1.0"
            } if include_metadata else {},
            "comparison_results": {}
        }
        
        for model_id, result in results.items():
            model_data = {
                "model_name": result.get('model_name') or 'Unknown',
                "model_type": result.get('model_type') or 'unknown',
                "status": result.get('status') or 'unknown',
                "content": result.get('content') or '',
                "error": result.get('error')
            }
            
            if include_stats and result.get('stats'):
                model_data["statistics"] = result['stats']
            
            export_data["comparison_results"][model_id] = model_data
        
        return json.dumps(export_data, ensure_ascii=False, indent=2)
    
    def _export_to_csv(
        self,
        results: Dict[str, Dict[str, Any]],
        include_metadata: bool,
        include_stats: bool
    ) -> str:
        """导出为CSV格式"""
        output = StringIO()
        
        # 定义CSV字段
        fieldnames = [
            'model_id',
            'model_name', 
            'model_type',
            'status',
            'content_length',
            'has_error',
            'error_message'
        ]
        
        if include_stats:
            fieldnames.extend([
                'start_time',
                'end_time', 
                'duration',
                'token_count',
                'tokens_per_second'
            ])
        
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        
        # 写入元数据注释
        if include_metadata:
            output.write(f"# 多LLM模型比较结果导出\n")
            output.write(f"# 导出时间: {datetime.now().isoformat()}\n")
            output.write(f"# 总模型数: {len(results)}\n")
            output.write(f"# 格式版本: 1.0\n")
            output.write("\n")
        
        # 写入表头
        writer.writeheader()
        
        # 写入数据行
        for model_id, result in results.items():
            content = result.get('content') or ''
            row = {
                'model_id': model_id,
                'model_name': result.get('model_name') or 'Unknown',
                'model_type': result.get('model_type') or 'unknown',
                'status': result.get('status') or 'unknown',
                'content_length': len(content),
                'has_error': bool(result.get('error')),
                'error_message': result.get('error') or ''
            }
            
            if include_stats and result.get('stats'):
                stats = result['stats']
                row.update({
                    'start_time': stats.get('start_time', ''),
                    'end_time': stats.get('end_time', ''),
                    'duration': stats.get('duration', 0),
                    'token_count': stats.get('token_count', 0),
                    'tokens_per_second': self._calculate_tokens_per_second(stats)
                })
            elif include_stats:
                # 填充空值
                row.update({
                    'start_time': '',
                    'end_time': '',
                    'duration': 0,
                    'token_count': 0,
                    'tokens_per_second': 0
                })
            
            writer.writerow(row)
        
        return output.getvalue()
    
    def _export_to_markdown(
        self,
        results: Dict[str, Dict[str, Any]],
        include_metadata: bool,
        include_stats: bool
    ) -> str:
        """导出为Markdown格式"""
        lines = []
        
        # 标题和元数据
        lines.append("# 多LLM模型比较结果报告")
        lines.append("")
        
        if include_metadata:
            lines.append("## 导出信息")
            lines.append("")
            lines.append(f"- **导出时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"- **总模型数**: {len(results)}")
            lines.append(f"- **格式版本**: 1.0")
            lines.append("")
        
        # 结果摘要
        completed_count = len([r for r in results.values() if r.get('status') == 'completed'])
        error_count = len([r for r in results.values() if r.get('error')])
        
        lines.append("## 结果摘要")
        lines.append("")
        lines.append(f"| 指标 | 数值 |")
        lines.append(f"|------|------|")
        lines.append(f"| 总模型数 | {len(results)} |")
        lines.append(f"| 成功完成 | {completed_count} |")
        lines.append(f"| 出现错误 | {error_count} |")
        lines.append(f"| 成功率 | {(completed_count/len(results)*100):.1f}% |")
        lines.append("")
        
        # 性能统计
        if include_stats:
            lines.append("## 性能统计")
            lines.append("")
            lines.append("| 模型 | 类型 | 状态 | 用时(s) | Token数 | 速度(t/s) |")
            lines.append("|------|------|------|---------|---------|-----------|")
            
            for model_id, result in results.items():
                model_name = result.get('model_name') or 'Unknown'
                model_type = (result.get('model_type') or 'unknown').upper()
                status = result.get('status') or 'unknown'
                
                if result.get('stats'):
                    stats = result['stats']
                    duration = stats.get('duration') or 0
                    token_count = stats.get('token_count') or 0
                    tps = self._calculate_tokens_per_second(stats)
                    
                    # 确保数值不为None
                    duration = float(duration) if duration is not None else 0.0
                    token_count = int(token_count) if token_count is not None else 0
                    tps = float(tps) if tps is not None else 0.0
                    
                    lines.append(f"| {model_name} | {model_type} | {status} | {duration:.1f} | {token_count} | {tps:.1f} |")
                else:
                    lines.append(f"| {model_name} | {model_type} | {status} | N/A | N/A | N/A |")
            
            lines.append("")
        
        # 详细结果
        lines.append("## 详细结果")
        lines.append("")
        
        for i, (model_id, result) in enumerate(results.items(), 1):
            model_name = result.get('model_name') or 'Unknown'
            model_type = (result.get('model_type') or 'unknown').upper()
            status = result.get('status') or 'unknown'
            
            lines.append(f"### {i}. {model_name} ({model_type})")
            lines.append("")
            lines.append(f"**状态**: {status}")
            lines.append("")
            
            # 错误信息
            if result.get('error'):
                lines.append(f"**错误**: {result['error']}")
                lines.append("")
            
            # 输出内容
            content = result.get('content') or ''
            if content:
                lines.append("**输出内容**:")
                lines.append("")
                lines.append("```")
                lines.append(content)
                lines.append("```")
                lines.append("")
            
            # 统计信息
            if include_stats and result.get('stats'):
                stats = result['stats']
                duration = stats.get('duration') or 0
                token_count = stats.get('token_count') or 0
                tps = self._calculate_tokens_per_second(stats)
                
                # 确保数值不为None
                duration = float(duration) if duration is not None else 0.0
                token_count = int(token_count) if token_count is not None else 0
                tps = float(tps) if tps is not None else 0.0
                
                lines.append("**性能统计**:")
                lines.append("")
                lines.append(f"- 开始时间: {self._format_timestamp(stats.get('start_time'))}")
                lines.append(f"- 结束时间: {self._format_timestamp(stats.get('end_time'))}")
                lines.append(f"- 用时: {duration:.1f}秒")
                lines.append(f"- Token数: {token_count}")
                lines.append(f"- 生成速度: {tps:.1f} tokens/秒")
                lines.append("")
            
            lines.append("---")
            lines.append("")
        
        return "\n".join(lines)
    
    def _export_to_txt(
        self,
        results: Dict[str, Dict[str, Any]],
        include_metadata: bool,
        include_stats: bool
    ) -> str:
        """导出为纯文本格式"""
        lines = []
        
        # 标题和元数据
        lines.append("=" * 60)
        lines.append("多LLM模型比较结果报告")
        lines.append("=" * 60)
        lines.append("")
        
        if include_metadata:
            lines.append("导出信息:")
            lines.append(f"  导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"  总模型数: {len(results)}")
            lines.append(f"  格式版本: 1.0")
            lines.append("")
        
        # 结果摘要
        completed_count = len([r for r in results.values() if r.get('status') == 'completed'])
        error_count = len([r for r in results.values() if r.get('error')])
        
        lines.append("结果摘要:")
        lines.append(f"  总模型数: {len(results)}")
        lines.append(f"  成功完成: {completed_count}")
        lines.append(f"  出现错误: {error_count}")
        lines.append(f"  成功率: {(completed_count/len(results)*100):.1f}%")
        lines.append("")
        
        # 详细结果
        lines.append("详细结果:")
        lines.append("-" * 60)
        
        for i, (model_id, result) in enumerate(results.items(), 1):
            model_name = result.get('model_name') or 'Unknown'
            model_type = (result.get('model_type') or 'unknown').upper()
            status = result.get('status') or 'unknown'
            
            lines.append(f"{i}. {model_name} ({model_type})")
            lines.append(f"   状态: {status}")
            
            # 错误信息
            if result.get('error'):
                lines.append(f"   错误: {result['error']}")
            
            # 统计信息
            if include_stats and result.get('stats'):
                stats = result['stats']
                duration = stats.get('duration') or 0
                token_count = stats.get('token_count') or 0
                tps = self._calculate_tokens_per_second(stats)
                
                # 确保数值不为None
                duration = float(duration) if duration is not None else 0.0
                token_count = int(token_count) if token_count is not None else 0
                tps = float(tps) if tps is not None else 0.0
                
                lines.append(f"   用时: {duration:.1f}秒")
                lines.append(f"   Token数: {token_count}")
                lines.append(f"   速度: {tps:.1f} tokens/秒")
            
            # 输出内容
            content = result.get('content') or ''
            if content:
                lines.append("   输出内容:")
                # 缩进内容
                content_lines = content.split('\n')
                for line in content_lines:
                    lines.append(f"     {line}")
            
            lines.append("")
            lines.append("-" * 60)
        
        return "\n".join(lines)
    
    def _generate_empty_export(self, format_type: str) -> str:
        """生成空结果的导出内容"""
        timestamp = datetime.now().isoformat()
        
        if format_type == 'json':
            return json.dumps({
                "export_info": {
                    "format": "json",
                    "timestamp": timestamp,
                    "total_models": 0,
                    "version": "1.0"
                },
                "comparison_results": {},
                "message": "暂无比较结果"
            }, ensure_ascii=False, indent=2)
        
        elif format_type == 'csv':
            return f"# 多LLM模型比较结果导出\n# 导出时间: {timestamp}\n# 暂无比较结果\n"
        
        elif format_type == 'markdown':
            return f"""# 多LLM模型比较结果报告

## 导出信息

- **导出时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **总模型数**: 0
- **状态**: 暂无比较结果

## 说明

请先进行模型比较，然后再导出结果。
"""
        
        elif format_type == 'txt':
            return f"""多LLM模型比较结果报告
{'=' * 60}

导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
总模型数: 0
状态: 暂无比较结果

说明: 请先进行模型比较，然后再导出结果。
"""
    
    def _calculate_tokens_per_second(self, stats: Dict[str, Any]) -> float:
        """计算每秒token数"""
        try:
            duration = float(stats.get('duration', 0))
            token_count = int(stats.get('token_count', 0))
            
            if duration > 0:
                return token_count / duration
            else:
                return 0.0
        except (ValueError, TypeError, ZeroDivisionError):
            return 0.0
    
    def _format_timestamp(self, timestamp: Optional[float]) -> str:
        """格式化时间戳"""
        if timestamp is None:
            return "N/A"
        
        try:
            dt = datetime.fromtimestamp(float(timestamp))
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        except (ValueError, TypeError, OSError):
            return "N/A"
    
    def get_filename(self, format_type: str, prefix: str = "llm_comparison") -> str:
        """
        生成导出文件名
        
        Args:
            format_type: 文件格式
            prefix: 文件名前缀
            
        Returns:
            文件名
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{prefix}_{timestamp}.{format_type}"
    
    def get_mime_type(self, format_type: str) -> str:
        """
        获取MIME类型
        
        Args:
            format_type: 文件格式
            
        Returns:
            MIME类型字符串
        """
        mime_types = {
            'json': 'application/json',
            'csv': 'text/csv',
            'markdown': 'text/markdown',
            'txt': 'text/plain'
        }
        
        return mime_types.get(format_type, 'text/plain')