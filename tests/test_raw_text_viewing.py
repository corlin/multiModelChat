"""
测试原始文本查看功能

测试UI组件中的原始文本查看和格式化功能。
"""

import pytest
import re
from unittest.mock import Mock, patch, MagicMock
from src.multi_llm_comparator.ui.components import (
    render_advanced_text_viewer,
    render_text_comparison_viewer,
    render_content_formatting_options,
    render_content_search_and_replace,
    _apply_formatting_options,
    _render_text_analysis
)


class TestAdvancedTextViewer:
    """测试高级文本查看器"""
    
    @patch('streamlit.subheader')
    @patch('streamlit.columns')
    @patch('streamlit.selectbox')
    @patch('streamlit.checkbox')
    @patch('streamlit.text_input')
    @patch('streamlit.container')
    def test_render_advanced_text_viewer_basic(self, mock_container, mock_text_input,
                                             mock_checkbox, mock_selectbox, mock_columns, mock_subheader):
        """测试高级文本查看器基础功能"""
        content = "这是测试内容\n第二行\n第三行"
        
        # 模拟Streamlit组件返回值
        mock_selectbox.return_value = "plain"
        mock_checkbox.side_effect = [True, True]  # line_numbers, word_wrap
        mock_text_input.return_value = ""
        
        # 模拟columns
        mock_col = MagicMock()
        mock_columns.return_value = [mock_col, mock_col, mock_col, mock_col]
        
        result = render_advanced_text_viewer(content, "测试查看器")
        
        # 检查返回的状态
        assert result["view_mode"] == "plain"
        assert result["char_count"] > 0
        assert result["word_count"] > 0
        assert result["line_count"] == 3
    
    @patch('streamlit.info')
    def test_render_advanced_text_viewer_empty(self, mock_info):
        """测试空内容的高级文本查看器"""
        result = render_advanced_text_viewer("", "测试查看器")
        
        mock_info.assert_called_once_with("暂无内容")
        assert result["view_mode"] == "plain"
        assert result["options"] == {}
    
    def test_apply_formatting_options_basic(self):
        """测试基础格式化选项应用"""
        content = "  这是测试   内容  \n  第二行  \n"
        
        result = _apply_formatting_options(
            content,
            remove_extra_spaces=True,
            trim_lines=True
        )
        
        # 检查多余空格被移除，行首尾空白被去除
        lines = result.split('\n')
        assert lines[0] == "这是测试 内容"
        assert lines[1] == "第二行"
    
    def test_apply_formatting_options_normalize_line_endings(self):
        """测试换行符标准化"""
        content = "第一行\r\n第二行\r第三行\n"
        
        result = _apply_formatting_options(
            content,
            normalize_line_endings=True
        )
        
        # 所有换行符应该被标准化为\n
        assert '\r\n' not in result
        assert '\r' not in result
        assert result.count('\n') == 3
    
    def test_apply_formatting_options_show_whitespace(self):
        """测试显示空白字符"""
        content = "文本 内容\t制表符"
        
        result = _apply_formatting_options(
            content,
            show_whitespace=True
        )
        
        # 空格应该被替换为·，制表符被替换为→
        assert '·' in result
        assert '→' in result
        # 检查原始空格和制表符不再存在
        assert '\t' not in result
    
    def test_apply_formatting_options_highlight_long_lines(self):
        """测试高亮长行"""
        short_line = "短行"
        long_line = "这是一个非常长的行，超过了指定的最大长度限制，应该被高亮显示"
        content = f"{short_line}\n{long_line}"
        
        result = _apply_formatting_options(
            content,
            highlight_long_lines=True,
            max_line_length=20
        )
        
        lines = result.split('\n')
        assert lines[0] == short_line  # 短行不变
        assert lines[1].startswith('⚠️')  # 长行被高亮
        assert lines[1].endswith('⚠️')


class TestTextAnalysis:
    """测试文本分析功能"""
    
    @patch('streamlit.metric')
    @patch('streamlit.subheader')
    @patch('streamlit.columns')
    @patch('streamlit.text')
    def test_render_text_analysis_basic(self, mock_text, mock_columns, mock_subheader, mock_metric):
        """测试基础文本分析"""
        content = "这是测试文本，包含123数字和标点符号！"
        
        # 模拟columns
        mock_col = MagicMock()
        mock_columns.return_value = [mock_col, mock_col, mock_col, mock_col]
        
        _render_text_analysis(content, "test_key")
        
        # 检查是否调用了metric来显示统计信息
        assert mock_metric.call_count >= 4  # 至少应该有4个基础指标
    
    def test_text_statistics_calculation(self):
        """测试文本统计计算"""
        content = "Hello World! 123 测试文本。\n第二行内容"
        
        char_count = len(content)
        word_count = len(content.split())
        line_count = len(content.split('\n'))
        
        # 字符类型统计
        alpha_count = sum(1 for c in content if c.isalpha())
        digit_count = sum(1 for c in content if c.isdigit())
        space_count = sum(1 for c in content if c.isspace())
        punct_count = sum(1 for c in content if not c.isalnum() and not c.isspace())
        
        assert char_count > 0
        assert word_count > 0
        assert line_count == 2
        assert alpha_count > 0
        assert digit_count > 0
        assert space_count > 0
        assert punct_count > 0
    
    def test_word_frequency_analysis(self):
        """测试词频分析"""
        content = "这是测试文本 测试 测试 其他词汇 其他词汇"
        
        words = content.split()
        from collections import Counter
        word_freq = Counter(words)
        
        # "测试"应该出现2次（"测试文本"是一个词，"测试"单独出现2次）
        assert word_freq["测试"] == 2
        # "其他词汇"应该出现2次
        assert word_freq["其他词汇"] == 2


class TestContentFormatting:
    """测试内容格式化功能"""
    
    @patch('streamlit.subheader')
    @patch('streamlit.columns')
    @patch('streamlit.checkbox')
    @patch('streamlit.number_input')
    @patch('streamlit.selectbox')
    @patch('streamlit.button')
    def test_render_content_formatting_options(self, mock_button, mock_selectbox, 
                                             mock_number_input, mock_checkbox, 
                                             mock_columns, mock_subheader):
        """测试内容格式化选项渲染"""
        content = "测试内容"
        
        # 模拟返回值
        mock_checkbox.side_effect = [True, False, True, False, True, False]
        mock_number_input.return_value = 80
        mock_selectbox.return_value = "txt"
        mock_button.return_value = False
        
        # 模拟columns
        mock_col = MagicMock()
        mock_columns.return_value = [mock_col, mock_col, mock_col]
        
        result = render_content_formatting_options(content, "test_key")
        
        # 检查返回的选项
        assert "remove_extra_spaces" in result
        assert "export_format" in result
        assert result["export_format"] == "txt"
    
    def test_render_content_formatting_options_empty(self):
        """测试空内容的格式化选项"""
        result = render_content_formatting_options("", "test_key")
        
        assert result == {}


class TestSearchAndReplace:
    """测试搜索和替换功能"""
    
    @patch('streamlit.subheader')
    @patch('streamlit.columns')
    @patch('streamlit.text_input')
    @patch('streamlit.checkbox')
    @patch('streamlit.success')
    @patch('streamlit.warning')
    def test_render_content_search_and_replace_basic(self, mock_warning, mock_success,
                                                   mock_checkbox, mock_text_input,
                                                   mock_columns, mock_subheader):
        """测试基础搜索和替换功能"""
        content = "这是测试文本，包含测试关键词"
        
        # 模拟输入
        mock_text_input.side_effect = ["测试", "替换"]
        mock_checkbox.side_effect = [False, False, False, True]  # case_sensitive, use_regex, whole_word, preview_only
        
        # 模拟columns - 需要返回正确数量的列
        mock_col = MagicMock()
        mock_columns.side_effect = [
            [mock_col, mock_col],  # 第一次调用返回2列
            [mock_col, mock_col, mock_col, mock_col]  # 第二次调用返回4列
        ]
        
        result = render_content_search_and_replace(content, "test_key")
        
        # 预览模式下应该返回原始内容
        assert result == content
        # 应该显示找到匹配项的消息
        mock_success.assert_called()
    
    @patch('streamlit.subheader')
    @patch('streamlit.columns')
    @patch('streamlit.text_input')
    @patch('streamlit.checkbox')
    @patch('streamlit.warning')
    def test_render_content_search_and_replace_no_match(self, mock_warning, mock_checkbox,
                                                       mock_text_input, mock_columns, mock_subheader):
        """测试无匹配的搜索"""
        content = "这是测试文本"
        
        # 模拟输入
        mock_text_input.side_effect = ["不存在", "替换"]
        mock_checkbox.side_effect = [False, False, False, True]
        
        # 模拟columns - 需要返回正确数量的列
        mock_col = MagicMock()
        mock_columns.side_effect = [
            [mock_col, mock_col],  # 第一次调用返回2列
            [mock_col, mock_col, mock_col, mock_col]  # 第二次调用返回4列
        ]
        
        result = render_content_search_and_replace(content, "test_key")
        
        assert result == content
        mock_warning.assert_called_with("未找到匹配项")
    
    def test_search_and_replace_logic_basic(self):
        """测试基础搜索替换逻辑"""
        content = "这是测试文本，包含测试关键词"
        search_text = "测试"
        replace_text = "替换"
        
        result = content.replace(search_text, replace_text)
        
        assert "替换" in result
        assert result.count("替换") == 2  # 应该有两个替换
        assert "测试" not in result
    
    def test_search_and_replace_case_sensitive(self):
        """测试大小写敏感搜索"""
        content = "This is a Test text with test keywords"
        search_text = "test"
        
        # 大小写敏感
        case_sensitive_count = content.count(search_text)
        assert case_sensitive_count == 1
        
        # 大小写不敏感
        case_insensitive_count = content.lower().count(search_text.lower())
        assert case_insensitive_count == 2
    
    def test_search_and_replace_regex(self):
        """测试正则表达式搜索"""
        content = "电话: 123-456-7890, 手机: 987-654-3210"
        pattern = r'\d{3}-\d{3}-\d{4}'
        
        matches = re.findall(pattern, content)
        assert len(matches) == 2
        
        # 替换为统一格式
        result = re.sub(pattern, "XXX-XXX-XXXX", content)
        assert "XXX-XXX-XXXX" in result
        assert result.count("XXX-XXX-XXXX") == 2


class TestTextComparison:
    """测试文本比较功能"""
    
    @patch('streamlit.subheader')
    @patch('streamlit.columns')
    @patch('streamlit.selectbox')
    @patch('streamlit.checkbox')
    @patch('streamlit.text_area')
    @patch('streamlit.markdown')
    def test_render_text_comparison_viewer_side_by_side(self, mock_markdown, mock_text_area,
                                                       mock_checkbox, mock_selectbox,
                                                       mock_columns, mock_subheader):
        """测试并排文本比较"""
        contents = {
            "文本1": "这是第一个文本",
            "文本2": "这是第二个文本"
        }
        
        # 模拟返回值
        mock_selectbox.return_value = "side_by_side"
        mock_checkbox.side_effect = [True, True]  # show_stats, sync_scroll
        
        # 模拟columns - 需要返回正确数量的列
        mock_col = MagicMock()
        mock_columns.side_effect = [
            [mock_col, mock_col, mock_col],  # 第一次调用返回3列（控制选项）
            [mock_col, mock_col]  # 第二次调用返回2列（内容显示）
        ]
        
        render_text_comparison_viewer(contents, "比较测试")
        
        # 检查是否调用了text_area来显示内容
        assert mock_text_area.call_count == 2
    
    @patch('streamlit.info')
    def test_render_text_comparison_viewer_empty(self, mock_info):
        """测试空内容的文本比较"""
        render_text_comparison_viewer({}, "比较测试")
        
        mock_info.assert_called_once_with("暂无内容进行比较")
    
    def test_text_diff_logic(self):
        """测试文本差异比较逻辑"""
        text1 = "第一行\n第二行\n第三行"
        text2 = "第一行\n修改的第二行\n第三行"
        
        lines1 = text1.split('\n')
        lines2 = text2.split('\n')
        
        # 找出不同的行
        differences = []
        for i, (line1, line2) in enumerate(zip(lines1, lines2)):
            if line1 != line2:
                differences.append((i, line1, line2))
        
        assert len(differences) == 1
        assert differences[0][0] == 1  # 第二行不同
        assert differences[0][1] == "第二行"
        assert differences[0][2] == "修改的第二行"


class TestContentStatistics:
    """测试内容统计功能"""
    
    def test_content_statistics_comprehensive(self):
        """测试全面的内容统计"""
        content = """这是一个测试文档。

包含多个段落和不同类型的内容：
- 列表项1
- 列表项2

还有数字123和特殊字符！@#$%

最后一段。"""
        
        # 基础统计
        char_count = len(content)
        word_count = len(content.split())
        line_count = len(content.split('\n'))
        paragraph_count = len([p for p in content.split('\n\n') if p.strip()])
        
        assert char_count > 0
        assert word_count > 0
        assert line_count > 1
        assert paragraph_count > 1
        
        # 字符类型统计
        alpha_count = sum(1 for c in content if c.isalpha())
        digit_count = sum(1 for c in content if c.isdigit())
        space_count = sum(1 for c in content if c.isspace())
        punct_count = sum(1 for c in content if not c.isalnum() and not c.isspace())
        
        assert alpha_count > 0
        assert digit_count > 0  # 包含"123"
        assert space_count > 0
        assert punct_count > 0  # 包含标点符号
    
    def test_line_length_analysis(self):
        """测试行长度分析"""
        content = "短行\n这是一个比较长的行，用于测试行长度分析功能\n中等长度的行"
        
        lines = content.split('\n')
        line_lengths = [len(line) for line in lines]
        
        avg_length = sum(line_lengths) / len(line_lengths)
        max_length = max(line_lengths)
        min_length = min(line_lengths)
        
        assert avg_length > 0
        assert max_length > min_length
        assert min_length == 2  # "短行"
        assert max_length > 20  # 最长的行


class TestUtilityFunctions:
    """测试工具函数"""
    
    def test_hash_consistency(self):
        """测试哈希一致性"""
        content1 = "相同内容"
        content2 = "相同内容"
        content3 = "不同内容"
        
        hash1 = hash(content1[:50])
        hash2 = hash(content2[:50])
        hash3 = hash(content3[:50])
        
        assert hash1 == hash2
        assert hash1 != hash3
    
    def test_content_truncation(self):
        """测试内容截断"""
        long_content = "这是一个很长的内容" * 100
        truncated = long_content[:50]
        
        assert len(truncated) == 50
        assert truncated in long_content
    
    def test_regex_escaping(self):
        """测试正则表达式转义"""
        special_chars = ".*+?^${}()|[]"
        escaped = re.escape(special_chars)
        
        # 转义后的字符串应该可以安全用于正则表达式
        pattern = re.compile(escaped)
        assert pattern.search(special_chars) is not None


if __name__ == "__main__":
    pytest.main([__file__])