"""
测试Markdown渲染功能

测试UI组件中的Markdown渲染和格式化功能。
"""

import pytest
import re
from unittest.mock import Mock, patch
from src.multi_llm_comparator.ui.components import (
    render_markdown_content,
    _enhance_code_blocks,
    _enhance_math_formulas,
    _enhance_tables,
    render_content_viewer,
    _highlight_search_term,
    render_collapsible_content
)


class TestMarkdownRendering:
    """测试Markdown渲染功能"""
    
    def test_enhance_code_blocks_basic(self):
        """测试基础代码块增强"""
        content = """
这是一些文本
```python
def hello():
    print("Hello, World!")
```
更多文本
"""
        result = _enhance_code_blocks(content, "test")
        
        # 检查是否包含增强的HTML结构
        assert '<div style="position: relative; margin: 10px 0;">' in result
        assert 'PYTHON' in result
        assert '📋 复制' in result
        assert 'def hello():' in result
        assert 'copyCode(' in result
    
    def test_enhance_code_blocks_no_language(self):
        """测试无语言标识的代码块"""
        content = """
```
echo "Hello"
```
"""
        result = _enhance_code_blocks(content, "test")
        
        assert 'TEXT' in result  # 默认语言
        assert 'echo "Hello"' in result
    
    def test_enhance_code_blocks_multiple(self):
        """测试多个代码块"""
        content = """
```python
print("Python")
```

```javascript
console.log("JavaScript");
```
"""
        result = _enhance_code_blocks(content, "test")
        
        assert 'PYTHON' in result
        assert 'JAVASCRIPT' in result
        assert result.count('📋 复制') == 2
    
    def test_enhance_math_formulas_inline(self):
        """测试行内数学公式"""
        content = "这是一个公式 $E = mc^2$ 在文本中"
        result = _enhance_math_formulas(content)
        
        assert r'\(E = mc^2\)' in result
        assert '$E = mc^2$' not in result
    
    def test_enhance_math_formulas_block(self):
        """测试块级数学公式"""
        content = """这是块级公式：$$\\int_{-\\infty}^{\\infty} e^{-x^2} dx = \\sqrt{\\pi}$$结束"""
        result = _enhance_math_formulas(content)
        
        assert r'\[' in result
        assert r'\]' in result
        assert '$$' not in result
    
    def test_enhance_tables_basic(self):
        """测试基础表格增强"""
        content = """
| 列1 | 列2 | 列3 |
|-----|-----|-----|
| 值1 | 值2 | 值3 |
| 值4 | 值5 | 值6 |
"""
        result = _enhance_tables(content)
        
        assert '<div style="overflow-x: auto; margin: 10px 0;">' in result
        assert '</div>' in result
        assert '| 列1 | 列2 | 列3 |' in result
    
    def test_enhance_tables_mixed_content(self):
        """测试混合内容中的表格"""
        content = """
这是一些文本

| 名称 | 值 |
|------|-----|
| A    | 1   |
| B    | 2   |

更多文本
"""
        result = _enhance_tables(content)
        
        # 应该只有一个表格包装
        assert result.count('<div style="overflow-x: auto; margin: 10px 0;">') == 1
        assert result.count('</div>') == 1
    
    def test_highlight_search_term_rendered(self):
        """测试渲染模式下的搜索高亮"""
        content = "这是一个测试文本，包含测试关键词"
        result = _highlight_search_term(content, "测试", is_raw=False)
        
        assert '<mark style="background-color: yellow' in result
        assert '测试' in result
    
    def test_highlight_search_term_raw(self):
        """测试原始文本模式下的搜索高亮"""
        content = "这是一个测试文本，包含测试关键词"
        result = _highlight_search_term(content, "测试", is_raw=True)
        
        assert '>>>测试<<<' in result
        assert '<mark' not in result
    
    def test_highlight_search_term_case_insensitive(self):
        """测试大小写不敏感的搜索"""
        content = "This is a TEST text with Test keywords"
        result = _highlight_search_term(content, "test", is_raw=False)
        
        # 应该高亮所有匹配项（不区分大小写）
        assert result.count('<mark') >= 2
    
    def test_highlight_search_term_special_characters(self):
        """测试特殊字符的搜索"""
        content = "文本包含特殊字符: $100 + 50% = $150"
        result = _highlight_search_term(content, "$100", is_raw=False)
        
        assert '<mark' in result
        assert '$100' in result
    
    def test_highlight_search_term_empty(self):
        """测试空搜索词"""
        content = "这是测试文本"
        result = _highlight_search_term(content, "", is_raw=False)
        
        assert result == content  # 应该返回原始内容
    
    @patch('streamlit.markdown')
    def test_render_markdown_content_basic(self, mock_markdown):
        """测试基础Markdown内容渲染"""
        content = "# 标题\n\n这是内容"
        render_markdown_content(content)
        
        mock_markdown.assert_called_once()
        args, kwargs = mock_markdown.call_args
        assert args[0] == content
        assert kwargs.get('unsafe_allow_html') is True
    
    @patch('streamlit.markdown')
    def test_render_markdown_content_with_code(self, mock_markdown):
        """测试包含代码的Markdown渲染"""
        content = """
# 标题

```python
print("Hello")
```
"""
        render_markdown_content(content, enable_code_copy=True)
        
        mock_markdown.assert_called_once()
        args, kwargs = mock_markdown.call_args
        rendered_content = args[0]
        
        # 检查代码块是否被增强
        assert '📋 复制' in rendered_content
        assert 'PYTHON' in rendered_content
    
    @patch('streamlit.markdown')
    def test_render_markdown_content_with_math(self, mock_markdown):
        """测试包含数学公式的Markdown渲染"""
        content = "公式: $E = mc^2$ 和 $$\\sum_{i=1}^n i$$"
        render_markdown_content(content, enable_math=True)
        
        mock_markdown.assert_called_once()
        args, kwargs = mock_markdown.call_args
        rendered_content = args[0]
        
        # 检查数学公式是否被转换
        assert r'\(' in rendered_content
        assert r'\[' in rendered_content
        assert '$E = mc^2$' not in rendered_content
    
    @patch('streamlit.markdown')
    def test_render_markdown_content_with_tables(self, mock_markdown):
        """测试包含表格的Markdown渲染"""
        content = """
| 列1 | 列2 |
|-----|-----|
| A   | B   |
"""
        render_markdown_content(content, enable_tables=True)
        
        mock_markdown.assert_called_once()
        args, kwargs = mock_markdown.call_args
        rendered_content = args[0]
        
        # 检查表格是否被增强
        assert 'overflow-x: auto' in rendered_content
    
    @patch('streamlit.info')
    def test_render_markdown_content_empty(self, mock_info):
        """测试空内容的处理"""
        render_markdown_content("")
        
        mock_info.assert_called_once_with("暂无内容")
    
    @patch('streamlit.subheader')
    @patch('streamlit.selectbox')
    @patch('streamlit.text_input')
    @patch('streamlit.container')
    def test_render_content_viewer_basic(self, mock_container, mock_text_input, 
                                       mock_selectbox, mock_subheader):
        """测试内容查看器基础功能"""
        content = "# 测试内容\n\n这是一些测试文本"
        
        # 模拟Streamlit组件返回值
        mock_selectbox.return_value = "rendered"
        mock_text_input.return_value = ""
        
        result = render_content_viewer(content, "测试标题")
        
        # 检查返回的状态
        assert result["view_mode"] == "rendered"
        assert result["search_term"] == ""
        assert result["char_count"] > 0
        assert result["word_count"] > 0
        assert result["line_count"] > 0
    
    @patch('streamlit.info')
    def test_render_content_viewer_empty(self, mock_info):
        """测试空内容的内容查看器"""
        result = render_content_viewer("", "测试标题")
        
        mock_info.assert_called_once_with("暂无内容")
        assert result["view_mode"] == "rendered"
        assert result["search_term"] == ""


class TestContentStatistics:
    """测试内容统计功能"""
    
    def test_content_statistics_basic(self):
        """测试基础内容统计"""
        content = "这是一个测试文本\n包含多行\n和多个单词"
        
        char_count = len(content)
        word_count = len(content.split())
        line_count = len(content.split('\n'))
        
        assert char_count > 0
        assert word_count > 0
        assert line_count == 3
    
    def test_content_statistics_empty(self):
        """测试空内容统计"""
        content = ""
        
        char_count = len(content)
        word_count = len(content.split())
        line_count = len(content.split('\n'))
        
        assert char_count == 0
        assert word_count == 0
        assert line_count == 1  # 空字符串split('\n')返回['']
    
    def test_content_statistics_whitespace(self):
        """测试只包含空白字符的内容"""
        content = "   \n\n   \n"
        
        char_count = len(content)
        word_count = len(content.split())
        line_count = len(content.split('\n'))
        
        assert char_count > 0
        assert word_count == 0  # 只有空白字符，没有单词
        assert line_count == 4


class TestSearchFunctionality:
    """测试搜索功能"""
    
    def test_search_count_basic(self):
        """测试基础搜索计数"""
        content = "这是测试文本，包含测试关键词和更多测试内容"
        search_term = "测试"
        
        count = content.lower().count(search_term.lower())
        assert count == 3
    
    def test_search_count_case_insensitive(self):
        """测试大小写不敏感搜索"""
        content = "This is a TEST text with Test and test keywords"
        search_term = "test"
        
        count = content.lower().count(search_term.lower())
        assert count == 3
    
    def test_search_count_no_matches(self):
        """测试无匹配的搜索"""
        content = "这是一些文本内容"
        search_term = "不存在"
        
        count = content.lower().count(search_term.lower())
        assert count == 0
    
    def test_search_count_empty_term(self):
        """测试空搜索词"""
        content = "这是一些文本内容"
        search_term = ""
        
        # 空字符串的count应该返回字符数+1
        count = content.count(search_term)
        assert count == len(content) + 1


class TestCodeBlockExtraction:
    """测试代码块提取功能"""
    
    def test_extract_code_blocks_single(self):
        """测试提取单个代码块"""
        content = """
文本内容
```python
def hello():
    return "world"
```
更多文本
"""
        pattern = r'```(\w+)?\n(.*?)\n```'
        matches = re.findall(pattern, content, re.DOTALL)
        
        assert len(matches) == 1
        assert matches[0][0] == 'python'
        assert 'def hello():' in matches[0][1]
    
    def test_extract_code_blocks_multiple(self):
        """测试提取多个代码块"""
        content = """
```python
print("Python")
```

```javascript
console.log("JS");
```

```
plain text
```
"""
        pattern = r'```(\w+)?\n(.*?)\n```'
        matches = re.findall(pattern, content, re.DOTALL)
        
        assert len(matches) == 3
        assert matches[0][0] == 'python'
        assert matches[1][0] == 'javascript'
        assert matches[2][0] == ''  # 无语言标识
    
    def test_extract_code_blocks_nested(self):
        """测试嵌套代码块（应该正确处理）"""
        content = """
```markdown
# 标题
```python
print("nested")
```
结束
```
"""
        pattern = r'```(\w+)?\n(.*?)\n```'
        matches = re.findall(pattern, content, re.DOTALL)
        
        # 由于正则表达式的贪婪匹配特性，会匹配到多个代码块
        # 这是预期行为，因为内容确实包含多个代码块标记
        assert len(matches) >= 1
        assert any(match[0] == 'markdown' for match in matches)


if __name__ == "__main__":
    pytest.main([__file__])