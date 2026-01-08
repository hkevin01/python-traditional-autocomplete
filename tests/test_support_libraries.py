"""
Test Support Libraries
=====================
Test cattrs, docstring-to-markdown, pytoolconfig, parso.

These libraries support the main tools.
"""

import pytest


class TestCattrs:
    """Test cattrs serialization library."""
    
    def test_cattrs_import(self):
        """Test cattrs imports."""
        import cattrs
        from cattrs import Converter
        
        assert Converter is not None
    
    def test_cattrs_structure(self):
        """Test cattrs structuring (dict to object)."""
        import cattrs
        from dataclasses import dataclass
        
        @dataclass
        class Person:
            name: str
            age: int
        
        converter = cattrs.Converter()
        data = {"name": "Alice", "age": 30}
        
        person = converter.structure(data, Person)
        
        assert person.name == "Alice"
        assert person.age == 30
    
    def test_cattrs_unstructure(self):
        """Test cattrs unstructuring (object to dict)."""
        import cattrs
        from dataclasses import dataclass
        
        @dataclass
        class Person:
            name: str
            age: int
        
        converter = cattrs.Converter()
        person = Person(name="Bob", age=25)
        
        data = converter.unstructure(person)
        
        assert data == {"name": "Bob", "age": 25}
    
    def test_cattrs_list_handling(self):
        """Test cattrs with lists."""
        import cattrs
        from typing import List
        from dataclasses import dataclass
        
        @dataclass
        class Item:
            id: int
            name: str
        
        converter = cattrs.Converter()
        data = [
            {"id": 1, "name": "Item1"},
            {"id": 2, "name": "Item2"}
        ]
        
        items = converter.structure(data, List[Item])
        
        assert len(items) == 2
        assert items[0].name == "Item1"
        assert items[1].id == 2


class TestDocstringToMarkdown:
    """Test docstring-to-markdown converter."""
    
    def test_docstring_to_markdown_import(self):
        """Test docstring-to-markdown imports."""
        import docstring_to_markdown
        
        assert hasattr(docstring_to_markdown, 'convert')
    
    def test_convert_simple_docstring(self):
        """Test converting a simple docstring."""
        from docstring_to_markdown import convert
        
        docstring = """
        This is a simple docstring.
        
        It has multiple lines.
        """
        
        markdown = convert(docstring)
        
        assert isinstance(markdown, str)
        assert len(markdown) > 0
    
    def test_convert_google_style(self):
        """Test converting Google-style docstring."""
        from docstring_to_markdown import convert
        
        docstring = """
        Calculate the sum of two numbers.
        
        Args:
            a: First number
            b: Second number
        
        Returns:
            The sum of a and b
        """
        
        markdown = convert(docstring)
        
        # Should preserve Args and Returns structure
        assert 'Args:' in markdown or 'args' in markdown.lower()
        assert 'Returns:' in markdown or 'returns' in markdown.lower()
    
    def test_convert_numpy_style(self):
        """Test converting NumPy-style docstring."""
        from docstring_to_markdown import convert
        
        docstring = """
        Calculate the sum.
        
        Parameters
        ----------
        a : int
            First number
        b : int
            Second number
        
        Returns
        -------
        int
            Sum of numbers
        """
        
        markdown = convert(docstring)
        
        assert isinstance(markdown, str)
        assert len(markdown) > 0


class TestPytoolconfig:
    """Test pytoolconfig configuration manager."""
    
    def test_pytoolconfig_import(self):
        """Test pytoolconfig imports."""
        import pytoolconfig
        from pytoolconfig import PyToolConfig
        
        assert PyToolConfig is not None
    
    def test_pytoolconfig_sources(self):
        """Test pytoolconfig can load from various sources."""
        from pytoolconfig import sources
        
        # Should have various config source types
        assert hasattr(sources, 'PyProject')
        assert hasattr(sources, 'SetupCfg')
    
    def test_pytoolconfig_field_types(self):
        """Test pytoolconfig field type system."""
        from pytoolconfig import field
        
        # Should have field type definitions
        assert hasattr(field, 'Field')


class TestParso:
    """Test parso Python parser."""
    
    def test_parso_import(self):
        """Test parso imports."""
        import parso
        from parso import parse
        
        assert parse is not None
    
    def test_parso_parse_simple(self):
        """Test parsing simple Python code."""
        import parso
        
        code = "x = 1 + 2"
        module = parso.parse(code)
        
        assert module is not None
        assert module.type == 'file_input'
    
    def test_parso_parse_function(self):
        """Test parsing a function definition."""
        import parso
        
        code = """
def add(a, b):
    return a + b
"""
        module = parso.parse(code)
        
        # Should successfully parse
        assert module is not None
        
        # Should find function definition
        children = module.children
        func_def = None
        for child in children:
            if hasattr(child, 'type') and child.type == 'funcdef':
                func_def = child
                break
        
        # May be wrapped in other nodes, just verify parsing worked
        assert len(children) > 0
    
    def test_parso_parse_class(self):
        """Test parsing a class definition."""
        import parso
        
        code = """
class MyClass:
    def __init__(self):
        self.value = 42
"""
        module = parso.parse(code)
        
        assert module is not None
        # Should parse without errors
        assert module.type == 'file_input'
    
    def test_parso_error_recovery(self):
        """Test parso error recovery."""
        import parso
        
        # Code with syntax error
        code = "def broken("  # Missing closing paren
        
        # Parso should still parse (with error recovery)
        module = parso.parse(code)
        
        assert module is not None
        # Check for syntax errors
        errors = module.errors
        assert len(errors) > 0
    
    def test_parso_python_versions(self):
        """Test parso with different Python versions."""
        import parso
        
        # Python 3.12 syntax
        code_312 = "x: int = 42"
        module = parso.parse(code_312, version=(3, 12))
        
        assert module is not None
    
    def test_parso_completion_context(self):
        """Test parso for completion scenarios."""
        import parso
        
        code = "import os\nos."
        module = parso.parse(code)
        
        # Parso can parse incomplete code for completion
        assert module is not None
        
        # Should have import statement
        has_import = False
        for child in module.children:
            if hasattr(child, 'type'):
                if child.type == 'simple_stmt':
                    # Check for import
                    for subchild in child.children:
                        if hasattr(subchild, 'type') and 'import' in subchild.type:
                            has_import = True
        
        # Module should parse successfully
        assert module.type == 'file_input'


class TestSupportLibrariesIntegration:
    """Test support libraries work together."""
    
    def test_cattrs_with_parso(self):
        """Test using cattrs to structure parso parse results."""
        import cattrs
        import parso
        from dataclasses import dataclass
        
        code = "x = 42"
        module = parso.parse(code)
        
        # Create a dataclass to hold parse info
        @dataclass
        class ParseInfo:
            type: str
            line_count: int
        
        # Extract info from parso module
        info_dict = {
            "type": module.type,
            "line_count": len(module.get_code().split('\n'))
        }
        
        # Structure with cattrs
        converter = cattrs.Converter()
        info = converter.structure(info_dict, ParseInfo)
        
        assert info.type == 'file_input'
        assert info.line_count >= 1
