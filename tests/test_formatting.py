"""
Test Code Formatting Tools
==========================
Test Black, isort, autopep8 functionality.

Verifies automatic code formatting works correctly.
"""

import pytest
import subprocess
from pathlib import Path
import tempfile


class TestBlack:
    """Test Black code formatter."""
    
    def test_black_import(self):
        """Test Black imports."""
        import black
        from black import format_str, Mode
        
        assert format_str is not None
        assert Mode is not None
    
    def test_black_format_code(self):
        """Test Black formats code correctly."""
        import black
        
        messy_code = '''
def function(x,y,z):
    result=x+y+z
    return result
'''
        
        formatted = black.format_str(messy_code, mode=black.Mode())
        
        # Black should add proper spacing
        assert 'x, y, z' in formatted
        assert 'result = x + y + z' in formatted
    
    def test_black_cli(self):
        """Test Black command-line interface."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text('def f(a,b,c):return a+b+c\n')
            
            # Run black
            result = subprocess.run(
                ['black', str(test_file)],
                capture_output=True,
                text=True
            )
            
            # Black should succeed
            assert result.returncode == 0
            
            # Check formatted content
            formatted_content = test_file.read_text()
            assert 'a, b, c' in formatted_content
            assert 'return a + b + c' in formatted_content
    
    def test_black_check_mode(self):
        """Test Black check mode (no modifications)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.py"
            original_code = 'def f(a, b, c):\n    return a + b + c\n'
            test_file.write_text(original_code)
            
            # Run black in check mode
            result = subprocess.run(
                ['black', '--check', str(test_file)],
                capture_output=True,
                text=True
            )
            
            # Should pass (already formatted)
            assert result.returncode == 0
            # File should not be modified
            assert test_file.read_text() == original_code


class TestIsort:
    """Test isort import sorter."""
    
    def test_isort_import(self):
        """Test isort imports."""
        import isort
        from isort import code as isort_code
        
        assert isort_code is not None
    
    def test_isort_sort_imports(self):
        """Test isort sorts imports correctly."""
        import isort
        
        messy_imports = '''
import sys
import os
from typing import List, Dict
import numpy
import pandas
from pathlib import Path
'''
        
        sorted_imports = isort.code(messy_imports)
        
        # Standard library should come first
        lines = sorted_imports.strip().split('\n')
        import_lines = [l for l in lines if l.strip()]
        
        # Check os and sys appear before numpy/pandas (standard lib first)
        os_idx = next(i for i, l in enumerate(import_lines) if 'os' in l)
        numpy_idx = next(i for i, l in enumerate(import_lines) if 'numpy' in l)
        
        assert os_idx < numpy_idx, "Standard library imports should come before third-party"
    
    def test_isort_cli(self):
        """Test isort command-line interface."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text('''
import pandas
import os
import numpy
import sys
''')
            
            # Run isort
            result = subprocess.run(
                ['isort', str(test_file)],
                capture_output=True,
                text=True
            )
            
            assert result.returncode == 0
            
            # Check sorted imports
            content = test_file.read_text()
            lines = [l.strip() for l in content.split('\n') if l.strip()]
            
            # os and sys should come before numpy and pandas
            os_idx = next(i for i, l in enumerate(lines) if 'import os' in l)
            pandas_idx = next(i for i, l in enumerate(lines) if 'import pandas' in l)
            
            assert os_idx < pandas_idx
    
    def test_isort_black_compatibility(self):
        """Test isort works with Black."""
        import isort
        import black
        
        code = '''
import pandas
import os
def function(x,y):
    return x+y
'''
        
        # First sort imports with isort
        sorted_code = isort.code(code)
        
        # Then format with Black
        formatted_code = black.format_str(sorted_code, mode=black.Mode())
        
        # Should have both: sorted imports and Black formatting
        assert 'import os' in formatted_code
        assert 'import pandas' in formatted_code
        assert 'x, y' in formatted_code  # Black formatting


class TestAutopep8:
    """Test autopep8 formatter."""
    
    def test_autopep8_import(self):
        """Test autopep8 imports."""
        import autopep8
        
        assert hasattr(autopep8, 'fix_code')
    
    def test_autopep8_fix_code(self):
        """Test autopep8 fixes PEP 8 violations."""
        import autopep8
        
        bad_code = '''
x=1+2
def f(a,b ):
    return a+b
'''
        
        fixed_code = autopep8.fix_code(bad_code)
        
        # Should fix spacing
        assert 'x = 1 + 2' in fixed_code
        assert 'def f(a, b):' in fixed_code
        assert 'return a + b' in fixed_code
    
    def test_autopep8_cli(self):
        """Test autopep8 command-line interface."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text('x=1+2\ny=3+4\n')
            
            # Run autopep8 in-place
            result = subprocess.run(
                ['autopep8', '--in-place', str(test_file)],
                capture_output=True,
                text=True
            )
            
            assert result.returncode == 0
            
            # Check fixed content
            fixed_content = test_file.read_text()
            assert 'x = 1 + 2' in fixed_content
            assert 'y = 3 + 4' in fixed_content


class TestFormattingWorkflow:
    """Test complete formatting workflow."""
    
    def test_full_formatting_pipeline(self):
        """Test running isort -> autopep8 -> black pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.py"
            messy_code = '''
import pandas
import os
import numpy
def function(x,y,z):
    result=x+y+z
    return result
'''
            test_file.write_text(messy_code)
            
            # Step 1: isort
            subprocess.run(['isort', str(test_file)], check=True)
            
            # Step 2: autopep8
            subprocess.run(['autopep8', '--in-place', str(test_file)], check=True)
            
            # Step 3: Black
            subprocess.run(['black', str(test_file)], check=True)
            
            # Check final result
            final_code = test_file.read_text()
            
            # Should have sorted imports
            lines = [l.strip() for l in final_code.split('\n') if l.strip() and 'import' in l]
            os_line = next((i for i, l in enumerate(lines) if 'import os' in l), -1)
            pandas_line = next((i for i, l in enumerate(lines) if 'import pandas' in l), -1)
            assert os_line < pandas_line
            
            # Should have Black formatting
            assert 'def function(x, y, z):' in final_code
            assert 'result = x + y + z' in final_code
    
    def test_formatting_consistency(self):
        """Test formatters produce consistent output."""
        import black
        import isort
        
        code = '''
def example(a, b, c):
    return a + b + c
'''
        
        # Format multiple times
        formatted1 = black.format_str(code, mode=black.Mode())
        formatted2 = black.format_str(formatted1, mode=black.Mode())
        formatted3 = black.format_str(formatted2, mode=black.Mode())
        
        # Should be identical (idempotent)
        assert formatted1 == formatted2 == formatted3
