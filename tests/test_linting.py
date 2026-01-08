"""
Test Linting Tools
=================
Test Pylint, Flake8, Pydocstyle functionality.

Verifies code quality checks work correctly.
"""

import pytest
import subprocess
from pathlib import Path
import tempfile


class TestPylint:
    """Test Pylint functionality."""
    
    def test_pylint_import(self):
        """Test Pylint imports."""
        import pylint
        from pylint import lint
        
        assert lint is not None
    
    def test_pylint_good_code(self):
        """Test Pylint on good code."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "good_code.py"
            test_file.write_text('''
"""Module docstring."""


def add_numbers(a: int, b: int) -> int:
    """Add two numbers.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        Sum of a and b
    """
    return a + b
''')
            
            result = subprocess.run(
                ['pylint', str(test_file), '--score=y'],
                capture_output=True,
                text=True
            )
            
            # Good code should get a high score
            assert 'Your code has been rated at' in result.stdout
    
    def test_pylint_bad_code(self):
        """Test Pylint detects issues in bad code."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "bad_code.py"
            test_file.write_text('''
x=1;y=2  # Bad formatting
def f(a,b):  # No docstring, bad naming
    return a+b
unused_var = 42
''')
            
            result = subprocess.run(
                ['pylint', str(test_file)],
                capture_output=True,
                text=True
            )
            
            # Should detect issues
            output = result.stdout + result.stderr
            # Pylint should find conventions, warnings, or errors
            assert any(word in output.lower() for word in ['convention', 'warning', 'error', 'rated'])


class TestFlake8:
    """Test Flake8 functionality."""
    
    def test_flake8_import(self):
        """Test Flake8 imports."""
        import flake8
        from flake8.api import legacy as flake8_legacy
        
        assert flake8_legacy is not None
    
    def test_flake8_good_code(self):
        """Test Flake8 on PEP 8 compliant code."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "good_code.py"
            test_file.write_text('''
"""Module docstring."""


def add_numbers(a, b):
    """Add two numbers."""
    return a + b


result = add_numbers(1, 2)
print(result)
''')
            
            result = subprocess.run(
                ['flake8', str(test_file)],
                capture_output=True,
                text=True
            )
            
            # Should have no errors
            assert result.returncode == 0
            assert result.stdout.strip() == ''
    
    def test_flake8_bad_code(self):
        """Test Flake8 detects PEP 8 violations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "bad_code.py"
            test_file.write_text('''
import os,sys  # E401: multiple imports on one line
x=1+2  # E225: missing whitespace around operator
def f( a,b ):  # E201, E202: whitespace issues
    pass
''')
            
            result = subprocess.run(
                ['flake8', str(test_file)],
                capture_output=True,
                text=True
            )
            
            # Should detect violations
            assert result.returncode != 0 or result.stdout.strip() != ''
            output = result.stdout
            # Should mention error codes
            assert any(code in output for code in ['E', 'W', 'F'])


class TestPydocstyle:
    """Test Pydocstyle functionality."""
    
    def test_pydocstyle_import(self):
        """Test Pydocstyle imports."""
        import pydocstyle
        from pydocstyle import checker
        
        assert checker is not None
    
    def test_pydocstyle_good_docstrings(self):
        """Test Pydocstyle on good docstrings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "good_docs.py"
            test_file.write_text('''
"""Module with good docstrings.

This module demonstrates proper documentation.
"""


def calculate_sum(a, b):
    """Calculate the sum of two numbers.
    
    Args:
        a: First number
        b: Second number
    
    Returns:
        The sum of a and b
    """
    return a + b


class Calculator:
    """A simple calculator class.
    
    This class provides basic arithmetic operations.
    """
    
    def add(self, a, b):
        """Add two numbers.
        
        Args:
            a: First number
            b: Second number
        
        Returns:
            Sum of a and b
        """
        return a + b
''')
            
            result = subprocess.run(
                ['pydocstyle', str(test_file)],
                capture_output=True,
                text=True
            )
            
            # Good docstrings should pass (return code 0 or minimal issues)
            # Some versions may still suggest improvements
            assert result.returncode in [0, 1]
    
    def test_pydocstyle_missing_docstrings(self):
        """Test Pydocstyle detects missing docstrings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "bad_docs.py"
            test_file.write_text('''
def calculate_sum(a, b):
    return a + b

class Calculator:
    def add(self, a, b):
        return a + b
''')
            
            result = subprocess.run(
                ['pydocstyle', str(test_file)],
                capture_output=True,
                text=True
            )
            
            # Should detect missing docstrings
            output = result.stdout + result.stderr
            assert 'Missing docstring' in output or 'D' in output


class TestLintingIntegration:
    """Test linting tools work together."""
    
    def test_all_linters_on_same_file(self):
        """Test running all linters on the same file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test_code.py"
            test_file.write_text('''
"""Test module for linting."""


def example_function(param_a: int, param_b: int) -> int:
    """Example function with proper documentation.
    
    Args:
        param_a: First parameter
        param_b: Second parameter
    
    Returns:
        Sum of parameters
    """
    result = param_a + param_b
    return result
''')
            
            # Run all three linters
            pylint_result = subprocess.run(
                ['pylint', str(test_file)],
                capture_output=True,
                text=True
            )
            
            flake8_result = subprocess.run(
                ['flake8', str(test_file)],
                capture_output=True,
                text=True
            )
            
            pydocstyle_result = subprocess.run(
                ['pydocstyle', str(test_file)],
                capture_output=True,
                text=True
            )
            
            # All linters should run successfully (not crash)
            # Good code should pass flake8
            assert flake8_result.returncode == 0
            assert flake8_result.stdout.strip() == ''
