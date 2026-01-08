"""
Test Type Checking with MyPy
============================
Test MyPy static type verification.

Verifies type hints are checked correctly.
"""

import pytest
import subprocess
from pathlib import Path
import tempfile


class TestMyPyImports:
    """Test MyPy imports."""
    
    def test_mypy_import(self):
        """Test MyPy main imports."""
        import mypy
        from mypy import api
        
        assert api is not None
    
    def test_mypy_api_run(self):
        """Test MyPy API can run type checking."""
        from mypy import api
        
        code = '''
def add(a: int, b: int) -> int:
    return a + b

result: int = add(1, 2)
'''
        
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text(code)
            
            # Run MyPy via API
            result = api.run([str(test_file)])
            normal_report, error_report, exit_status = result
            
            # Good code should pass
            assert exit_status == 0


class TestMyPyTypeChecking:
    """Test MyPy type checking functionality."""
    
    def test_mypy_correct_types(self):
        """Test MyPy passes correct type annotations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "correct_types.py"
            test_file.write_text('''
def calculate_sum(numbers: list[int]) -> int:
    """Calculate sum of numbers."""
    total: int = 0
    for num in numbers:
        total += num
    return total

result: int = calculate_sum([1, 2, 3, 4, 5])
''')
            
            result = subprocess.run(
                ['mypy', str(test_file)],
                capture_output=True,
                text=True
            )
            
            # Should pass type checking
            assert result.returncode == 0
            assert 'Success' in result.stdout or result.stdout.strip() == ''
    
    def test_mypy_type_errors(self):
        """Test MyPy detects type errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "type_errors.py"
            test_file.write_text('''
def add_numbers(a: int, b: int) -> int:
    return a + b

# Type error: passing string to int parameter
result: int = add_numbers("hello", 5)
''')
            
            result = subprocess.run(
                ['mypy', str(test_file)],
                capture_output=True,
                text=True
            )
            
            # Should detect type error
            assert result.returncode != 0
            output = result.stdout + result.stderr
            assert 'error' in output.lower()
    
    def test_mypy_return_type_mismatch(self):
        """Test MyPy detects return type mismatches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "return_mismatch.py"
            test_file.write_text('''
def get_number() -> int:
    return "not a number"  # Return type mismatch
''')
            
            result = subprocess.run(
                ['mypy', str(test_file)],
                capture_output=True,
                text=True
            )
            
            # Should detect return type error
            assert result.returncode != 0
            output = result.stdout + result.stderr
            assert 'error' in output.lower()
    
    def test_mypy_optional_types(self):
        """Test MyPy handles Optional types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "optional_types.py"
            test_file.write_text('''
from typing import Optional

def find_item(items: list[str], target: str) -> Optional[int]:
    """Find index of target, or None if not found."""
    try:
        return items.index(target)
    except ValueError:
        return None

result: Optional[int] = find_item(['a', 'b', 'c'], 'd')
''')
            
            result = subprocess.run(
                ['mypy', str(test_file)],
                capture_output=True,
                text=True
            )
            
            # Should pass
            assert result.returncode == 0


class TestMyPyGenericTypes:
    """Test MyPy generic type checking."""
    
    def test_mypy_list_types(self):
        """Test MyPy checks list element types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "list_types.py"
            test_file.write_text('''
def process_numbers(nums: list[int]) -> int:
    return sum(nums)

# This should work
good_result: int = process_numbers([1, 2, 3])

# This should error
bad_result: int = process_numbers(["a", "b"])
''')
            
            result = subprocess.run(
                ['mypy', str(test_file)],
                capture_output=True,
                text=True
            )
            
            # Should detect type error with string list
            assert result.returncode != 0
            output = result.stdout + result.stderr
            assert 'error' in output.lower()
    
    def test_mypy_dict_types(self):
        """Test MyPy checks dictionary types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "dict_types.py"
            test_file.write_text('''
def count_items(items: dict[str, int]) -> int:
    return sum(items.values())

good_dict: dict[str, int] = {"a": 1, "b": 2}
result: int = count_items(good_dict)
''')
            
            result = subprocess.run(
                ['mypy', str(test_file)],
                capture_output=True,
                text=True
            )
            
            # Should pass
            assert result.returncode == 0


class TestMyPyDataScience:
    """Test MyPy with data science packages."""
    
    def test_mypy_numpy_types(self):
        """Test MyPy with NumPy type stubs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "numpy_types.py"
            test_file.write_text('''
import numpy as np
from numpy.typing import NDArray

def process_array(arr: NDArray[np.float64]) -> float:
    """Process numpy array and return mean."""
    return float(np.mean(arr))

data: NDArray[np.float64] = np.array([1.0, 2.0, 3.0])
result: float = process_array(data)
''')
            
            result = subprocess.run(
                ['mypy', str(test_file)],
                capture_output=True,
                text=True
            )
            
            # Should pass (numpy has type stubs)
            # May warn about missing stubs, but shouldn't error
            assert 'error' not in result.stdout.lower() or result.returncode == 0
    
    def test_mypy_pandas_types(self):
        """Test MyPy with Pandas type stubs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "pandas_types.py"
            test_file.write_text('''
import pandas as pd

def process_dataframe(df: pd.DataFrame) -> int:
    """Get number of rows in DataFrame."""
    return len(df)

data: pd.DataFrame = pd.DataFrame({"a": [1, 2, 3]})
count: int = process_dataframe(data)
''')
            
            result = subprocess.run(
                ['mypy', str(test_file)],
                capture_output=True,
                text=True
            )
            
            # Should work (pandas has type stubs)
            # Some versions may have warnings but shouldn't fail on this code
            output = result.stdout.lower()
            # Accept if it passes or just has warnings about pandas-stubs
            assert result.returncode == 0 or 'pandas-stubs' in output


class TestMyPyConfiguration:
    """Test MyPy configuration options."""
    
    def test_mypy_strict_mode(self):
        """Test MyPy strict mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "untyped.py"
            test_file.write_text('''
def process(data):  # No type annotations
    return data * 2
''')
            
            # Normal mode might pass
            normal_result = subprocess.run(
                ['mypy', str(test_file)],
                capture_output=True,
                text=True
            )
            
            # Strict mode should complain about missing annotations
            strict_result = subprocess.run(
                ['mypy', '--strict', str(test_file)],
                capture_output=True,
                text=True
            )
            
            # Strict mode should detect missing annotations
            assert strict_result.returncode != 0
            strict_output = strict_result.stdout + strict_result.stderr
            assert 'error' in strict_output.lower()
    
    def test_mypy_ignore_errors(self):
        """Test MyPy type: ignore comments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "ignored.py"
            test_file.write_text('''
def add(a: int, b: int) -> int:
    return a + b

# This would normally error, but we ignore it
result: int = add("x", "y")  # type: ignore
''')
            
            result = subprocess.run(
                ['mypy', str(test_file)],
                capture_output=True,
                text=True
            )
            
            # Should pass because error is ignored
            assert result.returncode == 0
