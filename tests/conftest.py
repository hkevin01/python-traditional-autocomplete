"""
Pytest Configuration
===================
Shared fixtures and configuration for all tests.
"""

import pytest
import sys
import warnings
from pathlib import Path


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment before running tests."""
    # Add project root to path
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Filter warnings for cleaner output
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
    
    yield
    
    # Cleanup after tests
    pass


@pytest.fixture
def sample_python_code():
    """Provide sample Python code for testing."""
    return '''
"""Sample module for testing."""

import os
import sys
from typing import List, Optional


def calculate_sum(numbers: List[int]) -> int:
    """Calculate sum of numbers.
    
    Args:
        numbers: List of integers
    
    Returns:
        Sum of all numbers
    """
    return sum(numbers)


def find_max(numbers: List[int]) -> Optional[int]:
    """Find maximum number.
    
    Args:
        numbers: List of integers
    
    Returns:
        Maximum number or None if empty
    """
    if not numbers:
        return None
    return max(numbers)


class Calculator:
    """Simple calculator class."""
    
    def __init__(self):
        """Initialize calculator."""
        self.result = 0
    
    def add(self, value: int) -> int:
        """Add value to result.
        
        Args:
            value: Value to add
        
        Returns:
            New result
        """
        self.result += value
        return self.result
    
    def reset(self) -> None:
        """Reset calculator to zero."""
        self.result = 0


if __name__ == "__main__":
    numbers = [1, 2, 3, 4, 5]
    print(f"Sum: {calculate_sum(numbers)}")
    print(f"Max: {find_max(numbers)}")
'''


@pytest.fixture
def sample_notebook_json():
    """Provide sample Jupyter notebook structure."""
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# Test Notebook\n", "This is a test."]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["import numpy as np\n", "x = np.array([1, 2, 3])"]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }


@pytest.fixture
def temp_python_file(tmp_path):
    """Create a temporary Python file for testing."""
    def _create_file(content: str, filename: str = "test.py"):
        file_path = tmp_path / filename
        file_path.write_text(content)
        return file_path
    
    return _create_file


# Markers for test categories
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "lsp: marks tests for LSP functionality"
    )
    config.addinivalue_line(
        "markers", "jupyter: marks tests for Jupyter functionality"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests that measure performance"
    )


# Skip tests if dependencies are missing
def pytest_runtest_setup(item):
    """Setup for individual test runs."""
    # Check for required markers and skip if needed
    pass
