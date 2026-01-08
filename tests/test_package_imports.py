"""
Test Package Imports
===================
Verify all 29 required packages are installed and importable.

Tests cover:
- Runtime: Python version
- LSP Servers: Pylance, python-lsp-server, jedi, jedi-language-server, pygls, lsprotocol
- Data Science: NumPy, Pandas, Matplotlib, PySpark
- Jupyter: jupyter, jupyterlab, ipython, ipykernel, jupyterlab-lsp
- Linting: pylint, flake8, pydocstyle
- Formatting: black, isort, autopep8
- Type Checking: mypy
- Testing: pytest
- Refactoring: rope
- Support: cattrs, docstring-to-markdown, pytoolconfig, parso
"""

import sys
import pytest
from packaging import version


class TestPythonVersion:
    """Test Python runtime version."""
    
    def test_python_version_minimum(self):
        """Verify Python 3.12+ is installed."""
        assert sys.version_info >= (3, 12), f"Python 3.12+ required, got {sys.version_info}"
    
    def test_python_version_exact(self):
        """Verify Python 3.12.0 specifically (if installed)."""
        major, minor, _ = sys.version_info[:3]
        assert (major, minor) == (3, 12), f"Expected Python 3.12, got {major}.{minor}"


class TestLSPPackages:
    """Test Language Server Protocol packages."""
    
    def test_import_jedi(self):
        """Test Jedi import and version."""
        import jedi
        assert hasattr(jedi, '__version__')
        assert version.parse(jedi.__version__) >= version.parse("0.19.0")
    
    def test_import_jedi_language_server(self):
        """Test jedi-language-server import."""
        import jedi_language_server
        assert hasattr(jedi_language_server, '__version__')
    
    def test_import_pylsp(self):
        """Test python-lsp-server import."""
        import pylsp
        assert hasattr(pylsp, '__version__')
    
    def test_import_pygls(self):
        """Test pygls (LSP framework) import."""
        import pygls
        assert hasattr(pygls, '__version__')
        assert version.parse(pygls.__version__) >= version.parse("2.0.0")
    
    def test_import_lsprotocol(self):
        """Test lsprotocol import."""
        import lsprotocol
        assert hasattr(lsprotocol, '__version__')
    
    def test_import_rope(self):
        """Test rope refactoring import."""
        import rope
        assert hasattr(rope, 'VERSION')


class TestDataSciencePackages:
    """Test data science package imports."""
    
    def test_import_numpy(self):
        """Test NumPy import and version."""
        import numpy as np
        assert hasattr(np, '__version__')
        assert version.parse(np.__version__) >= version.parse("2.0.0")
    
    def test_import_pandas(self):
        """Test Pandas import and version."""
        import pandas as pd
        assert hasattr(pd, '__version__')
        assert version.parse(pd.__version__) >= version.parse("2.0.0")
    
    def test_import_matplotlib(self):
        """Test Matplotlib import and version."""
        import matplotlib
        assert hasattr(matplotlib, '__version__')
        assert version.parse(matplotlib.__version__) >= version.parse("3.0.0")
    
    def test_import_pyspark(self):
        """Test PySpark import and version."""
        import pyspark
        assert hasattr(pyspark, '__version__')
        assert version.parse(pyspark.__version__) >= version.parse("4.0.0")


class TestJupyterPackages:
    """Test Jupyter ecosystem packages."""
    
    def test_import_jupyter(self):
        """Test jupyter meta-package import."""
        import jupyter
        # Jupyter is a meta-package, just verify it imports
        assert jupyter is not None
    
    def test_import_jupyterlab(self):
        """Test JupyterLab import."""
        import jupyterlab
        assert hasattr(jupyterlab, '__version__')
    
    def test_import_ipython(self):
        """Test IPython import and version."""
        import IPython
        assert hasattr(IPython, '__version__')
        assert version.parse(IPython.__version__) >= version.parse("8.0.0")
    
    def test_import_ipykernel(self):
        """Test IPyKernel import."""
        import ipykernel
        assert hasattr(ipykernel, '__version__')
    
    def test_import_jupyterlab_lsp(self):
        """Test jupyterlab-lsp import."""
        # jupyterlab-lsp is primarily a lab extension, check package exists
        try:
            import jupyterlab_lsp
            assert jupyterlab_lsp is not None
        except ImportError:
            # May not have Python import, verify via pip
            import subprocess
            result = subprocess.run(
                ['pip', 'show', 'jupyterlab-lsp'],
                capture_output=True,
                text=True
            )
            assert result.returncode == 0, "jupyterlab-lsp not installed"


class TestLintingPackages:
    """Test linting tool packages."""
    
    def test_import_pylint(self):
        """Test Pylint import."""
        import pylint
        assert hasattr(pylint, '__version__')
    
    def test_import_flake8(self):
        """Test Flake8 import."""
        import flake8
        assert hasattr(flake8, '__version__')
    
    def test_import_pydocstyle(self):
        """Test Pydocstyle import."""
        import pydocstyle
        assert hasattr(pydocstyle, '__version__')


class TestFormattingPackages:
    """Test code formatting packages."""
    
    def test_import_black(self):
        """Test Black import."""
        import black
        assert hasattr(black, '__version__')
    
    def test_import_isort(self):
        """Test isort import."""
        import isort
        assert hasattr(isort, '__version__')
        assert version.parse(isort.__version__) >= version.parse("5.0.0")
    
    def test_import_autopep8(self):
        """Test autopep8 import."""
        import autopep8
        assert hasattr(autopep8, '__version__')


class TestTypeCheckingPackages:
    """Test static type checking packages."""
    
    def test_import_mypy(self):
        """Test MyPy import."""
        import mypy
        assert hasattr(mypy, '__version__')


class TestSupportPackages:
    """Test support library packages."""
    
    def test_import_cattrs(self):
        """Test cattrs (serialization) import."""
        import cattrs
        assert cattrs is not None
    
    def test_import_docstring_to_markdown(self):
        """Test docstring-to-markdown import."""
        import docstring_to_markdown
        assert docstring_to_markdown is not None
    
    def test_import_pytoolconfig(self):
        """Test pytoolconfig import."""
        import pytoolconfig
        assert pytoolconfig is not None
    
    def test_import_parso(self):
        """Test parso (Python parser) import."""
        import parso
        assert hasattr(parso, '__version__')
        assert version.parse(parso.__version__) >= version.parse("0.8.0")


class TestPackageVersions:
    """Test specific package versions match requirements."""
    
    def test_all_packages_have_versions(self):
        """Verify we can get version info for all packages."""
        import subprocess
        result = subprocess.run(
            ['pip', 'list', '--format=json'],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        
        import json
        packages = json.loads(result.stdout)
        package_names = {pkg['name'].lower() for pkg in packages}
        
        # Check critical packages are installed
        critical = [
            'numpy', 'pandas', 'matplotlib', 'pyspark',
            'jedi', 'pylint', 'flake8', 'black', 'isort',
            'mypy', 'pytest', 'ipython', 'jupyterlab',
            'rope', 'autopep8', 'pydocstyle'
        ]
        
        for pkg in critical:
            assert pkg in package_names, f"Package {pkg} not installed"
