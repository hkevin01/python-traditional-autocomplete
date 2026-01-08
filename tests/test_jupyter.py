"""
Test Jupyter Ecosystem
=====================
Test Jupyter, JupyterLab, IPython, IPyKernel functionality.

Tests notebook kernel execution, completion, and LSP integration.
"""

import pytest
import subprocess
import json
from pathlib import Path
import tempfile


class TestIPython:
    """Test IPython functionality."""
    
    def test_ipython_import(self):
        """Test IPython imports."""
        import IPython
        from IPython.core.interactiveshell import InteractiveShell
        
        assert InteractiveShell is not None
    
    def test_ipython_completion(self):
        """Test IPython TAB completion."""
        from IPython.core.completer import IPCompleter
        from IPython.core.interactiveshell import InteractiveShell
        
        shell = InteractiveShell.instance()
        completer = IPCompleter(shell=shell, namespace={'x': [1, 2, 3]})
        
        # Test completion on list object
        completions = completer.completions('x.', 2)
        completion_texts = [c.text for c in completions]
        
        assert 'append' in completion_texts
        assert 'pop' in completion_texts
    
    def test_ipython_magic_commands(self):
        """Test IPython magic commands."""
        from IPython.core.magic import register_line_magic
        from IPython.core.interactiveshell import InteractiveShell
        
        shell = InteractiveShell.instance()
        
        # Test that built-in magics exist
        assert 'timeit' in shell.magics_manager.magics['line']
        assert 'matplotlib' in shell.magics_manager.magics['line']
    
    def test_ipython_execution(self):
        """Test IPython code execution."""
        from IPython.core.interactiveshell import InteractiveShell
        
        shell = InteractiveShell.instance()
        result = shell.run_cell('2 + 2')
        
        assert result.success is True
        assert result.result == 4


class TestIPyKernel:
    """Test IPyKernel functionality."""
    
    def test_ipykernel_import(self):
        """Test IPyKernel imports."""
        from ipykernel.kernelapp import IPKernelApp
        from ipykernel.ipkernel import IPythonKernel
        
        assert IPKernelApp is not None
        assert IPythonKernel is not None
    
    def test_ipykernel_list(self):
        """Test listing installed Jupyter kernels."""
        result = subprocess.run(
            ['jupyter', 'kernelspec', 'list', '--json'],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        kernels = json.loads(result.stdout)
        
        # Should have at least one Python kernel
        assert 'kernelspecs' in kernels
        kernel_names = list(kernels['kernelspecs'].keys())
        
        # Check for Python kernel (could be 'python3', 'python', etc.)
        python_kernels = [k for k in kernel_names if 'python' in k.lower()]
        assert len(python_kernels) > 0, f"No Python kernel found. Available: {kernel_names}"


class TestJupyterLab:
    """Test JupyterLab functionality."""
    
    def test_jupyterlab_import(self):
        """Test JupyterLab imports."""
        import jupyterlab
        from jupyterlab.labapp import LabApp
        
        assert LabApp is not None
    
    def test_jupyterlab_version(self):
        """Test JupyterLab version command."""
        result = subprocess.run(
            ['jupyter', 'lab', '--version'],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert 'jupyter' in result.stdout.lower() or result.stdout.strip()
    
    def test_jupyterlab_extensions(self):
        """Test listing JupyterLab extensions."""
        result = subprocess.run(
            ['jupyter', 'labextension', 'list'],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        # Should have output (even if no extensions installed)
        assert result.stdout or result.stderr


class TestJupyterLSP:
    """Test Jupyter LSP functionality."""
    
    def test_jupyterlab_lsp_installed(self):
        """Test jupyterlab-lsp package is installed."""
        result = subprocess.run(
            ['pip', 'show', 'jupyterlab-lsp'],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert 'jupyterlab-lsp' in result.stdout
    
    def test_jedi_language_server_installed(self):
        """Test jedi-language-server is installed."""
        result = subprocess.run(
            ['pip', 'show', 'jedi-language-server'],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert 'jedi-language-server' in result.stdout
    
    def test_jedi_language_server_executable(self):
        """Test jedi-language-server is executable."""
        result = subprocess.run(
            ['jedi-language-server', '--help'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        # Should not error (exit code 0 or 2 for help)
        assert result.returncode in [0, 2]


class TestNotebookExecution:
    """Test Jupyter notebook execution."""
    
    def test_notebook_execution_simple(self):
        """Test executing a simple notebook."""
        with tempfile.TemporaryDirectory() as tmpdir:
            notebook_path = Path(tmpdir) / "test.ipynb"
            
            # Create a simple notebook
            notebook = {
                "cells": [
                    {
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": ["result = 2 + 2\nprint(result)"]
                    }
                ],
                "metadata": {
                    "kernelspec": {
                        "display_name": "Python 3",
                        "language": "python",
                        "name": "python3"
                    },
                    "language_info": {
                        "name": "python",
                        "version": "3.12.0"
                    }
                },
                "nbformat": 4,
                "nbformat_minor": 5
            }
            
            notebook_path.write_text(json.dumps(notebook))
            
            # Execute notebook using jupyter nbconvert
            result = subprocess.run(
                [
                    'jupyter', 'nbconvert',
                    '--to', 'notebook',
                    '--execute',
                    '--stdout',
                    str(notebook_path)
                ],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            assert result.returncode == 0, f"Notebook execution failed: {result.stderr}"
            
            # Parse output notebook
            output_notebook = json.loads(result.stdout)
            cell_outputs = output_notebook['cells'][0]['outputs']
            
            # Should have printed "4"
            assert len(cell_outputs) > 0
            output_text = cell_outputs[0].get('text', [''])[0]
            assert '4' in output_text
    
    def test_notebook_with_imports(self):
        """Test notebook with data science imports."""
        with tempfile.TemporaryDirectory() as tmpdir:
            notebook_path = Path(tmpdir) / "test_imports.ipynb"
            
            notebook = {
                "cells": [
                    {
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": [
                            "import numpy as np\n",
                            "import pandas as pd\n",
                            "result = np.array([1, 2, 3]).sum()\n",
                            "print(f'Sum: {result}')"
                        ]
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
            
            notebook_path.write_text(json.dumps(notebook))
            
            result = subprocess.run(
                [
                    'jupyter', 'nbconvert',
                    '--to', 'notebook',
                    '--execute',
                    '--stdout',
                    str(notebook_path)
                ],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            assert result.returncode == 0
            output_notebook = json.loads(result.stdout)
            cell_outputs = output_notebook['cells'][0]['outputs']
            
            assert len(cell_outputs) > 0
            output_text = cell_outputs[0].get('text', [''])[0]
            assert 'Sum: 6' in output_text
