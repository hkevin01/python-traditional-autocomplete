"""
Test LSP Tools Functionality
===========================
Test Language Server Protocol tools work correctly.

Tests:
- Jedi: Completion, definition, references
- python-lsp-server: Server initialization
- pygls: LSP framework functionality
- lsprotocol: Protocol message handling
"""

import pytest
from pathlib import Path
import tempfile


class TestJedi:
    """Test Jedi autocomplete functionality."""
    
    def test_jedi_completion(self):
        """Test Jedi provides completions."""
        import jedi
        
        source = "import os\nos."
        script = jedi.Script(source, path="test.py")
        completions = script.complete(2, 3)
        
        assert len(completions) > 0, "Jedi should provide completions for 'os.'"
        completion_names = [c.name for c in completions]
        assert 'path' in completion_names, "Should suggest os.path"
    
    def test_jedi_goto_definition(self):
        """Test Jedi goto definition."""
        import jedi
        
        source = """
def my_function():
    return 42

result = my_function()
"""
        script = jedi.Script(source, path="test.py")
        definitions = script.goto(5, 9)  # Position at my_function call
        
        assert len(definitions) > 0, "Should find function definition"
        assert definitions[0].line == 2, "Should point to function definition line"
    
    def test_jedi_inference(self):
        """Test Jedi type inference."""
        import jedi
        
        source = """
x = [1, 2, 3]
x.
"""
        script = jedi.Script(source, path="test.py")
        completions = script.complete(3, 2)
        
        completion_names = [c.name for c in completions]
        assert 'append' in completion_names, "Should infer list type and suggest append"
        assert 'pop' in completion_names, "Should suggest list methods"
    
    def test_jedi_numpy_completion(self):
        """Test Jedi completes NumPy correctly."""
        import jedi
        
        source = "import numpy as np\nnp.ar"
        script = jedi.Script(source, path="test.py")
        completions = script.complete(2, 5)
        
        completion_names = [c.name for c in completions]
        assert 'array' in completion_names, "Should suggest np.array"
        assert 'arange' in completion_names, "Should suggest np.arange"


class TestPythonLSP:
    """Test python-lsp-server functionality."""
    
    def test_pylsp_imports(self):
        """Test python-lsp-server imports."""
        from pylsp import lsp
        from pylsp import hookspecs
        
        assert hasattr(lsp, 'CompletionItem')
        assert hasattr(hookspecs, 'pylsp_completions')
    
    def test_pylsp_workspace(self):
        """Test python-lsp-server workspace handling."""
        from pylsp.workspace import Workspace, Document
        from pylsp.config.config import Config
        
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Workspace(tmpdir, None)
            assert workspace.root_path == tmpdir
            
            # Test document creation
            doc_uri = f"file://{tmpdir}/test.py"
            source = "import os\n"
            doc = Document(doc_uri, workspace, source)
            assert doc.source == source


class TestPygls:
    """Test pygls LSP framework."""
    
    def test_pygls_server_creation(self):
        """Test creating an LSP server with pygls."""
        from pygls.server import LanguageServer
        
        server = LanguageServer('test-server', 'v0.1')
        assert server.name == 'test-server'
        assert server.version == 'v0.1'
    
    def test_pygls_feature_decorator(self):
        """Test pygls feature registration."""
        from pygls.server import LanguageServer
        from lsprotocol.types import TEXT_DOCUMENT_COMPLETION
        
        server = LanguageServer('test', 'v1')
        
        @server.feature(TEXT_DOCUMENT_COMPLETION)
        def completions(params):
            return []
        
        assert completions is not None


class TestLSProtocol:
    """Test lsprotocol types and messages."""
    
    def test_lsprotocol_types(self):
        """Test lsprotocol type definitions."""
        from lsprotocol import types
        
        # Test key LSP types exist
        assert hasattr(types, 'Position')
        assert hasattr(types, 'Range')
        assert hasattr(types, 'TextDocumentItem')
        assert hasattr(types, 'CompletionItem')
    
    def test_lsprotocol_position(self):
        """Test Position type creation."""
        from lsprotocol.types import Position
        
        pos = Position(line=10, character=5)
        assert pos.line == 10
        assert pos.character == 5
    
    def test_lsprotocol_range(self):
        """Test Range type creation."""
        from lsprotocol.types import Position, Range
        
        start = Position(line=0, character=0)
        end = Position(line=0, character=10)
        range_obj = Range(start=start, end=end)
        
        assert range_obj.start.line == 0
        assert range_obj.end.character == 10
    
    def test_lsprotocol_completion_item(self):
        """Test CompletionItem creation."""
        from lsprotocol.types import CompletionItem, CompletionItemKind
        
        item = CompletionItem(
            label="test_function",
            kind=CompletionItemKind.Function
        )
        
        assert item.label == "test_function"
        assert item.kind == CompletionItemKind.Function


class TestRope:
    """Test Rope refactoring functionality."""
    
    def test_rope_project_creation(self):
        """Test creating a Rope project."""
        from rope.base.project import Project
        
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Project(tmpdir)
            assert project.root.exists()
            assert project.address == tmpdir
            project.close()
    
    def test_rope_rename_refactoring(self):
        """Test Rope rename refactoring."""
        from rope.base.project import Project
        from rope.refactor.rename import Rename
        
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Project(tmpdir)
            
            # Create a test file
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("def old_name():\n    pass\n")
            
            resource = project.root.get_file('test.py')
            renamer = Rename(project, resource, 4)  # Position of old_name
            
            # Just verify we can create the refactoring object
            assert renamer is not None
            
            project.close()
    
    def test_rope_completion(self):
        """Test Rope code completion."""
        from rope.base.project import Project
        from rope.contrib.codeassist import code_assist
        
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Project(tmpdir)
            
            source = "import os\nos."
            proposals = code_assist(project, source, len(source))
            
            assert len(proposals) > 0, "Should provide completions for os."
            
            project.close()
