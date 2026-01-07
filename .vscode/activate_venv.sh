#!/bin/bash

# Load system bash initialization
if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi

# Determine workspace folder
if [ -n "$VSCODE_WORKSPACE_FOLDER" ]; then
    WORKSPACE_DIR="$VSCODE_WORKSPACE_FOLDER"
else
    WORKSPACE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
fi

# Activate virtual environment
VENV_PATH="$WORKSPACE_DIR/venv"

if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
    echo "✅ Virtual environment activated: $VENV_PATH"
    echo "🐍 Python: $(which python)"
    echo "📦 pip: $(which pip)"
    echo ""
    echo "Available packages:"
    pip list --format=columns | grep -E "(pyspark|pandas|numpy|matplotlib)" || echo "  Loading..."
else
    echo "❌ Virtual environment not found at: $VENV_PATH"
    echo "Run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
fi

# Set PS1 prompt to show venv activation
export PS1="(venv) \[\e[32m\]\u@\h\[\e[0m\]:\[\e[34m\]\w\[\e[0m\]\$ "
