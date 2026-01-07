#!/bin/bash
# Manual venv activation script
# Run this if VSCode terminal auto-activation doesn't work

echo "🔧 Activating Python virtual environment..."
source venv/bin/activate

echo "✅ Virtual environment activated!"
echo "🐍 Python: $(which python)"
echo "�� pip: $(which pip)"
echo ""
echo "Installed packages:"
pip list --format=columns | head -20
