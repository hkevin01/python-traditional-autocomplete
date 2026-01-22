#!/bin/bash

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                                                                      ║"
echo "║        🔍 Virtual Environment Auto-Activation Verification 🔍        ║"
echo "║                                                                      ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 Checking Configuration Files"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if settings.json has terminal profile configured
if grep -q "bash (venv)" .vscode/settings.json 2>/dev/null; then
    echo -e "✅ ${GREEN}Terminal profile configured in settings.json${NC}"
else
    echo -e "❌ ${RED}Terminal profile NOT found in settings.json${NC}"
fi

# Check if activate_venv.sh exists
if [ -f ".vscode/activate_venv.sh" ]; then
    echo -e "✅ ${GREEN}Activation script exists: .vscode/activate_venv.sh${NC}"
else
    echo -e "❌ ${RED}Activation script NOT found${NC}"
fi

# Check if activate_venv.sh is executable
if [ -x ".vscode/activate_venv.sh" ]; then
    echo -e "✅ ${GREEN}Activation script is executable${NC}"
else
    echo -e "⚠️  ${YELLOW}Activation script exists but not executable (OK for --init-file)${NC}"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🐍 Checking Virtual Environment"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if venv exists
if [ -d "venv" ]; then
    echo -e "✅ ${GREEN}Virtual environment directory exists${NC}"
    
    # Check if activate script exists
    if [ -f "venv/bin/activate" ]; then
        echo -e "✅ ${GREEN}venv/bin/activate exists${NC}"
    else
        echo -e "❌ ${RED}venv/bin/activate NOT found${NC}"
    fi
    
    # Check if python exists in venv
    if [ -f "venv/bin/python" ]; then
        echo -e "✅ ${GREEN}venv/bin/python exists${NC}"
        VENV_PYTHON_VERSION=$(venv/bin/python --version 2>&1)
        echo "   Python version: $VENV_PYTHON_VERSION"
    else
        echo -e "❌ ${RED}venv/bin/python NOT found${NC}"
    fi
else
    echo -e "❌ ${RED}Virtual environment directory NOT found${NC}"
    echo ""
    echo "Create it with:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔬 Testing Current Terminal Session"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if VIRTUAL_ENV is set
if [ -n "$VIRTUAL_ENV" ]; then
    echo -e "✅ ${GREEN}VIRTUAL_ENV is set${NC}"
    echo "   Value: $VIRTUAL_ENV"
else
    echo -e "⚠️  ${YELLOW}VIRTUAL_ENV is NOT set (this terminal may not have venv active)${NC}"
    echo "   This is OK if you're testing the script directly"
    echo "   Open a NEW terminal in VS Code to see auto-activation"
fi

# Check which python is being used
CURRENT_PYTHON=$(which python 2>/dev/null || which python3)
if [ -n "$CURRENT_PYTHON" ]; then
    if [[ "$CURRENT_PYTHON" == *"venv/bin/python"* ]]; then
        echo -e "✅ ${GREEN}Using Python from venv${NC}"
        echo "   Path: $CURRENT_PYTHON"
    else
        echo -e "⚠️  ${YELLOW}Using system Python (venv not activated in this terminal)${NC}"
        echo "   Path: $CURRENT_PYTHON"
        echo "   Open a NEW terminal in VS Code to see auto-activation"
    fi
fi

# Show Python version
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1)
    echo "   Python version: $PYTHON_VERSION"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 Checking Installed Packages"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ -n "$VIRTUAL_ENV" ] && command -v pip &> /dev/null; then
    echo "Key packages installed:"
    pip list 2>/dev/null | grep -E "(pandas|numpy|pyspark|matplotlib|jupyter)" || echo "  No key packages found"
else
    echo "⚠️  Can't check packages (venv not active or pip not available)"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "💡 Next Steps"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ -n "$VIRTUAL_ENV" ]; then
    echo -e "${GREEN}✅ Virtual environment is ACTIVE in this terminal!${NC}"
    echo ""
    echo "You're all set! Try:"
    echo "  python -c 'import pandas; print(pandas.__version__)'"
    echo "  python -c 'import numpy; print(numpy.__version__)'"
else
    echo -e "${YELLOW}⚠️  Virtual environment NOT active in this terminal${NC}"
    echo ""
    echo "To test auto-activation:"
    echo "  1. Close this terminal"
    echo "  2. Open a NEW terminal in VS Code (Ctrl+Shift+\`)"
    echo "  3. You should see activation message and (venv) in prompt"
    echo ""
    echo "Or manually test the activation script:"
    echo "  bash --init-file .vscode/activate_venv.sh"
fi

echo ""
echo "📖 Read full documentation:"
echo "   → docs/VENV_AUTO_ACTIVATION.md"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
