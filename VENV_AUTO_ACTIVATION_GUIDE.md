# Virtual Environment Auto-Activation Guide

## ✅ What's Been Configured

Your workspace is now set up to **automatically activate the Python virtual environment** whenever you open a terminal in VSCode.

### Configuration Details

1. **VSCode Terminal Profile**: Custom bash profile that runs `.vscode/activate_venv.sh` on startup
2. **Environment Variables**: `VIRTUAL_ENV` and `PATH` automatically set in terminal
3. **Python Settings**: `python.terminal.activateEnvironment` enabled
4. **Window Title**: Shows "(venv)" indicator when environment is active

## 🚀 How It Works

### Automatic Activation (Primary Method)

When you open a new terminal in this workspace:
1. VSCode loads the custom terminal profile "bash (venv auto-activate)"
2. The `.vscode/activate_venv.sh` script runs automatically
3. Virtual environment activates with confirmation message
4. Prompt changes to show `(venv)` prefix
5. Python and pip commands use the venv versions

**Expected Terminal Output:**
```bash
✅ Virtual environment activated: /home/kevin/Projects/python-traditional-autocomplete/venv
🐍 Python: /home/kevin/Projects/python-traditional-autocomplete/venv/bin/python
📦 pip: /home/kevin/Projects/python-traditional-autocomplete/venv/bin/pip

Available packages:
pyspark      4.1.0
pandas       2.3.3
numpy        2.4.0
matplotlib   3.10.8

(venv) kevin@hostname:~/Projects/python-traditional-autocomplete$
```

### Manual Activation (Fallback Method)

If automatic activation doesn't work, run:
```bash
./activate_venv_manual.sh
```

Or the traditional way:
```bash
source venv/bin/activate
```

## 🔍 Verification

### Check if venv is active:
```bash
# Should show venv path
which python

# Should show venv path
echo $VIRTUAL_ENV

# Should see (venv) prefix
echo $PS1
```

### Test package availability:
```bash
python -c "import pyspark, pandas, numpy, matplotlib; print('✅ All packages available')"
```

## 🛠️ Troubleshooting

### Terminal doesn't auto-activate

**Solution 1**: Reload VSCode window
- Press `Ctrl+Shift+P`
- Type "Reload Window"
- Open new terminal

**Solution 2**: Check terminal profile
- Open new terminal
- Click dropdown arrow next to "+"
- Select "bash (venv auto-activate)"

**Solution 3**: Verify files exist
```bash
ls -la .vscode/activate_venv.sh
ls -la venv/bin/activate
```

**Solution 4**: Run manual activation
```bash
./activate_venv_manual.sh
```

### Venv activates but packages not found

**Check interpreter path:**
```bash
python --version
which python
pip list | grep -E "pyspark|pandas|numpy"
```

**Reinstall packages if needed:**
```bash
source venv/bin/activate
pip install pyspark pandas numpy matplotlib jupyter ipykernel
```

### Settings not applying

**Force settings reload:**
1. Close all terminals
2. Press `Ctrl+Shift+P`
3. Type "Preferences: Open Workspace Settings (JSON)"
4. Verify terminal settings are present
5. Save file
6. Reload window

## 📋 VSCode Settings Applied

The following settings ensure auto-activation:

```json
{
  "python.terminal.activateEnvironment": true,
  "python.terminal.activateEnvInCurrentTerminal": true,
  "terminal.integrated.env.linux": {
    "VIRTUAL_ENV": "${workspaceFolder}/venv",
    "PATH": "${workspaceFolder}/venv/bin:${env:PATH}"
  },
  "terminal.integrated.defaultProfile.linux": "bash (venv auto-activate)",
  "terminal.integrated.profiles.linux": {
    "bash (venv auto-activate)": {
      "path": "bash",
      "args": ["--init-file", "${workspaceFolder}/.vscode/activate_venv.sh"]
    }
  }
}
```

## �� Success Indicators

When venv is properly activated, you should see:

1. ✅ Terminal prompt shows `(venv)` prefix
2. ✅ Window title shows "[COPILOT DISABLED] - (venv)"
3. ✅ `which python` points to venv/bin/python
4. ✅ `pip list` shows PySpark, Pandas, NumPy, Matplotlib
5. ✅ Python code can import all packages without errors
6. ✅ Jupyter kernel shows "Python (No Copilot)"

## 📁 Files Created

- `.vscode/settings.json` - VSCode configuration with terminal auto-activation
- `.vscode/activate_venv.sh` - Automatic activation script for terminals
- `activate_venv_manual.sh` - Manual activation fallback script
- `VENV_AUTO_ACTIVATION_GUIDE.md` - This documentation file

## 🔄 Usage Workflow

### Every time you open the workspace:

1. Open VSCode to this project
2. Open terminal (`Ctrl+` ` or Terminal → New Terminal)
3. Venv activates automatically
4. Start coding - all packages available!

### No manual activation needed! Just open and code.

## ⚡ Quick Commands

```bash
# Verify venv is active
echo $VIRTUAL_ENV

# List installed packages
pip list

# Test data science imports
python -c "import pyspark, pandas as pd, numpy as np, matplotlib.pyplot as plt; print('✅ Success')"

# Open Jupyter notebook (will use venv kernel)
jupyter notebook

# Run Python file with venv Python
python src/test_data_science.py

# Deactivate venv (if needed)
deactivate
```

## 🆘 Still Having Issues?

If auto-activation still doesn't work after trying all troubleshooting steps:

1. Check VSCode terminal settings in UI:
   - File → Preferences → Settings
   - Search: "terminal integrated default profile"
   - Should show "bash (venv auto-activate)"

2. Manually set Python interpreter:
   - Press `Ctrl+Shift+P`
   - Type "Python: Select Interpreter"
   - Choose `./venv/bin/python`

3. Create new terminal:
   - Close all existing terminals
   - Open fresh terminal
   - Should auto-activate

4. Check terminal type:
   - Ensure using "bash" not "sh" or "zsh"
   - Click terminal dropdown to select correct profile

---

**Last Updated**: January 7, 2026
**Virtual Environment Path**: `~/Projects/python-traditional-autocomplete/venv`
**Python Version**: 3.12
**Key Packages**: PySpark 4.1.0, Pandas 2.3.3, NumPy 2.4.0, Matplotlib 3.10.8
