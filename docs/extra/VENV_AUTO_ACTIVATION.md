# 🔄 Virtual Environment Auto-Activation

## Overview

This workspace is configured to **automatically activate the Python virtual environment** when you open a terminal. No manual `source venv/bin/activate` needed!

---

## How It Works

### 1. VS Code Terminal Profile

The [.vscode/settings.json](../.vscode/settings.json) contains:

```json
{
  "terminal.integrated.defaultProfile.linux": "bash (venv)",
  "terminal.integrated.profiles.linux": {
    "bash (venv)": {
      "path": "bash",
      "args": ["--init-file", "${workspaceFolder}/.vscode/activate_venv.sh"]
    }
  }
}
```

This tells VS Code to use a custom bash profile that runs the activation script.

### 2. Activation Script

The [.vscode/activate_venv.sh](../.vscode/activate_venv.sh) script:
- Sources your `.bashrc` (preserves aliases, functions)
- Activates the virtual environment
- Sets a custom prompt showing `(venv)`
- Displays Python and pip paths

### 3. Python Extension Settings

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
  "python.terminal.activateEnvironment": true,
  "python.terminal.activateEnvInCurrentTerminal": true
}
```

These settings ensure Python extension also activates the venv.

---

## Verification

### Check If Auto-Activation Works

1. **Open a new terminal** (Ctrl+Shift+`)
2. You should see:
   ```
   ✅ Virtual environment activated: /path/to/venv
   🐍 Python: /path/to/venv/bin/python
   📦 pip: /path/to/venv/bin/pip
   ```
3. Your prompt should show `(venv)` prefix
4. Run: `which python`
   - Expected: `/path/to/workspace/venv/bin/python` ✅
   - NOT: `/usr/bin/python3` ❌

### Quick Test Commands

```bash
# Verify Python is from venv
which python
# Should show: .../venv/bin/python

# Check VIRTUAL_ENV variable
echo $VIRTUAL_ENV
# Should show: .../venv

# Verify pip is from venv
which pip
# Should show: .../venv/bin/pip

# Test installed packages
pip list | grep -E "pandas|numpy|pyspark"
# Should show installed data science packages
```

---

## What Happens When You Open Workspace

1. VS Code opens the workspace
2. You open a new terminal (or one opens automatically)
3. Terminal starts with `bash --init-file .vscode/activate_venv.sh`
4. Script loads your `.bashrc`
5. Script activates `venv/bin/activate`
6. You see activation confirmation
7. Ready to code! ✅

**No manual activation needed!**

---

## For Interviews

When presenting this workspace in interviews:

✅ **Just open the workspace** - venv activates automatically
✅ **Open terminal** - already activated
✅ **Run code** - uses correct Python/packages
✅ **Professional setup** - no fumbling with activation commands

**Demo script:**
```bash
# Open workspace in VS Code
code /path/to/python-traditional-autocomplete

# Open terminal (Ctrl+Shift+`) - venv auto-activates!
# Run your demo:
python example.py
# or
ipython
# or
jupyter notebook
```

---

## Troubleshooting

### Venv Not Activating?

1. **Check venv exists:**
   ```bash
   ls -la venv/bin/activate
   # Should exist
   ```

2. **Manually test activation script:**
   ```bash
   bash --init-file .vscode/activate_venv.sh
   # Should see activation message
   ```

3. **Verify settings.json:**
   ```bash
   cat .vscode/settings.json | grep "defaultProfile"
   # Should show: "bash (venv)"
   ```

4. **Reload VS Code window:**
   - Press `Ctrl+Shift+P`
   - Type: "Reload Window"
   - Try opening terminal again

### Create Venv If Missing

```bash
# From workspace root
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Alternative: Manual Activation

If you need a terminal WITHOUT auto-activation:

1. Open terminal profile selector (click `∨` next to `+`)
2. Select `bash` (not "bash (venv)")

Or in settings, change:
```json
{
  "terminal.integrated.defaultProfile.linux": "bash"
}
```

---

## Benefits

### ✅ For Development
- No manual activation needed
- Consistent environment every time
- Less cognitive load
- Professional setup

### ✅ For Interviews
- One-click workspace open
- Immediate readiness
- No terminal setup fumbling
- Focus on code, not environment

### ✅ For Team Sharing
- Same experience for everyone
- Reduces onboarding friction
- Documented and automated
- Version controlled

---

## Technical Details

### Why `--init-file` Instead of `--rcfile`?

- `--init-file`: Runs script and keeps shell interactive
- `--rcfile`: Replaces `.bashrc` (loses your aliases/functions)

We use `--init-file` and source `.bashrc` inside it to preserve your environment.

### Why Custom Profile Instead of `.bashrc`?

- Workspace-specific (doesn't affect system terminal)
- Version controlled with project
- Other users get same experience
- Easy to modify per-project

### What About Other Shells (zsh, fish)?

The current setup is for bash. For other shells:

**zsh:**
```json
{
  "terminal.integrated.profiles.linux": {
    "zsh (venv)": {
      "path": "zsh",
      "args": ["-c", "source ~/.zshrc && source venv/bin/activate && zsh"]
    }
  }
}
```

**fish:**
```json
{
  "terminal.integrated.profiles.linux": {
    "fish (venv)": {
      "path": "fish",
      "args": ["-C", "source venv/bin/activate.fish"]
    }
  }
}
```

---

## Summary

✅ Virtual environment activates automatically  
✅ No `source venv/bin/activate` needed  
✅ Works every time you open terminal  
✅ Perfect for interviews and demos  
✅ Professional, automated setup  

**Just open the workspace and start coding!** 🚀

---

*Configuration: [.vscode/settings.json](../.vscode/settings.json)*  
*Activation Script: [.vscode/activate_venv.sh](../.vscode/activate_venv.sh)*  
*Last Updated: January 22, 2026*
