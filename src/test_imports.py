"""
Test File 1: Import AutoComplete Testing

INSTRUCTIONS:
1. Place cursor after 'import ' and press Ctrl+Space
2. You should see a list of available modules
3. NO Copilot suggestions should appear (no ghost text)
4. Try typing 'import os.' and see method suggestions

Expected: Traditional IntelliSense only, NO AI completions
"""

# TEST 1: Basic imports - Type 'import ' and see suggestions
import os
import sys
import json

# TEST 2: From imports - Type 'from os import ' and see suggestions
from os import path, environ
from typing import List, Dict, Optional, Tuple

# TEST 3: Type hints - Hover over these to see type info
def process_data(data: List[int]) -> Dict[str, float]:
    """
    Traditional autocomplete should show:
    - Parameter hints when you call this function
    - Return type information
    - Method suggestions for List and Dict
    """
    # TEST 4: Type 'data.' and see List methods (append, extend, etc.)
    result = {
        "sum": sum(data),
        "length": len(data)
    }
    return result

# TEST 5: Type 'os.' and see os module methods
current_path = os.getcwd()
home_dir = os.path.expanduser("~")

# TEST 6: Type 'sys.' and see sys module methods  
python_version = sys.version
platform_info = sys.platform

# TEST 7: Dictionary methods - Type 'my_dict.' to see methods
my_dict = {"key": "value"}
# my_dict.

# TEST 8: String methods - Type 'my_string.' to see methods
my_string = "hello world"
# my_string.

# TEST 9: Type 'json.' to see json module methods
data_str = json.dumps({"test": "data"})
# json.

print("✅ If you only see traditional suggestions (no ghost text), Copilot is OFF!")
