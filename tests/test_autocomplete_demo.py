"""
Demo file to test traditional autocomplete (no AI)
Open this file in VS Code and try the following:
"""

# Test 1: Import autocomplete
# Type: import pan
# You should see "pandas" in autocomplete suggestions

# Test 2: Method autocomplete  
import pandas as pd
import numpy as np

df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

# Type: df.
# You should see all DataFrame methods (head, tail, describe, etc.)

# Test 3: Parameter hints
# Type: df.head(
# You should see parameter hints showing (n=5)

# Test 4: NumPy autocomplete
arr = np.array([1, 2, 3])

# Type: arr.
# You should see all ndarray methods

# Test 5: Type hints work
def process_data(data: pd.DataFrame) -> pd.DataFrame:
    # Type: data.
    # IntelliSense knows 'data' is a DataFrame
    return data.head()

# Test 6: Custom class autocomplete
class MyClass:
    def __init__(self):
        self.my_property = "test"
    
    def my_method(self):
        return "result"

obj = MyClass()
# Type: obj.
# You should see: my_property and my_method

print("✅ All tests ready - try typing in this file!")
