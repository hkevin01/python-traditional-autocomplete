"""
Test File 4: Data Science Libraries AutoComplete

Test traditional autocomplete with PySpark, Pandas, NumPy, and Matplotlib.
NO AI assistance - only traditional IntelliSense!

INSTRUCTIONS:
1. Import libraries and see traditional import suggestions
2. Create objects and see method suggestions
3. Use Ctrl+Space for completion menu
4. NO Copilot ghost text should appear
"""

# TEST 1: Import suggestions
# Type 'import ' and see module suggestions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession

# TEST 2: NumPy autocomplete
# Type 'np.' and see NumPy functions
array_data = np.array([1, 2, 3, 4, 5])
# np.

# TEST 3: Type 'array_data.' to see array methods
mean_value = array_data.mean()
# array_data.

# TEST 4: Pandas DataFrame autocomplete
# Type 'pd.' to see pandas functions
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'salary': [50000, 60000, 70000]
})

# TEST 5: Type 'df.' to see DataFrame methods
# Should see: head, tail, describe, groupby, etc.
# df.

# TEST 6: Type 'df.groupby(' to see parameter hints
grouped = df.groupby('name')
# grouped.

# TEST 7: Matplotlib autocomplete
# Type 'plt.' to see plotting functions
# plt.

# Create a simple plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Type 'plt.plot(' to see parameter hints
plt.plot(x, y)
# plt.

# TEST 8: PySpark autocomplete
# Type 'SparkSession.' to see methods
spark = SparkSession.builder \
    .appName("TraditionalAutocompleteTest") \
    .getOrCreate()

# TEST 9: Create Spark DataFrame
# Type 'spark.' to see SparkSession methods
spark_df = spark.createDataFrame([
    (1, "Alice", 25),
    (2, "Bob", 30),
    (3, "Charlie", 35)
], ["id", "name", "age"])

# TEST 10: Type 'spark_df.' to see DataFrame methods
# Should see: show, select, filter, groupBy, etc.
# spark_df.

# TEST 11: NumPy array operations
# Type 'np.random.' to see random functions
random_data = np.random.rand(10)
# np.random.

# TEST 12: Type 'np.linalg.' to see linear algebra functions
matrix = np.array([[1, 2], [3, 4]])
# np.linalg.

# TEST 13: Pandas Series autocomplete
series = pd.Series([1, 2, 3, 4, 5])
# Type 'series.' to see Series methods
# series.

# TEST 14: Type annotation hints
def analyze_data(data: np.ndarray) -> dict:
    """
    Traditional autocomplete should show:
    - Parameter type: np.ndarray
    - Return type: dict
    - When calling, parameter hints appear
    """
    return {
        'mean': data.mean(),
        'std': data.std(),
        'min': data.min(),
        'max': data.max()
    }

# Type 'analyze_data(' to see parameter hints
result = analyze_data(array_data)

print("""
✅ DATA SCIENCE AUTOCOMPLETE TEST RESULTS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TRADITIONAL FEATURES (Should Work):
✓ np.  → NumPy function suggestions
✓ pd.  → Pandas function suggestions
✓ plt. → Matplotlib function suggestions
✓ df.  → DataFrame method suggestions
✓ spark_df. → PySpark DataFrame methods
✓ Hover over functions → See docstrings
✓ Type function( → See parameter hints

COPILOT FEATURES (Should NOT Work):
✗ NO ghost text for completing code
✗ NO AI-suggested data transformations
✗ NO automatic plot generation suggestions
✗ NO context-aware code completion

If you only see dropdown menus (no ghost text):
✅ SUCCESS! Copilot is disabled, traditional autocomplete works!
""")
