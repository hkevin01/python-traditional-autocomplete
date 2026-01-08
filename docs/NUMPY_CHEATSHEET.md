# NumPy 2.4.0 Comprehensive Cheatsheet

**Level:** SparkNotes/Unbound Quality Reference  
**Date:** January 8, 2026  
**NumPy Version:** 2.4.0+  
**Performance:** 125x speedup vs pure Python

## Table of Contents

- [Quick Start & Core Concepts](#quick-start--core-concepts)
- [Array Creation](#array-creation)
- [Array Attributes & Info](#array-attributes--info)
- [Indexing & Slicing](#indexing--slicing)
- [Array Manipulation](#array-manipulation)
- [Mathematical Operations](#mathematical-operations)
- [Statistical Functions](#statistical-functions)
- [Linear Algebra](#linear-algebra)
- [Broadcasting](#broadcasting)
- [Advanced Indexing](#advanced-indexing)
- [Performance Optimization](#performance-optimization)
- [Common Patterns](#common-patterns)

---

## Quick Start & Core Concepts

### Import Convention
```python
import numpy as np
import numpy.typing as npt  # For type hints
```

### ndarray - The Core Data Structure
```python
# NumPy's main object is the homogeneous multidimensional array (ndarray)
# Elements are all the same type, indexed by positive integers
arr = np.array([1, 2, 3, 4, 5])
print(type(arr))          # <class 'numpy.ndarray'>

# Key properties
print(arr.shape)          # (5,) - dimensions
print(arr.dtype)          # int64 - data type
print(arr.size)           # 5 - total number of elements
print(arr.ndim)           # 1 - number of dimensions
```

---

## Array Creation

### From Python Sequences
```python
# 1D arrays
arr1d = np.array([1, 2, 3, 4, 5])
arr1d = np.array((1, 2, 3, 4, 5))  # From tuple

# 2D arrays (matrices)
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
arr2d = np.array([(1, 2, 3), (4, 5, 6)])

# 3D arrays
arr3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

# Specify data type
arr = np.array([1, 2, 3], dtype=np.float64)
arr = np.array(['a', 'b', 'c'], dtype='U1')  # Unicode strings
```

### Built-in Creation Functions
```python
# Zeros and ones
np.zeros(5)               # [0. 0. 0. 0. 0.]
np.zeros((2, 3))          # 2x3 array of zeros
np.ones((2, 3, 4))        # 2x3x4 array of ones
np.full((2, 3), 7)        # 2x3 array filled with 7

# Empty arrays (uninitialized memory)
np.empty(5)               # Random values
np.empty_like(arr)        # Same shape as arr

# Identity and eye matrices
np.eye(3)                 # 3x3 identity matrix
np.identity(4)            # 4x4 identity matrix

# Ranges and sequences
np.arange(10)             # [0 1 2 3 4 5 6 7 8 9]
np.arange(1, 10, 2)       # [1 3 5 7 9]
np.arange(0, 1, 0.1)      # [0. 0.1 0.2 ... 0.9]
np.linspace(0, 10, 5)     # [0. 2.5 5. 7.5 10.]
np.logspace(0, 2, 3)      # [1. 10. 100.]

# Random arrays
np.random.rand(2, 3)      # Uniform [0,1)
np.random.randn(2, 3)     # Standard normal
np.random.randint(0, 10, (2, 3))  # Random integers
np.random.choice([1, 2, 3], size=(2, 3), replace=True)

# From functions
def f(x, y):
    return x + y
np.fromfunction(f, (3, 3))

# Meshgrids for function evaluation
x = np.linspace(-2, 2, 5)
y = np.linspace(-2, 2, 5)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2
```

### Data Types
```python
# Common dtypes
np.int8, np.int16, np.int32, np.int64
np.uint8, np.uint16, np.uint32, np.uint64
np.float16, np.float32, np.float64, np.float128
np.complex64, np.complex128
np.bool_, np.object_

# Create with specific dtype
arr = np.array([1, 2, 3], dtype=np.int32)
arr = np.zeros(5, dtype=np.float32)

# Type conversion
arr = arr.astype(np.float64)
arr = arr.astype('f8')      # Shorthand for float64

# Check and compare dtypes
print(arr.dtype.name)       # 'float64'
print(arr.dtype.itemsize)  # 8 (bytes per element)
print(np.issubdtype(arr.dtype, np.floating))  # True
```

---

## Array Attributes & Info

### Shape and Size
```python
arr = np.random.rand(2, 3, 4)

# Dimensions
print(arr.shape)          # (2, 3, 4)
print(arr.ndim)           # 3
print(arr.size)           # 24 (total elements)

# Memory info
print(arr.nbytes)         # Memory in bytes
print(arr.itemsize)      # Bytes per element
print(arr.strides)       # Strides for each dimension

# Flags and properties
print(arr.flags)          # Memory layout info
print(arr.base)           # Base array (if view)
print(arr.data)           # Memory buffer
```

### Array Information
```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

# Quick info
np.info(arr)              # Detailed info
print(repr(arr))          # Full representation
print(str(arr))           # String representation

# Data type details
print(arr.dtype)          # Data type
print(arr.dtype.kind)     # 'i' for integer, 'f' for float, etc.
print(arr.dtype.char)     # Type character
print(arr.dtype.num)      # Type number

# Memory layout
print(arr.flags.c_contiguous)   # C-style contiguous?
print(arr.flags.f_contiguous)   # Fortran-style contiguous?
print(arr.flags.owndata)        # Owns the data?
print(arr.flags.writeable)      # Can modify?
```

---

## Indexing & Slicing

### Basic Indexing
```python
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# Single elements
arr[0]                    # First element: 0
arr[-1]                   # Last element: 9
arr[5]                    # Element at index 5: 5

# Slicing [start:stop:step]
arr[1:4]                  # [1 2 3]
arr[::2]                  # [0 2 4 6 8] (every 2nd)
arr[::-1]                 # [9 8 7 6 5 4 3 2 1 0] (reverse)
arr[1:-1]                 # [1 2 3 4 5 6 7 8] (exclude first/last)

# 2D indexing
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arr2d[0, 1]               # Element at row 0, col 1: 2
arr2d[1, :]               # Entire row 1: [4 5 6]
arr2d[:, 2]               # Entire column 2: [3 6 9]
arr2d[0:2, 1:3]           # Subarray: [[2 3] [5 6]]

# 3D indexing
arr3d = np.random.rand(2, 3, 4)
arr3d[0, :, :]            # First "page"
arr3d[:, 1, :]            # Second "row" of all pages
arr3d[..., -1]            # Last "column" (ellipsis notation)
```

### Assignment
```python
arr = np.arange(10)

# Single element
arr[0] = 42               # [42  1  2  3  4  5  6  7  8  9]

# Slice assignment
arr[1:4] = -1             # [42 -1 -1 -1  4  5  6  7  8  9]
arr[::2] = 0              # [ 0 -1  0 -1  0  5  0  7  0  9]

# Broadcasting in assignment
arr2d = np.zeros((3, 3))
arr2d[:, 1] = [1, 2, 3]   # Set middle column
arr2d[1, :] = 5           # Set middle row to 5
```

---

## Array Manipulation

### Reshaping
```python
arr = np.arange(12)

# Reshape (returns view when possible)
arr.reshape(3, 4)         # 3x4 matrix
arr.reshape((2, 6))       # 2x6 matrix
arr.reshape(2, -1)        # 2xN (infer last dimension)

# Resize (modifies original array)
arr.resize(3, 4)

# Flatten and ravel
arr2d = np.array([[1, 2], [3, 4]])
arr2d.flatten()           # [1 2 3 4] (always copy)
arr2d.ravel()             # [1 2 3 4] (view when possible)

# Transpose
arr2d.T                   # Transpose
arr2d.transpose()         # Transpose
np.transpose(arr2d)       # Transpose

# Add/remove dimensions
arr.reshape(-1, 1)        # Column vector
arr[np.newaxis, :]        # Row vector
np.expand_dims(arr, axis=0)  # Add dimension at axis 0
np.squeeze(arr)           # Remove single-dimensional entries
```

### Joining Arrays
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Concatenation
np.concatenate([a, b])              # [1 2 3 4 5 6]
np.concatenate([a, b], axis=0)      # Same as above

# Stack arrays
np.stack([a, b])                    # [[1 2 3] [4 5 6]]
np.stack([a, b], axis=1)            # [[1 4] [2 5] [3 6]]

# Horizontal and vertical stack
np.hstack([a, b])                   # [1 2 3 4 5 6]
np.vstack([a, b])                   # [[1 2 3] [4 5 6]]

# 2D examples
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])

np.concatenate([arr1, arr2], axis=0)  # Vertical concat
np.concatenate([arr1, arr2], axis=1)  # Horizontal concat
np.hstack([arr1, arr2])             # [[1 2 5 6] [3 4 7 8]]
np.vstack([arr1, arr2])             # [[1 2] [3 4] [5 6] [7 8]]

# Column and row stack
np.column_stack([a, b])             # [[1 4] [2 5] [3 6]]
np.row_stack([a, b])                # [[1 2 3] [4 5 6]]
```

### Splitting Arrays
```python
arr = np.arange(12).reshape(3, 4)

# Split into equal parts
np.split(arr, 3, axis=0)            # Split into 3 rows
np.split(arr, 2, axis=1)            # Split into 2 columns

# Split at specific indices
np.split(arr, [1, 3], axis=0)       # Split at row 1 and 3

# Horizontal and vertical split
np.hsplit(arr, 2)                   # Split columns
np.vsplit(arr, 3)                   # Split rows

# Array split (1D)
arr1d = np.arange(9)
np.array_split(arr1d, 4)            # Split into 4 parts (unequal OK)
```

### Copying
```python
arr = np.arange(5)

# View (shares data)
view = arr.view()
view[0] = 99              # Modifies original

# Copy (independent)
copy = arr.copy()
copy[0] = 99              # Doesn't modify original

# Deep copy for object arrays
arr_obj = np.array([{'a': 1}, {'b': 2}])
copy_obj = arr_obj.copy()  # Shallow copy of objects
```

---

## Mathematical Operations

### Element-wise Operations
```python
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

# Arithmetic operators
a + b                     # [11 22 33 44]
a - b                     # [-9 -18 -27 -36]
a * b                     # [10 40 90 160]
a / b                     # [0.1 0.1 0.1 0.1]
a // b                    # Floor division
a % b                     # Modulo
a ** 2                    # [1 4 9 16] (power)

# Comparison operators
a > 2                     # [False False True True]
a == b                    # [False False False False]
a != b                    # [True True True True]

# Logical operators
(a > 1) & (a < 4)         # [False True True False]
(a < 2) | (a > 3)         # [True False False True]
~(a > 2)                  # [True True False False] (not)
```

### Universal Functions (ufuncs)
```python
arr = np.array([1, 4, 9, 16])

# Mathematical functions
np.sqrt(arr)              # [1. 2. 3. 4.]
np.exp(arr)               # Exponential
np.log(arr)               # Natural logarithm
np.log10(arr)             # Base-10 logarithm
np.log2(arr)              # Base-2 logarithm

# Trigonometric functions
angles = np.array([0, np.pi/2, np.pi])
np.sin(angles)            # [0. 1. 0.]
np.cos(angles)            # [1. 0. -1.]
np.tan(angles)            # [0. inf 0.]

# Inverse trig functions
np.arcsin([0, 1, 0])      # [0. π/2 0.]
np.arccos([1, 0, -1])     # [0. π/2 π]
np.arctan([0, 1, np.inf]) # [0. π/4 π/2]

# Hyperbolic functions
np.sinh(arr)              # Hyperbolic sine
np.cosh(arr)              # Hyperbolic cosine
np.tanh(arr)              # Hyperbolic tangent

# Rounding
arr_float = np.array([1.1, 1.5, 1.9, 2.5])
np.round(arr_float)       # [1. 2. 2. 2.]
np.floor(arr_float)       # [1. 1. 1. 2.]
np.ceil(arr_float)        # [2. 2. 2. 3.]
np.trunc(arr_float)       # [1. 1. 1. 2.]

# Sign and absolute value
arr_mixed = np.array([-2, -1, 0, 1, 2])
np.abs(arr_mixed)         # [2 1 0 1 2]
np.sign(arr_mixed)        # [-1 -1 0 1 1]

# Min/max element-wise
a = np.array([1, 5, 3])
b = np.array([2, 2, 6])
np.maximum(a, b)          # [2 5 6]
np.minimum(a, b)          # [1 2 3]
```

### Aggregate Functions
```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

# Basic aggregations
np.sum(arr)               # 21 (sum of all elements)
np.sum(arr, axis=0)       # [5 7 9] (sum along rows)
np.sum(arr, axis=1)       # [6 15] (sum along columns)

np.mean(arr)              # 3.5 (mean of all)
np.mean(arr, axis=0)      # [2.5 3.5 4.5]

np.std(arr)               # Standard deviation
np.var(arr)               # Variance

np.min(arr)               # 1
np.max(arr)               # 6
np.argmin(arr)            # 0 (index of min in flattened)
np.argmax(arr)            # 5 (index of max in flattened)

# Percentiles and quantiles
np.percentile(arr, 50)    # Median (50th percentile)
np.quantile(arr, 0.25)    # 25th percentile
np.median(arr)            # Median

# Cumulative functions
arr1d = np.array([1, 2, 3, 4])
np.cumsum(arr1d)          # [1 3 6 10] (cumulative sum)
np.cumprod(arr1d)         # [1 2 6 24] (cumulative product)

# Other useful aggregates
np.all(arr > 0)           # True (all elements > 0)
np.any(arr > 5)           # True (any element > 5)
np.count_nonzero(arr)     # 6 (count non-zero elements)
```

---

## Statistical Functions

### Descriptive Statistics
```python
data = np.random.randn(1000)  # 1000 random normal values

# Central tendency
np.mean(data)             # Sample mean
np.median(data)           # Median
from scipy import stats
stats.mode(data)          # Mode (requires scipy)

# Variability
np.std(data)              # Standard deviation (sample)
np.std(data, ddof=1)      # Standard deviation (population)
np.var(data)              # Variance
np.ptp(data)              # Peak-to-peak (max - min)

# Distribution shape
from scipy.stats import skew, kurtosis
skew(data)                # Skewness
kurtosis(data)            # Kurtosis

# Quantiles
np.percentile(data, [25, 50, 75])  # Quartiles
np.quantile(data, [0.1, 0.9])      # 10th and 90th percentiles

# Correlation and covariance (2D)
data2d = np.random.randn(100, 3)
np.corrcoef(data2d, rowvar=False)  # Correlation matrix
np.cov(data2d, rowvar=False)       # Covariance matrix
```

### Histograms and Binning
```python
data = np.random.randn(1000)

# Histogram
counts, bins = np.histogram(data, bins=20)
counts, bins = np.histogram(data, bins=np.linspace(-3, 3, 21))

# 2D histogram
x = np.random.randn(1000)
y = np.random.randn(1000)
counts, xbins, ybins = np.histogram2d(x, y, bins=20)

# Digital histogram
indices = np.digitize(data, bins=np.linspace(-3, 3, 7))

# Binned statistics
from scipy.stats import binned_statistic
means, bin_edges, bin_numbers = binned_statistic(
    x, y, statistic='mean', bins=10
)
```

---

## Linear Algebra

### Basic Operations
```python
# Matrix creation
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
v = np.array([1, 2])

# Matrix multiplication
A @ B                     # Matrix multiplication (Python 3.5+)
np.dot(A, B)              # Matrix multiplication
A.dot(B)                  # Matrix multiplication

# Vector operations
np.dot(v, v)              # Dot product (scalar)
np.outer(v, v)            # Outer product (matrix)
np.cross([1, 2, 3], [4, 5, 6])  # Cross product

# Matrix properties
A.T                       # Transpose
np.trace(A)               # Trace (sum of diagonal)
np.linalg.det(A)          # Determinant
np.linalg.matrix_rank(A)  # Matrix rank
```

### Linear Algebra Operations
```python
# Solving linear systems (Ax = b)
A = np.array([[2, 1], [1, 3]])
b = np.array([1, 2])
x = np.linalg.solve(A, b)  # Solve Ax = b

# Matrix inversion
A_inv = np.linalg.inv(A)   # Inverse matrix
A_pinv = np.linalg.pinv(A) # Pseudo-inverse

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
eigenvals = np.linalg.eigvals(A)  # Eigenvalues only

# Singular Value Decomposition
U, s, Vt = np.linalg.svd(A)

# QR decomposition
Q, R = np.linalg.qr(A)

# Cholesky decomposition (for positive definite matrices)
L = np.linalg.cholesky(A.T @ A)  # A.T @ A is positive definite

# Norms
np.linalg.norm(v)         # Euclidean norm (L2)
np.linalg.norm(v, ord=1)  # L1 norm (Manhattan)
np.linalg.norm(v, ord=np.inf)  # Infinity norm
np.linalg.norm(A, 'fro')  # Frobenius norm for matrices

# Matrix condition number
np.linalg.cond(A)         # Condition number
```

### Advanced Linear Algebra
```python
# Least squares solution
# For overdetermined system Ax ≈ b
A = np.random.rand(10, 3)  # More equations than unknowns
b = np.random.rand(10)
x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

# Matrix exponential
from scipy.linalg import expm, logm, sqrtm
expm(A)                   # Matrix exponential
logm(A)                   # Matrix logarithm
sqrtm(A)                  # Matrix square root

# Schur decomposition
T, Z = np.linalg.schur(A)

# Generalized eigenvalue problem
A = np.random.rand(3, 3)
B = np.random.rand(3, 3)
eigenvals, eigenvecs = np.linalg.eig(A, B)
```

---

## Broadcasting

### Broadcasting Rules
```python
# Arrays with the same number of dimensions:
# - Dimensions are compatible if they're equal or one is 1
# - Missing dimensions are assumed to be 1

# Examples
a = np.array([1, 2, 3])        # shape (3,)
b = np.array([[1], [2], [3]])  # shape (3, 1)
result = a + b                 # Broadcasting: (3,) + (3,1) -> (3,3)

# More complex example
a = np.random.rand(3, 4, 5)    # shape (3, 4, 5)
b = np.random.rand(4, 1)       # shape (4, 1)
result = a + b                 # Broadcasting: (3,4,5) + (4,1) -> (3,4,5)

# Manual broadcasting
a = np.array([1, 2, 3])
b = np.array([[1], [2]])
# Understand shapes:
print(f"a.shape: {a.shape}")   # (3,)
print(f"b.shape: {b.shape}")   # (2, 1)
result = a + b                 # Result shape: (2, 3)
```

### Broadcasting Functions
```python
# Check if arrays broadcast
a = np.array([1, 2, 3])
b = np.array([[1], [2]])
try:
    result = np.broadcast_arrays(a, b)
    print("Broadcasting successful")
except ValueError:
    print("Broadcasting not possible")

# Manual shape manipulation for broadcasting
a = np.array([1, 2, 3])
a_broadcast = a[np.newaxis, :]  # Shape (1, 3)
a_broadcast = a[:, np.newaxis]  # Shape (3, 1)

# Einstein summation (advanced broadcasting)
A = np.random.rand(3, 4)
B = np.random.rand(4, 5)
C = np.einsum('ij,jk->ik', A, B)  # Matrix multiplication

# Common broadcasting patterns
# Normalize rows
matrix = np.random.rand(3, 4)
row_means = matrix.mean(axis=1, keepdims=True)
normalized = matrix - row_means

# Normalize columns
col_means = matrix.mean(axis=0)
normalized = matrix - col_means

# Distance matrix
points = np.random.rand(10, 2)  # 10 points in 2D
distances = np.sqrt(((points[:, np.newaxis] - points[np.newaxis, :]) ** 2).sum(axis=2))
```

---

## Advanced Indexing

### Boolean Indexing
```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

# Boolean conditions
mask = arr > 5            # [False False False False False True True True True]
arr[mask]                 # [6 7 8 9]
arr[arr > 5]              # Same as above

# Multiple conditions
arr[(arr > 3) & (arr < 7)]  # [4 5 6]
arr[(arr < 3) | (arr > 7)]  # [1 2 8 9]

# 2D boolean indexing
arr2d = np.random.randint(0, 10, (3, 4))
arr2d[arr2d > 5] = 0      # Set elements > 5 to 0

# Where function
np.where(arr > 5, arr, 0)   # arr if arr > 5, else 0
np.where(arr > 5)           # Indices where condition is True

# Non-zero indices
indices = np.nonzero(arr > 5)
arr[indices]
```

### Fancy Indexing
```python
arr = np.array([10, 20, 30, 40, 50])

# Index with integer arrays
indices = np.array([0, 2, 4])
arr[indices]              # [10 30 50]

# Negative indices
arr[[-1, -2, -3]]         # [50 40 30]

# 2D fancy indexing
arr2d = np.arange(12).reshape(3, 4)
rows = np.array([0, 2])
cols = np.array([1, 3])
arr2d[rows, cols]         # Elements at (0,1) and (2,3)

# Advanced indexing combinations
arr2d[np.ix_(rows, cols)]  # 2x2 subarray at intersection

# Take function
np.take(arr, indices)     # Same as arr[indices]
np.take(arr2d, indices, axis=0)  # Take rows
np.take(arr2d, indices, axis=1)  # Take columns

# Put function
arr_copy = arr.copy()
np.put(arr_copy, [0, 2], 99)  # Set arr[0] and arr[2] to 99
```

### Structured Arrays
```python
# Define structured array dtype
dt = np.dtype([
    ('name', 'U10'),      # Unicode string, max 10 chars
    ('age', 'i4'),        # 32-bit integer
    ('weight', 'f4')      # 32-bit float
])

# Create structured array
people = np.array([
    ('Alice', 25, 55.5),
    ('Bob', 30, 70.2),
    ('Charlie', 35, 80.1)
], dtype=dt)

# Access fields
people['name']            # Array of names
people['age']             # Array of ages
people[0]['name']         # 'Alice'
people[people['age'] > 30]['name']  # Names of people > 30

# Record arrays (similar but with attribute access)
people_rec = np.rec.array([
    ('Alice', 25, 55.5),
    ('Bob', 30, 70.2)
], dtype=[('name', 'U10'), ('age', 'i4'), ('weight', 'f4')])

people_rec.name           # Access by attribute
```

---

## Performance Optimization

### Memory Layout
```python
# C-order (row-major) vs F-order (column-major)
arr_c = np.array([[1, 2, 3], [4, 5, 6]], order='C')  # Default
arr_f = np.array([[1, 2, 3], [4, 5, 6]], order='F')

print(arr_c.flags.c_contiguous)  # True for C-order
print(arr_f.flags.f_contiguous)  # True for F-order

# Convert between orders
arr_f_copy = np.ascontiguousarray(arr_f)  # Convert to C-order
arr_c_copy = np.asfortranarray(arr_c)     # Convert to F-order

# Memory views vs copies
view = arr[::2]           # Creates view (shares memory)
copy = arr[::2].copy()    # Creates copy (new memory)

# Check if array is a view
print(view.base is arr)   # True if view
print(copy.base is None)  # True if copy
```

### Vectorization Techniques
```python
# Instead of Python loops, use NumPy operations
# BAD: Python loop
def slow_function(arr):
    result = np.zeros_like(arr)
    for i in range(len(arr)):
        result[i] = arr[i] ** 2 + 2 * arr[i] + 1
    return result

# GOOD: Vectorized
def fast_function(arr):
    return arr**2 + 2*arr + 1

# Timing comparison
arr = np.random.rand(1000000)
%timeit slow_function(arr)    # Slow
%timeit fast_function(arr)    # ~100x faster

# Use numpy functions instead of math functions
# BAD
import math
result = [math.sin(x) for x in arr]

# GOOD
result = np.sin(arr)

# Vectorize custom functions
def custom_func(x, y):
    if x > y:
        return x * 2
    else:
        return y * 2

vectorized_func = np.vectorize(custom_func)
result = vectorized_func(arr1, arr2)

# Numba for JIT compilation (if available)
try:
    from numba import jit
    
    @jit
    def fast_custom_func(arr):
        result = np.zeros_like(arr)
        for i in range(len(arr)):
            result[i] = arr[i] ** 2
        return result
except ImportError:
    pass
```

### Memory Management
```python
# Monitor memory usage
import psutil
import gc

def get_memory_usage():
    return psutil.Process().memory_info().rss / 1024 / 1024  # MB

# Memory-efficient operations
# Use views instead of copies when possible
large_arr = np.random.rand(1000, 1000)
sub_arr = large_arr[::10, ::10]  # View, not copy

# Delete large arrays when done
del large_arr
gc.collect()

# Use appropriate data types
# BAD: Using float64 when float32 is sufficient
arr_64 = np.random.rand(1000000).astype(np.float64)  # 8 bytes/element

# GOOD: Using float32 when precision allows
arr_32 = np.random.rand(1000000).astype(np.float32)  # 4 bytes/element

# Memory mapping for large files
# For very large arrays that don't fit in memory
memmap = np.memmap('large_file.dat', dtype='float32', mode='r', shape=(10000, 10000))

# Process in chunks
chunk_size = 1000
for i in range(0, memmap.shape[0], chunk_size):
    chunk = memmap[i:i+chunk_size]
    # Process chunk
    result = np.mean(chunk, axis=1)
```

### Optimization Tips
```python
# 1. Use in-place operations when possible
arr = np.random.rand(1000000)
arr += 1              # In-place addition (no new memory)
arr = arr + 1         # Creates new array

# 2. Use axis parameter for reductions
arr2d = np.random.rand(1000, 1000)
np.sum(arr2d, axis=0)  # Sum along first axis (faster)

# 3. Avoid repeated array creation
# BAD
result = np.array([])
for i in range(1000):
    result = np.append(result, i)  # Inefficient

# GOOD
result = np.zeros(1000)
for i in range(1000):
    result[i] = i

# BEST
result = np.arange(1000)

# 4. Use broadcasting to avoid loops
# BAD
result = np.zeros((1000, 1000))
for i in range(1000):
    result[i, :] = np.arange(1000) + i

# GOOD
result = np.arange(1000) + np.arange(1000)[:, np.newaxis]

# 5. Cache frequently used calculations
expensive_calc = np.linalg.inv(large_matrix)
# Use expensive_calc multiple times instead of recalculating
```

---

## Common Patterns

### Data Processing Patterns
```python
# 1. Normalization
def normalize(data, method='zscore'):
    if method == 'zscore':
        return (data - np.mean(data)) / np.std(data)
    elif method == 'minmax':
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    elif method == 'robust':
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        return (data - median) / mad

# 2. Moving averages
def moving_average(data, window):
    return np.convolve(data, np.ones(window)/window, mode='valid')

# 3. Outlier detection
def detect_outliers(data, method='iqr', threshold=1.5):
    if method == 'iqr':
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        return (data < lower) | (data > upper)
    elif method == 'zscore':
        z_scores = np.abs((data - np.mean(data)) / np.std(data))
        return z_scores > threshold

# 4. Binning/Discretization
def bin_data(data, n_bins, method='equal_width'):
    if method == 'equal_width':
        bins = np.linspace(np.min(data), np.max(data), n_bins + 1)
    elif method == 'equal_frequency':
        bins = np.percentile(data, np.linspace(0, 100, n_bins + 1))
    return np.digitize(data, bins)

# 5. Interpolation
from scipy import interpolate

def interpolate_missing(data, method='linear'):
    valid_mask = ~np.isnan(data)
    valid_indices = np.where(valid_mask)[0]
    if method == 'linear':
        f = interpolate.interp1d(valid_indices, data[valid_mask], 
                               kind='linear', fill_value='extrapolate')
    return f(np.arange(len(data)))
```

### Numerical Computing Patterns
```python
# 1. Finite difference derivatives
def gradient_1d(y, x=None, edge_order=1):
    if x is None:
        x = np.arange(len(y))
    return np.gradient(y, x, edge_order=edge_order)

# 2. Numerical integration (trapezoidal rule)
def integrate_trapz(y, x=None):
    if x is None:
        x = np.arange(len(y))
    return np.trapz(y, x)

# 3. Root finding (using numpy + scipy)
def find_roots(func, x_range, n_points=1000):
    x = np.linspace(*x_range, n_points)
    y = func(x)
    # Find sign changes
    sign_changes = np.diff(np.sign(y)) != 0
    return x[:-1][sign_changes]

# 4. Peak detection
def find_peaks_simple(data, height=None, distance=None):
    peaks = []
    for i in range(1, len(data) - 1):
        if data[i] > data[i-1] and data[i] > data[i+1]:
            if height is None or data[i] >= height:
                if distance is None or all(abs(i - p) >= distance for p in peaks):
                    peaks.append(i)
    return np.array(peaks)

# 5. Fourier analysis
def analyze_frequency(signal, sample_rate):
    n = len(signal)
    frequencies = np.fft.fftfreq(n, 1/sample_rate)
    fft_values = np.fft.fft(signal)
    magnitude = np.abs(fft_values)
    return frequencies[:n//2], magnitude[:n//2]  # Positive frequencies only
```

### Performance Patterns
```python
# 1. Batch processing
def process_in_batches(data, batch_size, process_func):
    results = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        result = process_func(batch)
        results.append(result)
    return np.concatenate(results)

# 2. Memory-efficient iteration
def process_large_array(arr, chunk_size=10000):
    n_chunks = (len(arr) + chunk_size - 1) // chunk_size
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(arr))
        chunk = arr[start_idx:end_idx]
        yield chunk

# 3. Parallel processing pattern (conceptual)
# Note: Actual parallel processing requires multiprocessing or joblib
def parallel_operation(data, n_cores=4):
    chunk_size = len(data) // n_cores
    chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
    # Process chunks in parallel (pseudo-code)
    # results = [process_chunk(chunk) for chunk in chunks]
    # return np.concatenate(results)
    pass

# 4. Caching expensive computations
cache = {}
def expensive_function(x):
    key = tuple(x) if hasattr(x, '__iter__') else x
    if key not in cache:
        cache[key] = complex_calculation(x)
    return cache[key]

# 5. Lazy evaluation pattern
class LazyArray:
    def __init__(self, shape, func):
        self.shape = shape
        self.func = func
        self._computed = False
        self._data = None
    
    def compute(self):
        if not self._computed:
            self._data = self.func()
            self._computed = True
        return self._data
    
    def __getitem__(self, key):
        return self.compute()[key]
```

### Common Algorithms
```python
# 1. K-means clustering (simplified)
def kmeans_simple(data, k, max_iters=100):
    # Initialize centroids randomly
    centroids = data[np.random.choice(len(data), k, replace=False)]
    
    for _ in range(max_iters):
        # Assign points to closest centroid
        distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        
        # Update centroids
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
        
        # Check convergence
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    
    return labels, centroids

# 2. Principal Component Analysis (simplified)
def pca_simple(data, n_components):
    # Center the data
    centered_data = data - np.mean(data, axis=0)
    
    # Compute covariance matrix
    cov_matrix = np.cov(centered_data.T)
    
    # Compute eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
    
    # Sort by eigenvalues (descending)
    idx = np.argsort(eigenvals)[::-1]
    eigenvecs = eigenvecs[:, idx]
    
    # Select top n_components
    components = eigenvecs[:, :n_components]
    
    # Transform data
    transformed = centered_data @ components
    
    return transformed, components

# 3. Linear regression
def linear_regression(X, y):
    # Add bias term
    X_with_bias = np.column_stack([np.ones(len(X)), X])
    
    # Normal equation: (X^T X)^(-1) X^T y
    coefficients = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
    
    return coefficients  # [bias, slope1, slope2, ...]

# 4. Convolution (1D)
def convolve1d(signal, kernel, mode='valid'):
    return np.convolve(signal, kernel, mode)

# 5. Distance matrix
def distance_matrix(points, metric='euclidean'):
    if metric == 'euclidean':
        diff = points[:, np.newaxis] - points[np.newaxis, :]
        return np.sqrt(np.sum(diff**2, axis=2))
    elif metric == 'manhattan':
        diff = points[:, np.newaxis] - points[np.newaxis, :]
        return np.sum(np.abs(diff), axis=2)
```

---

## Pro Tips & Best Practices

### Code Style & Conventions
```python
# 1. Use descriptive variable names
good_array = np.array([1, 2, 3])      # Good
a = np.array([1, 2, 3])               # Avoid

# 2. Specify axis parameter explicitly
data.mean(axis=0)                     # Good: explicit
data.mean()                           # OK: when you want overall mean

# 3. Use broadcasting instead of loops
# Good
result = array_2d + vector

# Avoid
result = np.zeros_like(array_2d)
for i in range(array_2d.shape[0]):
    result[i] = array_2d[i] + vector

# 4. Check array properties
assert arr.ndim == 2, f"Expected 2D array, got {arr.ndim}D"
assert arr.dtype == np.float64, f"Expected float64, got {arr.dtype}"

# 5. Use context managers for files
with open('data.npy', 'rb') as f:
    arr = np.load(f)
```

### Debugging Tips
```python
# 1. Inspect array properties
def debug_array(arr, name="array"):
    print(f"{name}:")
    print(f"  Shape: {arr.shape}")
    print(f"  Dtype: {arr.dtype}")
    print(f"  Min: {np.min(arr):.3f}, Max: {np.max(arr):.3f}")
    print(f"  Mean: {np.mean(arr):.3f}, Std: {np.std(arr):.3f}")
    print(f"  Has NaN: {np.isnan(arr).any()}")
    print(f"  Has Inf: {np.isinf(arr).any()}")
    print()

# 2. Check for common issues
def check_array_health(arr):
    issues = []
    if np.isnan(arr).any():
        issues.append(f"Contains {np.isnan(arr).sum()} NaN values")
    if np.isinf(arr).any():
        issues.append(f"Contains {np.isinf(arr).sum()} infinite values")
    if arr.size == 0:
        issues.append("Array is empty")
    return issues

# 3. Visualize array structure (for small arrays)
def print_array_structure(arr):
    if arr.size < 100:  # Only for small arrays
        print(f"Shape: {arr.shape}")
        print(repr(arr))
    else:
        print(f"Large array - Shape: {arr.shape}, Dtype: {arr.dtype}")
        print(f"Sample values: {arr.flat[:5]}...")

# 4. Compare arrays safely
def arrays_equal(arr1, arr2, rtol=1e-5, atol=1e-8):
    if arr1.shape != arr2.shape:
        return False
    return np.allclose(arr1, arr2, rtol=rtol, atol=atol)

# 5. Memory usage tracking
def array_memory_usage(arr):
    return f"{arr.nbytes / 1024 / 1024:.2f} MB"
```

---

**📚 Additional Resources:**
- [NumPy Documentation](https://numpy.org/doc/stable/)
- [NumPy User Guide](https://numpy.org/doc/stable/user/index.html)
- [NumPy Array Programming with Python](https://numpy.org/doc/stable/user/absolute_beginners.html)
- [SciPy Ecosystem](https://scipy.org/)

**🔗 Related Cheatsheets:**
- [Python Cheatsheet](PYTHON_CHEATSHEET.md)
- [Pandas Cheatsheet](PANDAS_CHEATSHEET.md)
- [PySpark Cheatsheet](PYSPARK_CHEATSHEET.md)
- [PyTorch Cheatsheet](PYTORCH_CHEATSHEET.md)

---
*Last Updated: January 8, 2026*
