# Python 3.12 Comprehensive Cheatsheet

**Level:** SparkNotes/Unbound Quality Reference  
**Date:** January 8, 2026  
**Python Version:** 3.12+

## Table of Contents

- [Core Syntax & Data Types](#core-syntax--data-types)
- [Control Flow](#control-flow)
- [Functions & Classes](#functions--classes)
- [Error Handling](#error-handling)
- [File I/O & Context Managers](#file-io--context-managers)
- [Comprehensions & Generators](#comprehensions--generators)
- [Decorators & Metaclasses](#decorators--metaclasses)
- [Built-in Functions](#built-in-functions)
- [Standard Library Essentials](#standard-library-essentials)
- [Type Hints & Annotations](#type-hints--annotations)
- [Performance & Optimization](#performance--optimization)
- [Testing & Debugging](#testing--debugging)
- [Modern Python Features (3.10+)](#modern-python-features-310)

---

## Core Syntax & Data Types

### Variables & Basic Types
```python
# Variable assignment (dynamic typing)
x = 42                    # int
y = 3.14                  # float
name = "Alice"            # str
is_active = True          # bool
data = None               # NoneType

# Multiple assignment
a, b, c = 1, 2, 3
x = y = z = 0

# Augmented assignment
x += 1    # x = x + 1
x *= 2    # x = x * 2
```

### Strings
```python
# String creation
s1 = 'Single quotes'
s2 = "Double quotes"
s3 = """Triple quotes
for multiline"""
s4 = r"Raw string \n"     # No escape processing

# F-strings (formatted string literals)
name, age = "Bob", 25
msg = f"Hello {name}, you are {age} years old"
msg = f"Calculation: {2 + 3 = }"  # Python 3.8+ debug syntax

# String methods (most common)
text = "  Hello World  "
text.strip()              # Remove whitespace
text.lower()              # Convert to lowercase
text.upper()              # Convert to uppercase
text.replace("World", "Python")
text.split()              # Split on whitespace
"_".join(["a", "b", "c"]) # Join with separator
text.startswith("Hello")
text.endswith("World")
text.find("World")        # Return index or -1
text.count("l")           # Count occurrences

# String formatting
"Hello {}".format("World")
"Hello {name}".format(name="Alice")
"Number: {:.2f}".format(3.14159)
```

### Collections

#### Lists (Ordered, Mutable)
```python
# Creation
lst = [1, 2, 3, 4, 5]
lst = list(range(10))     # [0, 1, 2, ..., 9]
empty = []

# Access & Modification
lst[0]                    # First element
lst[-1]                   # Last element
lst[1:4]                  # Slice [1, 2, 3]
lst[::2]                  # Every 2nd element
lst[::-1]                 # Reverse

# Methods
lst.append(6)             # Add to end
lst.insert(0, 0)          # Insert at index
lst.extend([7, 8])        # Add multiple elements
lst.remove(3)             # Remove first occurrence
lst.pop()                 # Remove and return last
lst.pop(0)                # Remove and return at index
lst.index(5)              # Find index of value
lst.count(2)              # Count occurrences
lst.sort()                # Sort in-place
lst.reverse()             # Reverse in-place

# List operations
len(lst)                  # Length
max(lst)                  # Maximum value
min(lst)                  # Minimum value
sum(lst)                  # Sum of elements
sorted(lst)               # Return sorted copy
```

#### Tuples (Ordered, Immutable)
```python
# Creation
tup = (1, 2, 3)
tup = 1, 2, 3             # Parentheses optional
single = (42,)            # Single element tuple
empty = ()

# Named tuples
from collections import namedtuple
Point = namedtuple('Point', ['x', 'y'])
p = Point(1, 2)
print(p.x, p.y)          # Access by name
```

#### Dictionaries (Key-Value, Ordered since 3.7)
```python
# Creation
dct = {"name": "Alice", "age": 30}
dct = dict(name="Alice", age=30)
empty = {}

# Access & Modification
dct["name"]               # Get value
dct.get("name", "Unknown") # Get with default
dct["city"] = "NYC"       # Add/update
del dct["age"]            # Delete key

# Methods
dct.keys()                # dict_keys object
dct.values()              # dict_values object
dct.items()               # dict_items object
dct.update({"age": 31})   # Update multiple
dct.pop("name", "Unknown") # Remove and return
dct.setdefault("country", "USA") # Set if not exists

# Dictionary comprehension
squares = {x: x**2 for x in range(5)}
```

#### Sets (Unordered, Unique Elements)
```python
# Creation
s = {1, 2, 3, 4, 5}
s = set([1, 2, 2, 3])     # From list, duplicates removed
empty = set()             # Empty set (not {})

# Operations
s.add(6)                  # Add element
s.update([7, 8])          # Add multiple
s.remove(3)               # Remove (error if not found)
s.discard(3)              # Remove (no error)
s.pop()                   # Remove arbitrary element

# Set operations
s1 = {1, 2, 3}
s2 = {3, 4, 5}
s1 | s2                   # Union {1, 2, 3, 4, 5}
s1 & s2                   # Intersection {3}
s1 - s2                   # Difference {1, 2}
s1 ^ s2                   # Symmetric difference {1, 2, 4, 5}
```

---

## Control Flow

### Conditionals
```python
# Basic if statement
if condition:
    pass
elif other_condition:
    pass
else:
    pass

# Ternary operator
result = value_if_true if condition else value_if_false

# Chained comparisons
if 0 < x < 10:
    print("x is between 0 and 10")

# Boolean operators
if condition1 and condition2:
    pass
if condition1 or condition2:
    pass
if not condition:
    pass

# Truthiness
# Falsy: None, False, 0, '', [], {}, set()
# Everything else is truthy
if lst:  # Check if list is not empty
    pass
```

### Loops

#### For Loops
```python
# Iterate over sequence
for item in [1, 2, 3]:
    print(item)

# Iterate with index
for i, item in enumerate(['a', 'b', 'c']):
    print(f"{i}: {item}")

# Iterate over dictionary
for key in dct:
    print(key, dct[key])

for key, value in dct.items():
    print(key, value)

# Range
for i in range(10):       # 0 to 9
    pass
for i in range(1, 11):    # 1 to 10
    pass
for i in range(0, 10, 2): # 0, 2, 4, 6, 8
    pass

# Zip for parallel iteration
names = ['Alice', 'Bob']
ages = [25, 30]
for name, age in zip(names, ages):
    print(f"{name} is {age} years old")
```

#### While Loops
```python
i = 0
while i < 10:
    print(i)
    i += 1

# While with else (executes if no break)
while condition:
    if exit_condition:
        break
else:
    print("Loop completed normally")
```

#### Loop Control
```python
for i in range(10):
    if i == 5:
        continue    # Skip to next iteration
    if i == 8:
        break       # Exit loop
    print(i)
```

---

## Functions & Classes

### Functions
```python
# Basic function
def greet(name, greeting="Hello"):
    """Function with default parameter."""
    return f"{greeting}, {name}!"

# Variable arguments
def sum_all(*args):
    return sum(args)

def print_info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

# Combined
def complex_func(required, default="value", *args, **kwargs):
    pass

# Lambda functions
square = lambda x: x ** 2
add = lambda x, y: x + y

# Higher-order functions
numbers = [1, 2, 3, 4, 5]
squares = list(map(lambda x: x**2, numbers))
evens = list(filter(lambda x: x % 2 == 0, numbers))

from functools import reduce
product = reduce(lambda x, y: x * y, numbers)
```

### Classes
```python
# Basic class
class Person:
    # Class variable
    species = "Homo sapiens"
    
    def __init__(self, name, age):
        # Instance variables
        self.name = name
        self.age = age
    
    def greet(self):
        return f"Hi, I'm {self.name}"
    
    def __str__(self):
        return f"Person(name={self.name}, age={self.age})"
    
    def __repr__(self):
        return f"Person('{self.name}', {self.age})"

# Inheritance
class Student(Person):
    def __init__(self, name, age, student_id):
        super().__init__(name, age)
        self.student_id = student_id
    
    def study(self):
        return f"{self.name} is studying"

# Property decorators
class Circle:
    def __init__(self, radius):
        self._radius = radius
    
    @property
    def radius(self):
        return self._radius
    
    @radius.setter
    def radius(self, value):
        if value < 0:
            raise ValueError("Radius cannot be negative")
        self._radius = value
    
    @property
    def area(self):
        return 3.14159 * self._radius ** 2

# Class methods and static methods
class MathUtils:
    @staticmethod
    def add(x, y):
        return x + y
    
    @classmethod
    def create_from_string(cls, string):
        # Alternative constructor
        return cls()
```

---

## Error Handling

### Try-Except Blocks
```python
try:
    # Code that might raise an exception
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero!")
except (ValueError, TypeError) as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
else:
    # Executes if no exception occurred
    print("Success!")
finally:
    # Always executes
    print("Cleanup code here")

# Raising exceptions
def validate_age(age):
    if age < 0:
        raise ValueError("Age cannot be negative")
    if age > 150:
        raise ValueError("Age seems unrealistic")
```

### Custom Exceptions
```python
class CustomError(Exception):
    """Custom exception class."""
    def __init__(self, message, error_code=None):
        super().__init__(message)
        self.error_code = error_code

# Usage
try:
    raise CustomError("Something went wrong", 500)
except CustomError as e:
    print(f"Error: {e}, Code: {e.error_code}")
```

---

## File I/O & Context Managers

### File Operations
```python
# Reading files
with open('file.txt', 'r') as f:
    content = f.read()          # Read entire file
    lines = f.readlines()       # Read all lines as list
    
    for line in f:              # Iterate line by line
        print(line.strip())

# Writing files
with open('output.txt', 'w') as f:
    f.write("Hello, World!")
    f.writelines(["Line 1\n", "Line 2\n"])

# File modes
'r'   # Read (default)
'w'   # Write (overwrites existing)
'a'   # Append
'r+'  # Read and write
'rb'  # Read binary
'wb'  # Write binary

# Path handling
from pathlib import Path

path = Path('data') / 'file.txt'
if path.exists():
    content = path.read_text()
    lines = path.read_text().splitlines()

path.write_text("New content")
```

### Context Managers
```python
# Built-in context managers
with open('file.txt') as f, open('output.txt', 'w') as out:
    out.write(f.read())

# Custom context manager
class Timer:
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"Elapsed: {time.time() - self.start:.2f}s")

with Timer():
    # Code to time
    time.sleep(1)

# Context manager decorator
from contextlib import contextmanager

@contextmanager
def managed_resource():
    resource = acquire_resource()
    try:
        yield resource
    finally:
        release_resource(resource)
```

---

## Comprehensions & Generators

### List Comprehensions
```python
# Basic syntax: [expression for item in iterable if condition]
squares = [x**2 for x in range(10)]
evens = [x for x in range(20) if x % 2 == 0]
words = [word.upper() for word in ["hello", "world"]]

# Nested loops
pairs = [(x, y) for x in range(3) for y in range(3)]

# Conditional expression
result = [x if x > 0 else 0 for x in [-2, -1, 0, 1, 2]]
```

### Dictionary & Set Comprehensions
```python
# Dictionary comprehension
squares_dict = {x: x**2 for x in range(5)}
filtered_dict = {k: v for k, v in original_dict.items() if v > 10}

# Set comprehension
unique_lengths = {len(word) for word in ["hello", "world", "python"]}
```

### Generator Expressions & Functions
```python
# Generator expression (lazy evaluation)
squares_gen = (x**2 for x in range(1000000))  # Memory efficient

# Generator function
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# Using generators
for num in fibonacci(10):
    print(num)

# Generator with send()
def accumulator():
    total = 0
    while True:
        value = yield total
        if value is not None:
            total += value

acc = accumulator()
next(acc)              # Prime the generator
acc.send(5)            # Send value
acc.send(3)            # Send another value
```

---

## Decorators & Metaclasses

### Function Decorators
```python
# Simple decorator
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start:.4f}s")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(1)
    return "Done"

# Decorator with arguments
def repeat(times):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(3)
def greet(name):
    print(f"Hello, {name}!")

# functools.wraps preserves metadata
from functools import wraps

def my_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
```

### Class Decorators
```python
# Property-like decorators
class MyClass:
    @staticmethod
    def static_method():
        pass
    
    @classmethod
    def class_method(cls):
        pass
    
    @property
    def computed_value(self):
        return self._value * 2

# Class decorator
def add_str_method(cls):
    def __str__(self):
        return f"{cls.__name__} instance"
    cls.__str__ = __str__
    return cls

@add_str_method
class MyClass:
    pass
```

---

## Built-in Functions

### Essential Built-ins
```python
# Type conversion
int("42")                 # String to int
float("3.14")             # String to float
str(42)                   # Any to string
bool(1)                   # Any to boolean
list("hello")             # Iterable to list
tuple([1, 2, 3])          # Iterable to tuple
set([1, 2, 2, 3])         # Iterable to set
dict([('a', 1), ('b', 2)]) # Pairs to dict

# Math functions
abs(-5)                   # Absolute value
round(3.14159, 2)         # Round to 2 decimal places
pow(2, 3)                 # 2^3
divmod(17, 5)             # (quotient, remainder)

# Sequence functions
len([1, 2, 3])            # Length
max([1, 5, 3])            # Maximum
min([1, 5, 3])            # Minimum
sum([1, 2, 3])            # Sum
sorted([3, 1, 2])         # Sorted copy
reversed([1, 2, 3])       # Reverse iterator

# Iteration helpers
enumerate(['a', 'b', 'c']) # (index, value) pairs
zip([1, 2], ['a', 'b'])   # Parallel iteration
all([True, True, False])  # All truthy?
any([False, True, False]) # Any truthy?

# Object introspection
type(42)                  # <class 'int'>
isinstance(42, int)       # True
hasattr(obj, 'attr')      # Has attribute?
getattr(obj, 'attr', default) # Get attribute
setattr(obj, 'attr', value) # Set attribute
dir(obj)                  # List attributes
vars(obj)                 # Object's __dict__
```

### Advanced Built-ins
```python
# Functional programming
map(func, iterable)       # Apply function to each element
filter(predicate, iterable) # Filter elements
reduce(func, iterable)    # Reduce to single value (from functools)

# Dynamic code execution
eval("2 + 2")             # Evaluate expression
exec("x = 5")             # Execute statement
compile(code, '<string>', 'eval') # Compile code

# Memory and performance
id(obj)                   # Object memory address
hash(obj)                 # Object hash value
sys.getsizeof(obj)        # Object memory size
```

---

## Standard Library Essentials

### Collections Module
```python
from collections import (
    defaultdict, Counter, deque, namedtuple, OrderedDict
)

# defaultdict - dict with default factory
dd = defaultdict(list)
dd['key'].append('value')  # No KeyError

# Counter - count hashable objects
counter = Counter("hello world")
counter.most_common(3)     # Most frequent items

# deque - double-ended queue
dq = deque([1, 2, 3])
dq.appendleft(0)           # Add to left
dq.pop()                   # Remove from right

# namedtuple - immutable data structure
Point = namedtuple('Point', ['x', 'y'])
p = Point(1, 2)
```

### Itertools Module
```python
import itertools

# Infinite iterators
itertools.count(10, 2)     # 10, 12, 14, 16, ...
itertools.cycle(['A', 'B', 'C']) # A, B, C, A, B, C, ...
itertools.repeat('hello', 3) # hello, hello, hello

# Finite iterators
itertools.chain([1, 2], [3, 4]) # 1, 2, 3, 4
itertools.compress(['A', 'B', 'C'], [1, 0, 1]) # A, C
itertools.dropwhile(lambda x: x < 5, [1, 3, 6, 2]) # 6, 2

# Combinatorial iterators
itertools.permutations(['A', 'B', 'C'], 2) # AB, AC, BA, BC, CA, CB
itertools.combinations(['A', 'B', 'C'], 2)  # AB, AC, BC
itertools.product(['A', 'B'], [1, 2])       # A1, A2, B1, B2
```

### Datetime Module
```python
from datetime import datetime, date, time, timedelta

# Current time
now = datetime.now()
today = date.today()

# Create specific datetime
dt = datetime(2023, 12, 25, 15, 30, 0)

# String parsing and formatting
dt = datetime.strptime("2023-12-25", "%Y-%m-%d")
formatted = dt.strftime("%B %d, %Y")  # December 25, 2023

# Timedelta arithmetic
tomorrow = today + timedelta(days=1)
week_ago = now - timedelta(weeks=1)
```

### OS and Path Operations
```python
import os
from pathlib import Path

# OS operations
os.getcwd()               # Current working directory
os.chdir('/path/to/dir')  # Change directory
os.listdir('.')          # List directory contents
os.makedirs('path/to/dir', exist_ok=True) # Create directories

# Environment variables
os.environ['HOME']
os.getenv('PYTHON_PATH', default='')

# Modern path handling with pathlib
path = Path('/usr/local/bin')
path.exists()             # Check if exists
path.is_file()            # Is it a file?
path.is_dir()             # Is it a directory?
path.parent               # Parent directory
path.name                 # File name
path.suffix               # File extension
path.stem                 # File name without extension

# Path operations
new_path = path / 'python'  # Join paths
path.glob('*.py')          # Find files matching pattern
path.rglob('*.txt')        # Recursive glob
```

---

## Type Hints & Annotations

### Basic Type Hints
```python
from typing import List, Dict, Optional, Union, Tuple, Callable

# Basic types
def greet(name: str) -> str:
    return f"Hello, {name}!"

age: int = 25
price: float = 19.99
is_active: bool = True

# Collections
names: List[str] = ["Alice", "Bob"]
scores: Dict[str, int] = {"Alice": 95, "Bob": 87}
coordinates: Tuple[float, float] = (1.5, 2.3)

# Optional and Union
def process_data(data: Optional[str] = None) -> Union[str, int]:
    if data is None:
        return 0
    return data.upper()

# Callable types
def apply_operation(func: Callable[[int, int], int], a: int, b: int) -> int:
    return func(a, b)
```

### Advanced Type Hints (Python 3.9+)
```python
# Built-in generics (no imports needed)
def process_items(items: list[str]) -> dict[str, int]:
    return {item: len(item) for item in items}

# Literal types
from typing import Literal

def set_mode(mode: Literal["read", "write", "append"]) -> None:
    pass

# TypedDict for structured dictionaries
from typing import TypedDict

class PersonDict(TypedDict):
    name: str
    age: int
    email: str

def create_person(name: str, age: int, email: str) -> PersonDict:
    return {"name": name, "age": age, "email": email}

# Generic classes
from typing import TypeVar, Generic

T = TypeVar('T')

class Stack(Generic[T]):
    def __init__(self) -> None:
        self._items: list[T] = []
    
    def push(self, item: T) -> None:
        self._items.append(item)
    
    def pop(self) -> T:
        return self._items.pop()
```

---

## Performance & Optimization

### Timing Code
```python
import time
import timeit
from functools import wraps

# Simple timing
start = time.perf_counter()
# ... code to time ...
elapsed = time.perf_counter() - start

# Timeit for precise measurements
time_taken = timeit.timeit(
    'sum(range(100))',
    number=10000
)

# Profile decorator
def profile(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__}: {end - start:.4f}s")
        return result
    return wrapper
```

### Memory Optimization
```python
# Use generators for large datasets
def read_large_file(filename):
    with open(filename) as f:
        for line in f:
            yield line.strip()

# Use __slots__ to reduce memory in classes
class Point:
    __slots__ = ['x', 'y']
    def __init__(self, x, y):
        self.x = x
        self.y = y

# Use array for numeric data
from array import array
numbers = array('i', [1, 2, 3, 4, 5])  # int array

# Use sets for membership testing
large_set = set(range(1000000))
if 500000 in large_set:  # O(1) average case
    pass
```

### Algorithm Optimization
```python
# Use collections.Counter for counting
from collections import Counter
counts = Counter(some_list)

# Use bisect for sorted insertions
import bisect
sorted_list = [1, 3, 5, 7, 9]
bisect.insort(sorted_list, 4)  # Maintains sorted order

# Use functools.lru_cache for memoization
from functools import lru_cache

@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

---

## Testing & Debugging

### Unit Testing with pytest
```python
# test_example.py
import pytest

def add(x, y):
    return x + y

def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0

def test_add_strings():
    assert add("hello", "world") == "helloworld"

# Fixtures
@pytest.fixture
def sample_data():
    return [1, 2, 3, 4, 5]

def test_sum_with_fixture(sample_data):
    assert sum(sample_data) == 15

# Parametrized tests
@pytest.mark.parametrize("x,y,expected", [
    (1, 2, 3),
    (0, 0, 0),
    (-1, 1, 0),
])
def test_add_parametrized(x, y, expected):
    assert add(x, y) == expected

# Exception testing
def test_division_by_zero():
    with pytest.raises(ZeroDivisionError):
        1 / 0
```

### Debugging
```python
# Print debugging
print(f"Variable x = {x}")

# Logging
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")

# Python debugger
import pdb
pdb.set_trace()  # Breakpoint

# Rich debugging with breakpoint() (Python 3.7+)
breakpoint()  # Uses PYTHONBREAKPOINT environment variable

# Assertions for debugging
def calculate_average(numbers):
    assert len(numbers) > 0, "Cannot calculate average of empty list"
    assert all(isinstance(n, (int, float)) for n in numbers), "All items must be numbers"
    return sum(numbers) / len(numbers)
```

---

## Modern Python Features (3.10+)

### Pattern Matching (3.10+)
```python
def handle_data(data):
    match data:
        case int() if data > 0:
            return f"Positive integer: {data}"
        case int() if data < 0:
            return f"Negative integer: {data}"
        case 0:
            return "Zero"
        case str() if len(data) > 0:
            return f"Non-empty string: {data}"
        case []:
            return "Empty list"
        case [x] if isinstance(x, int):
            return f"Single integer list: {x}"
        case [x, y]:
            return f"Two-element list: {x}, {y}"
        case {"name": str(name), "age": int(age)}:
            return f"Person: {name}, {age} years old"
        case _:
            return "Unknown data type"

# Usage examples
print(handle_data(42))                    # Positive integer: 42
print(handle_data("hello"))               # Non-empty string: hello
print(handle_data([1, 2]))               # Two-element list: 1, 2
print(handle_data({"name": "Alice", "age": 30})) # Person: Alice, 30 years old
```

### Union Types with | (3.10+)
```python
# Old way
from typing import Union
def process(value: Union[int, str]) -> str:
    pass

# New way (3.10+)
def process(value: int | str) -> str:
    if isinstance(value, int):
        return str(value)
    return value

# Multiple types
def handle_multiple(data: int | float | str | None) -> str:
    match data:
        case None:
            return "No data"
        case int() | float():
            return f"Number: {data}"
        case str():
            return f"Text: {data}"
```

### Enhanced Error Messages (3.10+)
```python
# Better error location in tracebacks
def complex_calculation():
    result = (
        some_function(param1, param2) +
        another_function(param3, param4) *
        third_function(param5, param6)
    )
    return result

# Parenthesized context managers (3.10+)
with (
    open('file1.txt') as f1,
    open('file2.txt') as f2
):
    data = f1.read() + f2.read()
```

### Dataclasses Enhancements (3.10+)
```python
from dataclasses import dataclass, field

@dataclass
class Person:
    name: str
    age: int = 0
    hobbies: list[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.age < 0:
            raise ValueError("Age cannot be negative")

# Slots for better performance (3.10+)
@dataclass(slots=True)
class Point:
    x: float
    y: float
```

---

## Pro Tips & Best Practices

### Pythonic Code
```python
# Use enumerate instead of range(len())
for i, item in enumerate(items):
    print(f"{i}: {item}")

# Use zip for parallel iteration
for name, score in zip(names, scores):
    print(f"{name}: {score}")

# Use dict.get() with defaults
value = my_dict.get('key', 'default_value')

# Use list comprehensions over map/filter when readable
squares = [x**2 for x in range(10)]
evens = [x for x in range(20) if x % 2 == 0]

# Use f-strings for formatting
message = f"Hello {name}, you have {count} items"

# Use context managers for resource management
with open('file.txt') as f:
    content = f.read()

# Use pathlib for file operations
from pathlib import Path
path = Path('data') / 'file.txt'
if path.exists():
    content = path.read_text()
```

### Performance Tips
```python
# Use sets for membership testing
if item in large_set:  # O(1) average
    pass

# Use collections.deque for queues
from collections import deque
queue = deque()
queue.appendleft(item)  # O(1)
queue.pop()             # O(1)

# Use generators for memory efficiency
def read_lines():
    with open('huge_file.txt') as f:
        for line in f:
            yield line.strip()

# Cache expensive computations
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_function(n):
    # Complex computation here
    return result

# Use operator module for simple operations
from operator import itemgetter, attrgetter
students.sort(key=itemgetter('grade'))
employees.sort(key=attrgetter('salary'))
```

---

**📚 Additional Resources:**
- [Python Documentation](https://docs.python.org/3/)
- [PEP Index](https://www.python.org/dev/peps/)
- [Real Python Tutorials](https://realpython.com/)
- [Python Package Index (PyPI)](https://pypi.org/)

**🔗 Related Cheatsheets:**
- [NumPy Cheatsheet](NUMPY_CHEATSHEET.md)
- [Pandas Cheatsheet](PANDAS_CHEATSHEET.md)
- [PySpark Cheatsheet](PYSPARK_CHEATSHEET.md)
- [PyTorch Cheatsheet](PYTORCH_CHEATSHEET.md)

---
*Last Updated: January 8, 2026*
