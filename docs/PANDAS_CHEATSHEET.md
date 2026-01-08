# Pandas 2.3.3 Comprehensive Cheatsheet

**Level:** SparkNotes/Unbound Quality Reference  
**Date:** January 8, 2026  
**Pandas Version:** 2.3.3+  
**Performance:** 40x speedup vs Python, 60% less memory with copy-on-write

## Table of Contents

- [Quick Start & Core Concepts](#quick-start--core-concepts)
- [Data Structures](#data-structures)
- [Data Input/Output](#data-inputoutput)
- [Data Selection & Filtering](#data-selection--filtering)
- [Data Manipulation](#data-manipulation)
- [Grouping & Aggregation](#grouping--aggregation)
- [Merging & Joining](#merging--joining)
- [Time Series Analysis](#time-series-analysis)
- [String Operations](#string-operations)
- [Missing Data Handling](#missing-data-handling)
- [Performance Optimization](#performance-optimization)
- [Advanced Techniques](#advanced-techniques)

---

## Quick Start & Core Concepts

### Import Convention
```python
import pandas as pd
import numpy as np
import pandas.api.types as ptypes  # For type checking
```

### Core Data Structures
```python
# Series - 1D labeled array
s = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'])
print(s.dtype)           # int64
print(s.index)           # Index(['a', 'b', 'c', 'd', 'e'])
print(s.values)          # [1 2 3 4 5] (numpy array)

# DataFrame - 2D labeled data structure
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['NYC', 'LA', 'Chicago']
})
print(df.shape)          # (3, 3)
print(df.columns)        # Index(['Name', 'Age', 'City'])
print(df.index)          # RangeIndex(start=0, stop=3, step=1)
```

### Key Concepts
```python
# Index - labels for rows/columns
# loc - label-based indexing
# iloc - integer-based indexing
# Copy-on-Write (CoW) - memory optimization in Pandas 2.0+
# Method chaining - fluent interface pattern

# Enable copy-on-write for better performance
pd.options.mode.copy_on_write = True
```

---

## Data Structures

### Series Creation
```python
# From list/array
s1 = pd.Series([1, 2, 3, 4])
s2 = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
s3 = pd.Series(np.random.randn(5))

# From dictionary
s4 = pd.Series({'a': 1, 'b': 2, 'c': 3})

# From scalar
s5 = pd.Series(5, index=['x', 'y', 'z'])

# Named series
s6 = pd.Series([1, 2, 3], name='my_series')

# Series with specific dtype
s7 = pd.Series([1, 2, 3], dtype='float64')
s8 = pd.Series(['a', 'b', 'c'], dtype='category')
```

### DataFrame Creation
```python
# From dictionary
df1 = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]
})

# From list of dictionaries
df2 = pd.DataFrame([
    {'A': 1, 'B': 4, 'C': 7},
    {'A': 2, 'B': 5, 'C': 8},
    {'A': 3, 'B': 6, 'C': 9}
])

# From 2D array with custom index/columns
data = np.random.randn(4, 3)
df3 = pd.DataFrame(
    data,
    index=['row1', 'row2', 'row3', 'row4'],
    columns=['col1', 'col2', 'col3']
)

# From Series
s = pd.Series([1, 2, 3])
df4 = s.to_frame('column_name')

# Empty DataFrame with structure
df5 = pd.DataFrame(columns=['A', 'B', 'C'])
df6 = pd.DataFrame(index=range(5), columns=['X', 'Y'])

# From records
records = [('Alice', 25), ('Bob', 30), ('Charlie', 35)]
df7 = pd.DataFrame.from_records(
    records,
    columns=['Name', 'Age']
)
```

### Index and MultiIndex
```python
# Custom index
df = pd.DataFrame({'A': [1, 2, 3]})
df.index = ['x', 'y', 'z']

# DatetimeIndex
dates = pd.date_range('2024-01-01', periods=5, freq='D')
df_time = pd.DataFrame({'values': [1, 2, 3, 4, 5]}, index=dates)

# MultiIndex (hierarchical)
arrays = [
    ['A', 'A', 'B', 'B'],
    ['one', 'two', 'one', 'two']
]
multi_idx = pd.MultiIndex.from_arrays(arrays, names=['first', 'second'])
df_multi = pd.DataFrame({'values': [1, 2, 3, 4]}, index=multi_idx)

# MultiIndex from tuples
tuples = [('A', 'one'), ('A', 'two'), ('B', 'one'), ('B', 'two')]
multi_idx2 = pd.MultiIndex.from_tuples(tuples)

# MultiIndex from product
multi_idx3 = pd.MultiIndex.from_product([['A', 'B'], ['one', 'two']])

# Working with MultiIndex
df_multi.loc['A']                    # Select first level
df_multi.loc[('A', 'one')]          # Select specific combination
df_multi.unstack()                  # Convert to columns
df_multi.reset_index()              # Convert to regular columns
```

---

## Data Input/Output

### Reading Data
```python
# CSV files
df = pd.read_csv('file.csv')
df = pd.read_csv('file.csv', 
                 index_col=0,           # Use first column as index
                 parse_dates=['date'],  # Parse date columns
                 dtype={'col': 'str'},  # Specify dtypes
                 na_values=['NULL'],    # Custom NA values
                 skiprows=1,            # Skip rows
                 nrows=1000,            # Read only first 1000 rows
                 chunksize=1000)        # Read in chunks

# Excel files
df = pd.read_excel('file.xlsx', sheet_name='Sheet1')
df = pd.read_excel('file.xlsx', sheet_name=None)  # All sheets

# JSON files
df = pd.read_json('file.json')
df = pd.read_json('file.json', orient='records')

# SQL databases
import sqlite3
conn = sqlite3.connect('database.db')
df = pd.read_sql_query('SELECT * FROM table', conn)
df = pd.read_sql_table('table_name', conn)

# Web data
df = pd.read_html('https://example.com/table.html')  # Returns list
df = pd.read_clipboard()                             # From clipboard

# Parquet files
df = pd.read_parquet('file.parquet')

# Feather files
df = pd.read_feather('file.feather')

# HDF5 files
df = pd.read_hdf('file.h5', key='data')

# Pickle files
df = pd.read_pickle('file.pkl')
```

### Writing Data
```python
# CSV
df.to_csv('output.csv', index=False)
df.to_csv('output.csv', columns=['col1', 'col2'])  # Specific columns

# Excel
df.to_excel('output.xlsx', sheet_name='Sheet1', index=False)

# Multiple sheets
with pd.ExcelWriter('output.xlsx') as writer:
    df1.to_excel(writer, sheet_name='Sheet1')
    df2.to_excel(writer, sheet_name='Sheet2')

# JSON
df.to_json('output.json', orient='records')

# SQL
df.to_sql('table_name', conn, if_exists='replace', index=False)

# Parquet (efficient for large datasets)
df.to_parquet('output.parquet', compression='snappy')

# Feather (fast read/write)
df.to_feather('output.feather')

# HDF5
df.to_hdf('output.h5', key='data', mode='w')

# Pickle (preserves all pandas features)
df.to_pickle('output.pkl')
```

### Data Inspection
```python
# Quick overview
df.head(10)              # First 10 rows
df.tail(5)               # Last 5 rows
df.sample(10)            # Random 10 rows
df.shape                 # (rows, columns)
df.size                  # Total elements
df.columns               # Column names
df.index                 # Index
df.dtypes                # Data types

# Statistical summary
df.info()                # Memory usage, dtypes, non-null counts
df.describe()            # Statistical summary for numeric columns
df.describe(include='all')  # Summary for all columns
df.nunique()             # Number of unique values per column
df.value_counts()        # For Series
df['column'].value_counts()  # Value counts for specific column

# Memory usage
df.memory_usage(deep=True)   # Detailed memory usage
df.info(memory_usage='deep') # Include in info

# Missing data overview
df.isnull().sum()        # Count of missing values per column
df.isnull().any()        # Any missing values per column
df.isnull().all()        # All missing values per column
```

---

## Data Selection & Filtering

### Column Selection
```python
# Single column
df['Name']               # Returns Series
df[['Name']]             # Returns DataFrame

# Multiple columns
df[['Name', 'Age']]
df.loc[:, 'Name':'Age']  # Slice by labels (inclusive)
df.iloc[:, 0:2]          # Slice by position (exclusive end)

# Dynamic column selection
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols]

# Column selection by pattern
df.filter(regex='^A')    # Columns starting with 'A'
df.filter(like='_total') # Columns containing '_total'
df.filter(items=['Name', 'Age'])  # Specific columns
```

### Row Selection
```python
# By index label
df.loc['row_label']
df.loc['start':'end']    # Slice by labels (inclusive)

# By integer position
df.iloc[0]               # First row
df.iloc[-1]              # Last row
df.iloc[0:5]             # First 5 rows
df.iloc[::2]             # Every 2nd row

# Boolean indexing
df[df['Age'] > 25]       # Rows where Age > 25
df[df['Name'].isin(['Alice', 'Bob'])]  # Name in list
df[~df['Name'].isin(['Alice'])]        # Name not in list (~ means NOT)

# Multiple conditions
df[(df['Age'] > 25) & (df['City'] == 'NYC')]  # AND
df[(df['Age'] > 30) | (df['City'] == 'LA')]   # OR

# Query method (more readable for complex conditions)
df.query('Age > 25 and City == "NYC"')
df.query('Age > @threshold')  # Use external variable
df.query('Name in ["Alice", "Bob"]')

# First/last N rows by condition
df.head(10)
df.tail(5)
df.nlargest(3, 'Age')    # Top 3 by Age
df.nsmallest(2, 'Age')   # Bottom 2 by Age
```

### Advanced Selection
```python
# loc and iloc combinations
df.loc[df['Age'] > 25, 'Name']           # Specific rows and columns
df.loc[0:2, 'Name':'Age']                # Slice rows and columns
df.iloc[0:3, 1:3]                        # Position-based slice

# Conditional selection with assignment
df.loc[df['Age'] > 30, 'Category'] = 'Senior'
df.loc[df['City'] == 'NYC', ['Bonus']] = 1000

# Where method
df.where(df > 0, -999)   # Replace values that don't meet condition

# Mask method (opposite of where)
df.mask(df < 0, 999)     # Replace values that meet condition

# Sample selection
df.sample(n=5)           # Random 5 rows
df.sample(frac=0.1)      # Random 10% of rows
df.sample(n=5, weights=df['Age'])  # Weighted sampling

# Unique and duplicates
df['Name'].unique()      # Unique values
df.drop_duplicates()     # Remove duplicate rows
df.drop_duplicates(subset=['Name'])  # Based on specific columns
df.duplicated()          # Boolean mask of duplicates
```

---

## Data Manipulation

### Adding and Removing Columns
```python
# Add new columns
df['NewCol'] = 0         # Constant value
df['Age2'] = df['Age'] * 2  # Based on existing column
df['FullName'] = df['FirstName'] + ' ' + df['LastName']  # String concat

# Add multiple columns
df = df.assign(
    NewCol1=df['Age'] + 10,
    NewCol2=lambda x: x['Age'] * 2,
    NewCol3='constant'
)

# Insert at specific position
df.insert(1, 'NewCol', [1, 2, 3])

# Remove columns
df.drop('ColumnName', axis=1, inplace=True)  # Single column
df.drop(['Col1', 'Col2'], axis=1, inplace=True)  # Multiple columns
del df['ColumnName']     # Alternative deletion
df = df[['Col1', 'Col2']]  # Keep only specific columns
```

### Adding and Removing Rows
```python
# Add single row
new_row = pd.Series({'Name': 'David', 'Age': 40}, name='new_index')
df = pd.concat([df, new_row.to_frame().T])

# Add multiple rows
new_rows = pd.DataFrame({'Name': ['Eve', 'Frank'], 'Age': [28, 45]})
df = pd.concat([df, new_rows], ignore_index=True)

# Insert row at specific position
df.loc[1.5] = ['Alice', 25]  # Insert between index 1 and 2
df = df.sort_index().reset_index(drop=True)

# Remove rows
df.drop(0, inplace=True)     # Drop by index
df.drop([0, 1], inplace=True)  # Drop multiple indices
df = df[df['Age'] >= 18]     # Drop by condition
```

### Data Transformation
```python
# Apply functions
df['Age'].apply(lambda x: x * 2)         # Apply to column
df.apply(lambda x: x.max() - x.min())    # Apply to each column
df.apply(lambda x: x.sum(), axis=1)      # Apply to each row

# Map values
mapping = {25: 'Young', 30: 'Middle', 35: 'Old'}
df['AgeGroup'] = df['Age'].map(mapping)

# Replace values
df['Name'].replace('Alice', 'Alicia', inplace=True)
df.replace({'Alice': 'Alicia', 'Bob': 'Robert'}, inplace=True)
df.replace([25, 30], [20, 25], inplace=True)  # Multiple replacements

# Transform (apply function and return same shape)
df.transform(lambda x: x - x.mean())     # Center data
df['Age'].transform('sqrt')              # Apply numpy function

# Pipe (method chaining)
result = (df
    .pipe(lambda x: x[x['Age'] > 25])
    .pipe(lambda x: x.assign(AgeSquared=x['Age']**2))
    .pipe(lambda x: x.sort_values('Age'))
)

# Assign (add columns with method chaining)
df = (df
    .assign(Age2=df['Age'] * 2)
    .assign(AgeGroup=lambda x: pd.cut(x['Age'], bins=[0, 30, 60, 100]))
)
```

### Sorting
```python
# Sort by single column
df.sort_values('Age')                    # Ascending
df.sort_values('Age', ascending=False)   # Descending

# Sort by multiple columns
df.sort_values(['City', 'Age'])
df.sort_values(['City', 'Age'], ascending=[True, False])

# Sort by index
df.sort_index()          # Sort by row index
df.sort_index(axis=1)    # Sort by column names

# Sort with custom key
df.sort_values('Name', key=lambda x: x.str.lower())
```

### Reshaping Data
```python
# Pivot tables
pivot_df = df.pivot_table(
    values='Sales',
    index='Date',
    columns='Product',
    aggfunc='sum',
    fill_value=0
)

# Pivot (simple reshape, no aggregation)
df_pivot = df.pivot(index='Date', columns='Product', values='Sales')

# Melt (unpivot)
df_melted = pd.melt(df, 
                   id_vars=['Name'], 
                   value_vars=['Age', 'Salary'],
                   var_name='Metric', 
                   value_name='Value')

# Stack and unstack
df_stacked = df.set_index(['Name', 'City']).stack()
df_unstacked = df_stacked.unstack()

# Transpose
df_transposed = df.T

# Wide to long format
df_long = pd.wide_to_long(df, 
                         stubnames='Q', 
                         i='ID', 
                         j='Quarter', 
                         sep='')
```

---

## Grouping & Aggregation

### Basic Grouping
```python
# Group by single column
grouped = df.groupby('City')
grouped.mean()           # Mean for each group
grouped.sum()            # Sum for each group
grouped.count()          # Count non-null values
grouped.size()           # Count all values (including null)

# Group by multiple columns
df.groupby(['City', 'Department']).mean()

# Group with custom aggregation
df.groupby('City').agg({
    'Age': 'mean',
    'Salary': ['sum', 'count'],
    'Name': lambda x: x.nunique()
})

# Named aggregations (pandas 0.25+)
df.groupby('City').agg(
    avg_age=('Age', 'mean'),
    total_salary=('Salary', 'sum'),
    unique_names=('Name', 'nunique')
)
```

### Advanced Grouping
```python
# Group by multiple functions
def age_group(age):
    if age < 30:
        return 'Young'
    elif age < 50:
        return 'Middle'
    else:
        return 'Old'

df.groupby(df['Age'].apply(age_group)).mean()

# Group by time periods
df_time = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=100, freq='D'),
    'sales': np.random.randn(100)
})
df_time.set_index('date').groupby(pd.Grouper(freq='M')).sum()

# Group by bins
df.groupby(pd.cut(df['Age'], bins=[0, 30, 50, 100])).mean()
df.groupby(pd.qcut(df['Salary'], q=4)).mean()  # Quantile-based bins

# Transform (return same size as input)
df['Age_centered'] = df.groupby('City')['Age'].transform('mean')
df['Salary_zscore'] = df.groupby('City')['Salary'].transform(
    lambda x: (x - x.mean()) / x.std()
)

# Filter groups
df.groupby('City').filter(lambda x: len(x) > 5)  # Groups with >5 members
df.groupby('City').filter(lambda x: x['Age'].mean() > 30)

# Apply custom function to groups
def top_performers(group):
    return group.nlargest(2, 'Salary')

df.groupby('City').apply(top_performers)
```

### Aggregation Functions
```python
# Built-in aggregations
agg_funcs = [
    'count', 'sum', 'mean', 'median', 'std', 'var',
    'min', 'max', 'first', 'last', 'nunique'
]

df.groupby('City').agg(agg_funcs)

# Custom aggregation functions
def range_func(x):
    return x.max() - x.min()

def q75(x):
    return x.quantile(0.75)

df.groupby('City').agg({
    'Age': [range_func, q75, 'mean'],
    'Salary': ['sum', 'count']
})

# Weighted average
def weighted_avg(group):
    return (group['Salary'] * group['Hours']).sum() / group['Hours'].sum()

df.groupby('Department').apply(weighted_avg)

# Multiple aggregations with different functions
df.groupby('City').agg(
    age_stats=('Age', lambda x: f"{x.mean():.1f}±{x.std():.1f}"),
    salary_range=('Salary', lambda x: f"{x.min()}-{x.max()}"),
    count=('Name', 'count')
)
```

### Window Functions (Rolling, Expanding, EWM)
```python
# Rolling windows
df['Sales_MA7'] = df['Sales'].rolling(window=7).mean()      # 7-day moving average
df['Sales_MA7_min3'] = df['Sales'].rolling(window=7, min_periods=3).mean()

# Rolling with different aggregations
df['Sales'].rolling(7).agg(['mean', 'std', 'min', 'max'])

# Expanding windows (cumulative)
df['Cumulative_Mean'] = df['Sales'].expanding().mean()
df['Cumulative_Sum'] = df['Sales'].expanding().sum()

# Exponentially weighted moving average
df['Sales_EWM'] = df['Sales'].ewm(span=10).mean()
df['Sales_EWM_alpha'] = df['Sales'].ewm(alpha=0.3).mean()

# Group-wise rolling
df.groupby('Product')['Sales'].rolling(7).mean()

# Center the window
df['Sales'].rolling(7, center=True).mean()
```

---

## Merging & Joining

### DataFrame Merging
```python
df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'value1': [1, 2, 3]})
df2 = pd.DataFrame({'key': ['A', 'B', 'D'], 'value2': [4, 5, 6]})

# Inner join (default)
pd.merge(df1, df2, on='key')

# Different join types
pd.merge(df1, df2, on='key', how='inner')    # Intersection
pd.merge(df1, df2, on='key', how='outer')    # Union
pd.merge(df1, df2, on='key', how='left')     # All from left
pd.merge(df1, df2, on='key', how='right')    # All from right

# Merge on multiple columns
pd.merge(df1, df2, on=['key1', 'key2'])

# Merge with different column names
pd.merge(df1, df2, left_on='key1', right_on='key2')

# Merge on index
pd.merge(df1, df2, left_index=True, right_index=True)

# Suffix for overlapping columns
pd.merge(df1, df2, on='key', suffixes=('_left', '_right'))

# Indicator to show merge source
pd.merge(df1, df2, on='key', how='outer', indicator=True)
```

### Concatenation
```python
# Vertical concatenation (stack rows)
pd.concat([df1, df2])                    # Stack vertically
pd.concat([df1, df2], ignore_index=True) # Reset index

# Horizontal concatenation (side by side)
pd.concat([df1, df2], axis=1)

# Concatenate with keys
pd.concat([df1, df2], keys=['first', 'second'])

# Concatenate Series to DataFrame
s1 = pd.Series([1, 2, 3], name='col1')
s2 = pd.Series([4, 5, 6], name='col2')
pd.concat([s1, s2], axis=1)
```

### Join Operations
```python
# DataFrame join (default: left join)
df1.join(df2, rsuffix='_right')

# Join on specific columns
df1.set_index('key').join(df2.set_index('key'))

# Join with different types
df1.join(df2, how='inner')   # Inner join
df1.join(df2, how='outer')   # Outer join
```

### Combine Operations
```python
# Combine first (fill NaNs from first DataFrame with second)
df1.combine_first(df2)

# Combine with custom function
def take_smaller(x, y):
    return x if x <= y else y

df1.combine(df2, take_smaller)

# Update (modify df1 with non-null values from df2)
df1.update(df2)
```

---

## Time Series Analysis

### DateTime Handling
```python
# Create datetime index
dates = pd.date_range('2024-01-01', periods=100, freq='D')
df_time = pd.DataFrame({'value': np.random.randn(100)}, index=dates)

# Parse dates from strings
df['date'] = pd.to_datetime(df['date_string'])
df['date'] = pd.to_datetime(df['date_string'], format='%Y-%m-%d')

# Extract date components
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek
df['quarter'] = df['date'].dt.quarter
df['weekday'] = df['date'].dt.day_name()

# Set datetime index
df = df.set_index('date')
```

### Time-based Selection
```python
# Select by date
df_time['2024-01-01']                    # Specific date
df_time['2024-01']                       # Entire month
df_time['2024-01-01':'2024-01-31']       # Date range

# Select by time components
df_time[df_time.index.month == 1]        # January data
df_time[df_time.index.dayofweek < 5]     # Weekdays only

# Recent data
df_time.last('30D')      # Last 30 days
df_time.first('2M')      # First 2 months

# Between times
df_time.between_time('09:00', '17:00')   # Business hours
```

### Time Series Operations
```python
# Resampling (change frequency)
df_time.resample('M').mean()             # Monthly averages
df_time.resample('Q').sum()              # Quarterly sums
df_time.resample('W').agg(['mean', 'std']) # Weekly stats

# Upsampling (increase frequency)
df_monthly = df_time.resample('M').mean()
df_daily = df_monthly.resample('D').ffill()  # Forward fill

# Period conversion
df_time.to_period('M')                   # Convert to period index
df_time.to_period('M').to_timestamp()    # Back to timestamp

# Timezone handling
df_time_utc = df_time.tz_localize('UTC')
df_time_eastern = df_time_utc.tz_convert('US/Eastern')

# Shift operations
df_time['value_lag1'] = df_time['value'].shift(1)      # Lag 1
df_time['value_lead1'] = df_time['value'].shift(-1)    # Lead 1

# Time differences
df_time['value_diff'] = df_time['value'].diff()        # First difference
df_time['value_pct_change'] = df_time['value'].pct_change()  # Percent change

# Holiday calendar
from pandas.tseries.holiday import USFederalHolidayCalendar
cal = USFederalHolidayCalendar()
holidays = cal.holidays(start='2024-01-01', end='2024-12-31')
```

### Advanced Time Series
```python
# Business day operations
bdays = pd.bdate_range('2024-01-01', periods=50)  # Business days only
df_bday = pd.DataFrame({'value': np.random.randn(50)}, index=bdays)

# Custom business calendar
from pandas.tseries.offsets import CustomBusinessDay
us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
custom_bdays = pd.date_range('2024-01-01', periods=50, freq=us_bd)

# Time zone aware operations
df_tz = df_time.tz_localize('US/Eastern')
df_tz.tz_convert('UTC')

# Seasonal decomposition (requires statsmodels)
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    decomposition = seasonal_decompose(df_time['value'], model='additive', period=30)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
except ImportError:
    print("Install statsmodels for seasonal decomposition")

# Rolling correlations with time
df_time['value2'] = np.random.randn(len(df_time))
rolling_corr = df_time['value'].rolling(30).corr(df_time['value2'])
```

---

## String Operations

### Basic String Operations
```python
# Access string methods with .str
df['Name'].str.upper()          # Uppercase
df['Name'].str.lower()          # Lowercase
df['Name'].str.title()          # Title case
df['Name'].str.capitalize()     # Capitalize first letter

df['Name'].str.len()            # String length
df['Name'].str.strip()          # Remove whitespace
df['Name'].str.lstrip()         # Remove left whitespace
df['Name'].str.rstrip()         # Remove right whitespace

# String concatenation
df['FullName'] = df['First'].str.cat(df['Last'], sep=' ')
df['FullName'] = df['First'] + ' ' + df['Last']  # Alternative

# String replacement
df['Name'].str.replace('Alice', 'Alicia')
df['Name'].str.replace(r'A\w+', 'Anonymous', regex=True)  # Regex replace
```

### String Patterns and Regex
```python
# Contains pattern
df['Name'].str.contains('Al')              # Contains 'Al'
df['Name'].str.contains(r'^A', regex=True) # Starts with 'A'
df['Name'].str.contains('alice', case=False) # Case insensitive

# Starts with / Ends with
df['Name'].str.startswith('A')
df['Name'].str.endswith('e')

# Find and extract
df['Name'].str.find('l')                   # Position of first 'l'
df['Name'].str.findall(r'\w+')             # Find all words

# Extract with regex groups
pattern = r'(\w+)\s+(\w+)'  # First and last name
df['Name'].str.extract(pattern)  # Returns DataFrame with groups
df['Name'].str.extractall(pattern)  # All matches

# Split strings
df['Name'].str.split()             # Split on whitespace
df['Name'].str.split(' ', expand=True)  # Split into columns
df['Name'].str.split(' ', n=1, expand=True)  # Max 1 split

# Slice strings
df['Name'].str[0:3]                # First 3 characters
df['Name'].str[-3:]                # Last 3 characters
```

### Advanced String Operations
```python
# String normalization
df['Name'].str.normalize('NFD')    # Unicode normalization

# Encoding/Decoding
df['Name'].str.encode('utf-8')
df['Name'].str.decode('utf-8')  # If Series contains bytes

# Padding
df['Name'].str.pad(10, side='left', fillchar='0')   # Left pad with zeros
df['Name'].str.center(10, fillchar='-')             # Center with dashes
df['Name'].str.zfill(5)                             # Zero-fill to width 5

# Count pattern occurrences
df['Text'].str.count(r'\w+')       # Count words
df['Text'].str.count('the')        # Count specific word

# String indexing and slicing
df['Name'].str.get(0)              # First character
df['Name'].str.slice(1, 3)         # Characters 1-2
df['Name'].str.slice_replace(0, 1, 'X')  # Replace first char with 'X'

# Categorical string operations
df['Category'] = df['Name'].astype('category')
df['Category'].cat.categories      # View categories
df['Category'].cat.add_categories(['New'])  # Add category
df['Category'].cat.remove_categories(['Old'])  # Remove category
```

### Text Analytics
```python
# Word frequency
all_text = ' '.join(df['Text'].dropna())
words = all_text.split()
word_freq = pd.Series(words).value_counts()

# N-grams
def get_ngrams(text, n=2):
    words = text.split()
    return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]

df['bigrams'] = df['Text'].apply(lambda x: get_ngrams(x, 2))

# Text cleaning pipeline
def clean_text(text):
    if pd.isna(text):
        return text
    # Remove special characters, convert to lowercase, strip whitespace
    return text.str.replace(r'[^\w\s]', '', regex=True).str.lower().str.strip()

df['clean_text'] = clean_text(df['Text'])

# Sentiment analysis (conceptual - requires TextBlob or VADER)
# from textblob import TextBlob
# df['sentiment'] = df['Text'].apply(lambda x: TextBlob(x).sentiment.polarity)
```

---

## Missing Data Handling

### Detecting Missing Data
```python
# Check for missing values
df.isnull()              # Boolean DataFrame
df.isnull().sum()        # Count per column
df.isnull().any()        # Any missing per column
df.isnull().all()        # All missing per column

df.isna()                # Alias for isnull()
df.notna()               # Opposite of isna()

# Missing data info
df.info()                # Shows non-null count
missing_percent = (df.isnull().sum() / len(df)) * 100

# Missing data patterns
import missingno as msno  # Optional library
msno.matrix(df)          # Visualize missing patterns
msno.bar(df)             # Bar chart of missing counts
```

### Handling Missing Values
```python
# Drop missing values
df.dropna()              # Drop rows with any missing values
df.dropna(axis=1)        # Drop columns with any missing values
df.dropna(subset=['col1', 'col2'])  # Drop if missing in specific columns
df.dropna(thresh=2)      # Drop rows with less than 2 non-null values
df.dropna(how='all')     # Drop rows where all values are missing

# Fill missing values
df.fillna(0)             # Fill with constant
df.fillna(df.mean())     # Fill with mean
df.fillna({'col1': 0, 'col2': 'Unknown'})  # Different values per column

# Forward/backward fill
df.fillna(method='ffill')  # Forward fill (use last valid)
df.fillna(method='bfill')  # Backward fill (use next valid)
df.fillna(method='ffill', limit=2)  # Limit consecutive fills

# Interpolation
df.interpolate()                    # Linear interpolation
df.interpolate(method='polynomial', order=2)  # Polynomial
df.interpolate(method='time')       # Time-based interpolation

# Fill with group statistics
df['Age'].fillna(df.groupby('City')['Age'].transform('mean'))

# Advanced missing value imputation
from sklearn.impute import SimpleImputer, KNNImputer

# Simple imputation
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(
    imputer.fit_transform(df.select_dtypes(include=[np.number])),
    columns=df.select_dtypes(include=[np.number]).columns
)

# KNN imputation
knn_imputer = KNNImputer(n_neighbors=5)
df_knn = pd.DataFrame(
    knn_imputer.fit_transform(df.select_dtypes(include=[np.number])),
    columns=df.select_dtypes(include=[np.number]).columns
)
```

### Missing Value Analysis
```python
# Create missing value summary
def missing_summary(df):
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100
    missing_df = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': missing_count,
        'Missing_Percent': missing_percent
    }).sort_values('Missing_Percent', ascending=False)
    return missing_df[missing_df['Missing_Count'] > 0]

# Missing value correlation
def missing_correlation(df):
    missing_df = df.isnull().astype(int)
    return missing_df.corr()

# Impute based on other columns
def conditional_fill(df, target_col, condition_col, fill_value):
    mask = df[condition_col].isnull() & df[target_col].notnull()
    df.loc[mask, condition_col] = fill_value
    return df
```

---

## Performance Optimization

### Memory Optimization
```python
# Check memory usage
df.memory_usage(deep=True)
df.info(memory_usage='deep')

# Optimize data types
def optimize_dtypes(df):
    # Integer optimization
    int_cols = df.select_dtypes(include=['int']).columns
    for col in int_cols:
        if df[col].min() >= 0:
            if df[col].max() < 255:
                df[col] = df[col].astype(np.uint8)
            elif df[col].max() < 65535:
                df[col] = df[col].astype(np.uint16)
            elif df[col].max() < 4294967295:
                df[col] = df[col].astype(np.uint32)
        else:
            if df[col].min() > -128 and df[col].max() < 127:
                df[col] = df[col].astype(np.int8)
            elif df[col].min() > -32768 and df[col].max() < 32767:
                df[col] = df[col].astype(np.int16)
            elif df[col].min() > -2147483648 and df[col].max() < 2147483647:
                df[col] = df[col].astype(np.int32)
    
    # Float optimization
    float_cols = df.select_dtypes(include=['float']).columns
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    # Object to category
    for col in df.select_dtypes(include=['object']).columns:
        num_unique = df[col].nunique()
        num_total = len(df)
        if num_unique / num_total < 0.5:  # Less than 50% unique values
            df[col] = df[col].astype('category')
    
    return df

# Use categorical data
df['Category'] = df['Category'].astype('category')

# Sparse arrays for mostly-zero data
sparse_series = pd.arrays.SparseArray([0, 0, 1, 0, 0, 0, 2])
df['sparse_col'] = sparse_series
```

### Computational Optimization
```python
# Use vectorized operations instead of apply when possible
# SLOW
df['result'] = df.apply(lambda x: x['A'] + x['B'], axis=1)

# FAST
df['result'] = df['A'] + df['B']

# Use query() for complex boolean indexing
# SLOW
result = df[(df['A'] > 5) & (df['B'] < 10) & (df['C'] == 'value')]

# FAST
result = df.query('A > 5 and B < 10 and C == "value"')

# Use loc for setting values
# SLOW
for i, row in df.iterrows():
    if row['condition']:
        df.at[i, 'result'] = some_value

# FAST
mask = df['condition']
df.loc[mask, 'result'] = some_value

# Avoid chained indexing
# SLOW (and potentially incorrect)
df[df['A'] > 5]['B'] = 0

# FAST
df.loc[df['A'] > 5, 'B'] = 0

# Use eval() for complex expressions
# FAST for large DataFrames
df.eval('D = A + B * C')
df.eval('result = A + B', inplace=True)
```

### I/O Optimization
```python
# Use efficient file formats
# Parquet for structured data
df.to_parquet('data.parquet', compression='snappy')
df = pd.read_parquet('data.parquet')

# Feather for fastest I/O
df.to_feather('data.feather')
df = pd.read_feather('data.feather')

# HDF5 for time series
df.to_hdf('data.h5', key='df', mode='w')
df = pd.read_hdf('data.h5', key='df')

# Read large CSV in chunks
chunk_size = 10000
chunks = []
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    # Process chunk
    processed = chunk.groupby('category').sum()
    chunks.append(processed)
result = pd.concat(chunks)

# Specify dtypes when reading
dtypes = {
    'col1': 'int32',
    'col2': 'float32',
    'col3': 'category'
}
df = pd.read_csv('file.csv', dtype=dtypes)

# Use columns parameter to read only needed columns
df = pd.read_csv('file.csv', usecols=['col1', 'col2', 'col5'])
```

### Copy-on-Write Optimization
```python
# Enable Copy-on-Write mode (pandas 2.0+)
pd.options.mode.copy_on_write = True

# This reduces memory usage by avoiding unnecessary copies
df2 = df[['A', 'B']]  # No copy made until modification
df2.loc[0, 'A'] = 999  # Copy made only when modifying

# Check if DataFrame is a view or copy
def is_view(df_subset, df_original):
    return df_subset._mgr.refs is df_original._mgr.refs

# Use inplace operations when appropriate
df.drop('column', axis=1, inplace=True)  # Modifies original
df.fillna(0, inplace=True)               # Modifies original
```

---

## Advanced Techniques

### MultiIndex Operations
```python
# Create MultiIndex
arrays = [['A', 'A', 'B', 'B'], ['one', 'two', 'one', 'two']]
multi_idx = pd.MultiIndex.from_arrays(arrays, names=['first', 'second'])
df_multi = pd.DataFrame(np.random.randn(4, 2), index=multi_idx, columns=['X', 'Y'])

# Access MultiIndex levels
df_multi.index.levels[0]         # First level values
df_multi.index.get_level_values(0)  # First level for each row

# Select from MultiIndex
df_multi.loc['A']                # All rows with first level = 'A'
df_multi.loc[('A', 'one')]       # Specific combination
df_multi.loc[pd.IndexSlice['A', :], 'X']  # Advanced slicing

# Swap levels
df_multi.swaplevel(0, 1)

# Sort MultiIndex
df_multi.sort_index()            # Sort by all levels
df_multi.sort_index(level=1)     # Sort by specific level

# Stack/Unstack operations
df_stacked = df_multi.stack()    # Move column level to index
df_unstacked = df_stacked.unstack()  # Move index level to columns
```

### Custom Functions and Operations
```python
# Custom aggregation functions
def custom_agg(x):
    return pd.Series({
        'min': x.min(),
        'max': x.max(),
        'range': x.max() - x.min(),
        'q75': x.quantile(0.75)
    })

df.groupby('category').apply(custom_agg)

# Rolling custom functions
def rolling_sharpe(returns, window=30):
    return returns.rolling(window).apply(
        lambda x: x.mean() / x.std() * np.sqrt(252)
    )

# Custom window functions
from pandas.api.indexers import BaseIndexer

class CustomIndexer(BaseIndexer):
    def get_window_bounds(self, num_values, min_periods, center, closed):
        # Custom window logic
        pass

# Pipe for method chaining with custom functions
def remove_outliers(df, column, threshold=3):
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    return df[z_scores <= threshold]

def add_features(df):
    return df.assign(
        feature1=df['col1'] * df['col2'],
        feature2=df['col1'].rolling(5).mean()
    )

result = (df
    .pipe(remove_outliers, 'value')
    .pipe(add_features)
    .groupby('category')
    .mean()
)
```

### Performance Monitoring
```python
# Timing operations
import time

def time_operation(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    print(f"Operation took {end - start:.4f} seconds")
    return result

# Memory profiling
import psutil
import gc

def memory_usage():
    return psutil.Process().memory_info().rss / 1024 / 1024  # MB

def profile_memory(func, *args, **kwargs):
    gc.collect()
    start_memory = memory_usage()
    result = func(*args, **kwargs)
    end_memory = memory_usage()
    print(f"Memory usage increased by {end_memory - start_memory:.2f} MB")
    return result

# Pandas performance warnings
pd.options.mode.chained_assignment = 'warn'  # Warn about chained assignment
pd.options.mode.use_inf_as_na = True         # Treat inf as NaN
```

### Data Validation
```python
# Schema validation
def validate_schema(df, expected_dtypes):
    for col, dtype in expected_dtypes.items():
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
        if df[col].dtype != dtype:
            raise ValueError(f"Column {col} has dtype {df[col].dtype}, expected {dtype}")
    return True

# Data quality checks
def data_quality_report(df):
    report = {}
    
    for col in df.columns:
        col_report = {
            'dtype': df[col].dtype,
            'missing_count': df[col].isnull().sum(),
            'missing_percent': (df[col].isnull().sum() / len(df)) * 100,
            'unique_count': df[col].nunique(),
            'unique_percent': (df[col].nunique() / len(df)) * 100
        }
        
        if pd.api.types.is_numeric_dtype(df[col]):
            col_report.update({
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'outliers': len(df[col][(np.abs(df[col] - df[col].mean()) > 3 * df[col].std())])
            })
        
        report[col] = col_report
    
    return pd.DataFrame(report).T

# Constraint validation
def validate_constraints(df):
    constraints = []
    
    # Check for negative values in positive-only columns
    positive_cols = ['age', 'salary', 'price']
    for col in positive_cols:
        if col in df.columns:
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                constraints.append(f"{col} has {negative_count} negative values")
    
    # Check date ranges
    if 'date' in df.columns:
        min_date = df['date'].min()
        max_date = df['date'].max()
        if min_date < pd.to_datetime('1900-01-01'):
            constraints.append(f"Date column has values before 1900: {min_date}")
        if max_date > pd.to_datetime('2030-01-01'):
            constraints.append(f"Date column has future values: {max_date}")
    
    return constraints
```

---

## Pro Tips & Best Practices

### Code Style & Conventions
```python
# 1. Use method chaining for readability
result = (df
    .dropna()
    .query('age > 18')
    .groupby('category')
    .agg({'sales': 'sum', 'quantity': 'mean'})
    .reset_index()
    .sort_values('sales', ascending=False)
)

# 2. Prefer vectorized operations
# Good
df['result'] = np.where(df['value'] > 0, df['value'] * 2, 0)

# Avoid
df['result'] = df['value'].apply(lambda x: x * 2 if x > 0 else 0)

# 3. Use descriptive variable names
sales_by_region = df.groupby('region')['sales'].sum()  # Good
x = df.groupby('region')['sales'].sum()                # Avoid

# 4. Handle missing data explicitly
df = df.dropna(subset=['critical_column'])
df['optional_column'] = df['optional_column'].fillna('Unknown')

# 5. Set appropriate data types early
df['category'] = df['category'].astype('category')
df['date'] = pd.to_datetime(df['date'])
```

### Common Gotchas and Solutions
```python
# 1. Chained assignment warning
# Wrong
df[df['A'] > 5]['B'] = 0  # SettingWithCopyWarning

# Correct
df.loc[df['A'] > 5, 'B'] = 0

# 2. Index alignment in operations
df1 = pd.DataFrame({'A': [1, 2, 3]}, index=[0, 1, 2])
df2 = pd.DataFrame({'A': [4, 5, 6]}, index=[1, 2, 3])
result = df1 + df2  # Aligned by index, not position

# 3. Copy vs view
df_slice = df[['A', 'B']]  # Could be view or copy
df_copy = df[['A', 'B']].copy()  # Explicit copy

# 4. Boolean indexing with NaNs
# NaN values are considered False in boolean operations
mask = df['column'] > 5  # NaN values become False
df_filtered = df[mask]   # Excludes NaN rows

# Include NaN explicitly if needed
mask_with_nan = (df['column'] > 5) | df['column'].isna()

# 5. String operations on mixed types
# Ensure column is string type before string operations
df['text_col'] = df['text_col'].astype(str)
df['text_col'].str.upper()
```

### Debugging Tips
```python
# 1. Inspect data at each step
def debug_dataframe(df, name="DataFrame"):
    print(f"\n{name} Info:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Index: {df.index}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())

# 2. Check operation results
original_length = len(df)
df_filtered = df[df['age'] > 18]
print(f"Filtered from {original_length} to {len(df_filtered)} rows")

# 3. Validate merge results
left_shape = df1.shape[0]
right_shape = df2.shape[0]
merged = pd.merge(df1, df2, on='key')
print(f"Merged {left_shape} and {right_shape} rows to {merged.shape[0]} rows")

# 4. Use assert statements for data validation
assert df['age'].min() >= 0, "Age cannot be negative"
assert df['date'].notna().all(), "Date column has missing values"
assert len(df) > 0, "DataFrame is empty"

# 5. Profile slow operations
import cProfile
cProfile.run('df.groupby("category").sum()')
```

---

**📚 Additional Resources:**
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html)
- [10 Minutes to Pandas](https://pandas.pydata.org/docs/user_guide/10min.html)
- [Pandas Cookbook](https://pandas.pydata.org/docs/user_guide/cookbook.html)

**🔗 Related Cheatsheets:**
- [Python Cheatsheet](PYTHON_CHEATSHEET.md)
- [NumPy Cheatsheet](NUMPY_CHEATSHEET.md)
- [PySpark Cheatsheet](PYSPARK_CHEATSHEET.md)
- [PyTorch Cheatsheet](PYTORCH_CHEATSHEET.md)

---
*Last Updated: January 8, 2026*
