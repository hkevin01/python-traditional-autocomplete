"""
Test Data Science Packages
=========================
Test NumPy, Pandas, Matplotlib, PySpark functionality.

Performance benchmarks included to verify claimed speedups.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
import time


class TestNumPy:
    """Test NumPy functionality and performance."""
    
    def test_numpy_array_creation(self):
        """Test NumPy array creation."""
        arr = np.array([1, 2, 3, 4, 5])
        assert arr.shape == (5,)
        assert arr.dtype in [np.int64, np.int32]
    
    def test_numpy_operations(self):
        """Test NumPy mathematical operations."""
        arr = np.array([1, 2, 3, 4, 5])
        
        assert np.sum(arr) == 15
        assert np.mean(arr) == 3.0
        assert np.std(arr) > 0
    
    def test_numpy_broadcasting(self):
        """Test NumPy broadcasting."""
        a = np.array([[1, 2, 3], [4, 5, 6]])
        b = np.array([10, 20, 30])
        
        result = a + b
        expected = np.array([[11, 22, 33], [14, 25, 36]])
        
        assert np.array_equal(result, expected)
    
    def test_numpy_performance(self):
        """Test NumPy performance vs pure Python."""
        size = 100000
        
        # Pure Python
        python_list = list(range(size))
        start = time.perf_counter()
        result_py = [x * 2 for x in python_list]
        python_time = time.perf_counter() - start
        
        # NumPy
        numpy_array = np.arange(size)
        start = time.perf_counter()
        result_np = numpy_array * 2
        numpy_time = time.perf_counter() - start
        
        # NumPy should be significantly faster
        speedup = python_time / numpy_time
        assert speedup > 5, f"NumPy should be >5x faster, got {speedup:.1f}x"


class TestPandas:
    """Test Pandas functionality."""
    
    def test_pandas_dataframe_creation(self):
        """Test Pandas DataFrame creation."""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': ['x', 'y', 'z']
        })
        
        assert df.shape == (3, 3)
        assert list(df.columns) == ['A', 'B', 'C']
    
    def test_pandas_operations(self):
        """Test Pandas data operations."""
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })
        
        # Test filtering
        filtered = df[df['A'] > 2]
        assert len(filtered) == 3
        
        # Test aggregation
        assert df['A'].sum() == 15
        assert df['B'].mean() == 30.0
    
    def test_pandas_groupby(self):
        """Test Pandas groupby operations."""
        df = pd.DataFrame({
            'category': ['A', 'B', 'A', 'B', 'A'],
            'value': [10, 20, 15, 25, 12]
        })
        
        grouped = df.groupby('category')['value'].sum()
        
        assert grouped['A'] == 37
        assert grouped['B'] == 45
    
    def test_pandas_merge(self):
        """Test Pandas merge operations."""
        df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'value1': [1, 2, 3]})
        df2 = pd.DataFrame({'key': ['A', 'B', 'D'], 'value2': [4, 5, 6]})
        
        merged = pd.merge(df1, df2, on='key', how='inner')
        
        assert len(merged) == 2
        assert list(merged['key']) == ['A', 'B']
    
    def test_pandas_performance(self):
        """Test Pandas performance vs pure Python."""
        size = 10000
        data = {'A': list(range(size)), 'B': list(range(size, size * 2))}
        
        # Pure Python
        start = time.perf_counter()
        result_py = [data['A'][i] + data['B'][i] for i in range(size)]
        python_time = time.perf_counter() - start
        
        # Pandas
        df = pd.DataFrame(data)
        start = time.perf_counter()
        result_pd = df['A'] + df['B']
        pandas_time = time.perf_counter() - start
        
        speedup = python_time / pandas_time
        assert speedup > 2, f"Pandas should be >2x faster, got {speedup:.1f}x"


class TestMatplotlib:
    """Test Matplotlib functionality."""
    
    def test_matplotlib_figure_creation(self):
        """Test creating a matplotlib figure."""
        fig, ax = plt.subplots()
        assert fig is not None
        assert ax is not None
        plt.close(fig)
    
    def test_matplotlib_line_plot(self):
        """Test creating a line plot."""
        fig, ax = plt.subplots()
        
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        
        line, = ax.plot(x, y)
        assert line is not None
        assert len(ax.lines) == 1
        
        plt.close(fig)
    
    def test_matplotlib_scatter_plot(self):
        """Test creating a scatter plot."""
        fig, ax = plt.subplots()
        
        x = np.random.rand(50)
        y = np.random.rand(50)
        
        scatter = ax.scatter(x, y)
        assert scatter is not None
        
        plt.close(fig)
    
    def test_matplotlib_histogram(self):
        """Test creating a histogram."""
        fig, ax = plt.subplots()
        
        data = np.random.normal(0, 1, 1000)
        n, bins, patches = ax.hist(data, bins=30)
        
        assert len(patches) == 30
        assert sum(n) == 1000
        
        plt.close(fig)
    
    def test_matplotlib_customization(self):
        """Test plot customization."""
        fig, ax = plt.subplots()
        
        ax.plot([1, 2, 3], [1, 4, 9])
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_title('Test Plot')
        
        assert ax.get_xlabel() == 'X Label'
        assert ax.get_ylabel() == 'Y Label'
        assert ax.get_title() == 'Test Plot'
        
        plt.close(fig)


class TestPySpark:
    """Test PySpark functionality."""
    
    @pytest.fixture(scope="class")
    def spark(self):
        """Create a SparkSession for testing."""
        spark = SparkSession.builder \
            .appName("pytest") \
            .master("local[2]") \
            .config("spark.driver.memory", "1g") \
            .getOrCreate()
        
        yield spark
        
        spark.stop()
    
    def test_pyspark_dataframe_creation(self, spark):
        """Test PySpark DataFrame creation."""
        data = [("Alice", 34), ("Bob", 45), ("Charlie", 28)]
        columns = ["name", "age"]
        
        df = spark.createDataFrame(data, columns)
        
        assert df.count() == 3
        assert df.columns == columns
    
    def test_pyspark_operations(self, spark):
        """Test PySpark data operations."""
        data = [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)]
        df = spark.createDataFrame(data, ["id", "value"])
        
        # Test filtering
        filtered = df.filter(df.id > 2)
        assert filtered.count() == 3
        
        # Test aggregation
        total = df.agg({"value": "sum"}).collect()[0][0]
        assert total == 150
    
    def test_pyspark_sql(self, spark):
        """Test PySpark SQL operations."""
        data = [("Alice", 34), ("Bob", 45), ("Charlie", 28)]
        df = spark.createDataFrame(data, ["name", "age"])
        df.createOrReplaceTempView("people")
        
        result = spark.sql("SELECT name FROM people WHERE age > 30")
        names = [row.name for row in result.collect()]
        
        assert "Alice" in names
        assert "Bob" in names
        assert "Charlie" not in names
    
    def test_pyspark_groupby(self, spark):
        """Test PySpark groupBy operations."""
        data = [
            ("A", 10), ("B", 20), ("A", 15),
            ("B", 25), ("A", 12)
        ]
        df = spark.createDataFrame(data, ["category", "value"])
        
        grouped = df.groupBy("category").sum("value")
        results = {row.category: row['sum(value)'] for row in grouped.collect()}
        
        assert results["A"] == 37
        assert results["B"] == 45
