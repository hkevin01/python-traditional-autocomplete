# PySpark 3.5.4 Comprehensive Cheatsheet

**Level:** SparkNotes/Unbound Quality Reference  
**Date:** January 8, 2026  
**PySpark Version:** 3.5.4+  
**Performance:** 100x speedup for big data vs single-machine processing

## Table of Contents

- [Quick Start & Core Concepts](#quick-start--core-concepts)
- [Spark Session & Context](#spark-session--context)
- [RDD Operations](#rdd-operations)
- [DataFrame API](#dataframe-api)
- [SQL Operations](#sql-operations)
- [Data Sources](#data-sources)
- [Transformations & Actions](#transformations--actions)
- [Aggregations & Window Functions](#aggregations--window-functions)
- [Machine Learning (MLlib)](#machine-learning-mllib)
- [Streaming](#streaming)
- [Performance Optimization](#performance-optimization)
- [Advanced Techniques](#advanced-techniques)

---

## Quick Start & Core Concepts

### Import Conventions
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import pyspark.sql.functions as F
from pyspark import SparkContext, SparkConf
```

### Core Concepts
```python
# Spark Context - Entry point for RDD operations
sc = SparkContext()

# Spark Session - Entry point for DataFrame operations (Spark 2.0+)
spark = SparkSession.builder \
    .appName("MyApp") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

# Key abstractions:
# RDD (Resilient Distributed Dataset) - Low-level API
# DataFrame - High-level API with SQL-like operations
# Dataset - Typed version of DataFrame (Scala/Java)

# Lazy evaluation - transformations are not executed until an action
# Distributed computing across cluster nodes
# Fault tolerance through lineage tracking
```

### Basic Setup
```python
# Local development
spark = SparkSession.builder \
    .appName("LocalDev") \
    .master("local[*]") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .getOrCreate()

# Cluster deployment
spark = SparkSession.builder \
    .appName("ClusterApp") \
    .master("yarn") \
    .config("spark.executor.memory", "4g") \
    .config("spark.executor.cores", "4") \
    .config("spark.dynamicAllocation.enabled", "true") \
    .getOrCreate()

# Configure logging
spark.sparkContext.setLogLevel("WARN")

# Stop session when done
spark.stop()
```

---

## Spark Session & Context

### SparkSession Configuration
```python
# Comprehensive configuration
spark = SparkSession.builder \
    .appName("ComprehensiveApp") \
    .master("local[*]") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .config("spark.sql.adaptive.skewJoin.enabled", "true") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.sql.execution.arrow.maxRecordsPerBatch", "10000") \
    .getOrCreate()

# Memory configuration
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
spark.conf.set("spark.sql.adaptive.advisoryPartitionSizeInBytes", "128MB")

# Check current configuration
spark.conf.get("spark.sql.adaptive.enabled")
for key, value in spark.conf.getAll():
    print(f"{key}: {value}")

# Runtime configuration changes
spark.conf.set("spark.sql.shuffle.partitions", "200")
```

### SparkContext Operations
```python
# Access SparkContext from SparkSession
sc = spark.sparkContext

# Application info
print(f"Application ID: {sc.applicationId}")
print(f"Application Name: {sc.appName}")
print(f"Master URL: {sc.master}")
print(f"Spark Version: {sc.version}")

# Parallelism and partitioning
print(f"Default parallelism: {sc.defaultParallelism}")
print(f"Default min partitions: {sc.defaultMinPartitions}")

# Broadcast variables
broadcast_var = sc.broadcast([1, 2, 3, 4, 5])
print(broadcast_var.value)

# Accumulators
accumulator = sc.accumulator(0)
def increment_counter(x):
    accumulator.add(1)
    return x

rdd = sc.parallelize([1, 2, 3, 4, 5])
rdd.map(increment_counter).collect()
print(f"Accumulator value: {accumulator.value}")
```

---

## RDD Operations

### RDD Creation
```python
# From Python collections
rdd1 = sc.parallelize([1, 2, 3, 4, 5])
rdd2 = sc.parallelize([("Alice", 25), ("Bob", 30), ("Charlie", 35)])

# From external data sources
rdd3 = sc.textFile("file.txt")
rdd4 = sc.wholeTextFiles("directory/*.txt")  # Returns (filename, content) pairs

# Create empty RDD
empty_rdd = sc.emptyRDD()

# From existing RDD
filtered_rdd = rdd1.filter(lambda x: x > 2)

# Specify number of partitions
rdd_partitioned = sc.parallelize(range(1000), 4)  # 4 partitions
```

### RDD Transformations (Lazy)
```python
# Map operations
rdd_squared = rdd1.map(lambda x: x ** 2)
rdd_words = sc.textFile("file.txt").flatMap(lambda line: line.split())

# Filter operations
even_numbers = rdd1.filter(lambda x: x % 2 == 0)

# Distinct and sampling
unique_values = rdd1.distinct()
sample_data = rdd1.sample(False, 0.1)  # 10% sample without replacement

# Set operations
rdd_a = sc.parallelize([1, 2, 3, 4])
rdd_b = sc.parallelize([3, 4, 5, 6])
union_rdd = rdd_a.union(rdd_b)
intersection_rdd = rdd_a.intersection(rdd_b)
subtract_rdd = rdd_a.subtract(rdd_b)

# Key-Value operations
pairs_rdd = sc.parallelize([("a", 1), ("b", 2), ("a", 3)])
grouped = pairs_rdd.groupByKey()
reduced = pairs_rdd.reduceByKey(lambda x, y: x + y)
sorted_by_key = pairs_rdd.sortByKey()

# Join operations
rdd_left = sc.parallelize([("a", 1), ("b", 2)])
rdd_right = sc.parallelize([("a", "x"), ("b", "y"), ("c", "z")])
joined = rdd_left.join(rdd_right)          # Inner join
left_joined = rdd_left.leftOuterJoin(rdd_right)
right_joined = rdd_left.rightOuterJoin(rdd_right)
full_joined = rdd_left.fullOuterJoin(rdd_right)

# Repartitioning
repartitioned = rdd1.repartition(4)
coalesced = rdd1.coalesce(2)  # Reduce partitions (no shuffling)
```

### RDD Actions (Eager)
```python
# Collect operations
all_data = rdd1.collect()                    # Bring all data to driver
first_element = rdd1.first()                 # First element
sample_elements = rdd1.take(5)               # First 5 elements
top_elements = rdd1.top(3)                   # Top 3 elements (desc order)

# Aggregation actions
total = rdd1.reduce(lambda x, y: x + y)      # Sum all elements
count = rdd1.count()                         # Number of elements
count_by_value = rdd1.countByValue()         # Count of each unique value
count_by_key = pairs_rdd.countByKey()        # Count by key

# Statistical operations
stats = rdd1.stats()                         # Summary statistics
print(f"Mean: {stats.mean()}, Stdev: {stats.stdev()}")

# Save operations
rdd1.saveAsTextFile("output/rdd_text")
pairs_rdd.saveAsSequenceFile("output/rdd_seq")

# Foreach (side effects)
rdd1.foreach(lambda x: print(x))             # Print each element
```

### RDD Persistence and Caching
```python
# Cache in memory
cached_rdd = rdd1.cache()  # Same as persist(StorageLevel.MEMORY_ONLY)

# Different storage levels
from pyspark import StorageLevel

memory_only = rdd1.persist(StorageLevel.MEMORY_ONLY)
memory_and_disk = rdd1.persist(StorageLevel.MEMORY_AND_DISK)
memory_ser = rdd1.persist(StorageLevel.MEMORY_ONLY_SER)
disk_only = rdd1.persist(StorageLevel.DISK_ONLY)

# Check if RDD is cached
print(f"Is cached: {rdd1.is_cached}")

# Unpersist
rdd1.unpersist()

# Custom partitioner for key-value RDDs
from pyspark import HashPartitioner, RangePartitioner

hash_partitioned = pairs_rdd.partitionBy(4, HashPartitioner(4))
range_partitioned = pairs_rdd.partitionBy(4, RangePartitioner(4, pairs_rdd))
```

---

## DataFrame API

### DataFrame Creation
```python
# From RDD
rdd = sc.parallelize([("Alice", 25), ("Bob", 30), ("Charlie", 35)])
df = rdd.toDF(["name", "age"])

# From Python data structures
data = [("Alice", 25), ("Bob", 30), ("Charlie", 35)]
df = spark.createDataFrame(data, ["name", "age"])

# With explicit schema
schema = StructType([
    StructField("name", StringType(), True),
    StructField("age", IntegerType(), True)
])
df = spark.createDataFrame(data, schema)

# From pandas DataFrame (requires PyArrow)
import pandas as pd
pandas_df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
spark_df = spark.createDataFrame(pandas_df)

# Empty DataFrame with schema
empty_df = spark.createDataFrame([], schema)

# Range DataFrame
range_df = spark.range(1000)  # 0 to 999
range_df_custom = spark.range(10, 100, 2)  # 10 to 98, step 2
```

### Basic DataFrame Operations
```python
# Display operations
df.show()                        # Show first 20 rows
df.show(5, truncate=False)       # Show 5 rows, don't truncate
df.display()                     # Notebook-friendly display
df.head(3)                       # First 3 rows as Row objects

# Schema and structure
df.printSchema()                 # Print schema tree
df.dtypes                        # List of (column_name, data_type)
df.columns                       # Column names
df.count()                       # Number of rows
len(df.columns)                  # Number of columns

# Summary statistics
df.describe().show()             # Statistical summary
df.summary().show()              # Extended summary (includes quartiles)

# Column operations
df.select("name").show()                    # Select single column
df.select("name", "age").show()             # Select multiple columns
df.select(F.col("age") + 1).show()         # Expression on column
df.selectExpr("name", "age * 2 as double_age").show()  # SQL expressions

# Add/rename/drop columns
df_with_new = df.withColumn("age_plus_10", F.col("age") + 10)
df_renamed = df.withColumnRenamed("age", "years")
df_dropped = df.drop("age")

# Filter/where operations
df.filter(F.col("age") > 25).show()
df.where("age > 25").show()              # SQL-style
df.filter((F.col("age") > 25) & (F.col("name").like("A%"))).show()
```

### DataFrame Transformations
```python
# Sorting
df.orderBy("age").show()                     # Ascending
df.orderBy(F.col("age").desc()).show()       # Descending
df.sort("name", "age").show()                # Multiple columns

# Aggregations
df.groupBy("department").count().show()
df.groupBy("department").agg(
    F.avg("salary").alias("avg_salary"),
    F.max("age").alias("max_age"),
    F.min("age").alias("min_age")
).show()

# Joins
df1 = spark.createDataFrame([("Alice", "eng"), ("Bob", "sales")], ["name", "dept"])
df2 = spark.createDataFrame([("Alice", 100000), ("Bob", 80000)], ["name", "salary"])

# Different join types
inner_join = df1.join(df2, "name")                    # Inner join
left_join = df1.join(df2, "name", "left")             # Left join
right_join = df1.join(df2, "name", "right")           # Right join
full_join = df1.join(df2, "name", "outer")            # Full outer join

# Complex join conditions
complex_join = df1.join(df2, (df1.name == df2.name) & (df1.dept == "eng"))

# Union operations
df_union = df1.union(df2)                             # Union (must have same schema)
df_union_by_name = df1.unionByName(df2)               # Union by column names

# Window functions
from pyspark.sql.window import Window

windowSpec = Window.partitionBy("department").orderBy("salary")
df.withColumn("rank", F.rank().over(windowSpec)).show()
df.withColumn("row_number", F.row_number().over(windowSpec)).show()
```

### Advanced DataFrame Operations
```python
# Pivot operations
pivot_df = df.groupBy("year").pivot("quarter").sum("sales")

# Unpivot (stack)
unpivot_df = df.select("id", F.stack(3, 
    F.lit("Q1"), F.col("Q1_sales"),
    F.lit("Q2"), F.col("Q2_sales"), 
    F.lit("Q3"), F.col("Q3_sales")
).alias("quarter", "sales"))

# Sampling
sample_df = df.sample(0.1)                   # 10% sample
stratified_sample = df.sampleBy("category", {"A": 0.1, "B": 0.2})

# Distinct and duplicate handling
unique_df = df.distinct()
dedupe_df = df.dropDuplicates()
dedupe_subset = df.dropDuplicates(["name"])

# Missing data handling
df.na.drop().show()                          # Drop rows with any null
df.na.drop(subset=["age"]).show()            # Drop if specific columns are null
df.na.fill({"age": 0, "name": "Unknown"}).show()  # Fill nulls with values
df.fillna(0).show()                          # Fill all nulls with 0

# Replace values
df.na.replace(["Alice", "Bob"], ["A", "B"], "name").show()
df.replace(25, 26, "age").show()

# Column operations with when/otherwise
df.withColumn("age_category",
    F.when(F.col("age") < 30, "Young")
     .when(F.col("age") < 60, "Middle")
     .otherwise("Senior")
).show()
```

---

## SQL Operations

### SQL Registration and Queries
```python
# Register DataFrame as temporary view
df.createOrReplaceTempView("people")

# SQL queries
result = spark.sql("SELECT * FROM people WHERE age > 25")
result.show()

# Complex SQL queries
spark.sql("""
    SELECT department, 
           AVG(salary) as avg_salary,
           COUNT(*) as employee_count
    FROM employees
    GROUP BY department
    HAVING AVG(salary) > 50000
    ORDER BY avg_salary DESC
""").show()

# Global temporary views (accessible across sessions)
df.createGlobalTempView("global_people")
spark.sql("SELECT * FROM global_temp.global_people").show()

# List tables and views
spark.catalog.listTables().show()
```

### Advanced SQL Operations
```python
# Window functions in SQL
spark.sql("""
    SELECT name, salary, department,
           ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) as rank,
           LAG(salary) OVER (PARTITION BY department ORDER BY salary) as prev_salary,
           salary - LAG(salary) OVER (PARTITION BY department ORDER BY salary) as salary_diff
    FROM employees
""").show()

# Common Table Expressions (CTEs)
spark.sql("""
    WITH high_performers AS (
        SELECT * FROM employees WHERE performance_score > 8
    ),
    department_stats AS (
        SELECT department, AVG(salary) as avg_salary FROM high_performers GROUP BY department
    )
    SELECT hp.*, ds.avg_salary as dept_avg_salary
    FROM high_performers hp
    JOIN department_stats ds ON hp.department = ds.department
""").show()

# Subqueries
spark.sql("""
    SELECT * FROM employees e1
    WHERE salary > (SELECT AVG(salary) FROM employees e2 WHERE e1.department = e2.department)
""").show()
```

---

## Data Sources

### File Formats
```python
# CSV files
df = spark.read.csv("file.csv", header=True, inferSchema=True)
df = spark.read.option("header", "true") \
               .option("inferSchema", "true") \
               .option("delimiter", ",") \
               .csv("file.csv")

# JSON files
df = spark.read.json("file.json")
df = spark.read.option("multiline", "true").json("file.json")

# Parquet files (columnar format, excellent for analytics)
df = spark.read.parquet("file.parquet")
df_partitioned = spark.read.parquet("partitioned_data/year=*/month=*")

# ORC files (optimized for Hive)
df = spark.read.orc("file.orc")

# Avro files
df = spark.read.format("avro").load("file.avro")

# Delta Lake (requires delta-spark package)
df = spark.read.format("delta").load("delta-table")

# Text files
df = spark.read.text("file.txt")  # Each line becomes a row
```

### Database Connections
```python
# JDBC connections
df = spark.read.format("jdbc") \
    .option("url", "jdbc:postgresql://localhost:5432/mydb") \
    .option("dbtable", "employees") \
    .option("user", "username") \
    .option("password", "password") \
    .option("driver", "org.postgresql.Driver") \
    .load()

# Read with custom SQL
df = spark.read.format("jdbc") \
    .option("url", "jdbc:mysql://localhost:3306/mydb") \
    .option("query", "SELECT * FROM employees WHERE department = 'engineering'") \
    .option("user", "username") \
    .option("password", "password") \
    .load()

# Partitioned reads for large tables
df = spark.read.format("jdbc") \
    .option("url", "jdbc:postgresql://localhost:5432/mydb") \
    .option("dbtable", "large_table") \
    .option("partitionColumn", "id") \
    .option("lowerBound", "1") \
    .option("upperBound", "1000000") \
    .option("numPartitions", "10") \
    .option("user", "username") \
    .option("password", "password") \
    .load()
```

### Write Operations
```python
# CSV output
df.write.csv("output/csv_data", header=True, mode="overwrite")

# Parquet output with partitioning
df.write.partitionBy("year", "month").parquet("output/partitioned_parquet")

# JSON output
df.write.mode("overwrite").json("output/json_data")

# Database writes
df.write.format("jdbc") \
    .option("url", "jdbc:postgresql://localhost:5432/mydb") \
    .option("dbtable", "output_table") \
    .option("user", "username") \
    .option("password", "password") \
    .mode("overwrite") \
    .save()

# Write modes
df.write.mode("overwrite").parquet("output")     # Overwrite existing data
df.write.mode("append").parquet("output")        # Append to existing data
df.write.mode("ignore").parquet("output")        # Ignore if exists
df.write.mode("error").parquet("output")         # Error if exists (default)

# Bucketing for performance
df.write.bucketBy(4, "user_id").saveAsTable("bucketed_table")

# Save as Hive table
df.write.saveAsTable("my_table")
df.write.mode("overwrite").saveAsTable("my_table")

# Repartition before writing for optimal file sizes
df.repartition(4).write.parquet("output/repartitioned")
df.coalesce(1).write.csv("output/single_file")
```

---

## Transformations & Actions

### Built-in Functions
```python
# String functions
from pyspark.sql.functions import *

df.withColumn("upper_name", upper(col("name"))).show()
df.withColumn("name_length", length(col("name"))).show()
df.withColumn("substr", substring(col("name"), 1, 3)).show()
df.withColumn("concat", concat(col("name"), lit(" - "), col("department"))).show()
df.withColumn("split", split(col("full_name"), " ")).show()
df.filter(col("name").like("A%")).show()          # Like pattern
df.filter(col("name").rlike("^A.*")).show()       # Regex pattern

# Numeric functions
df.withColumn("rounded", round(col("salary"), -3)).show()
df.withColumn("ceiling", ceil(col("value"))).show()
df.withColumn("floored", floor(col("value"))).show()
df.withColumn("absolute", abs(col("value"))).show()
df.withColumn("power", pow(col("base"), col("exponent"))).show()

# Date/Time functions
from datetime import date, datetime
df.withColumn("current_date", current_date()).show()
df.withColumn("current_timestamp", current_timestamp()).show()
df.withColumn("year", year(col("date_column"))).show()
df.withColumn("month", month(col("date_column"))).show()
df.withColumn("dayofweek", dayofweek(col("date_column"))).show()
df.withColumn("date_add", date_add(col("date_column"), 30)).show()
df.withColumn("months_between", months_between(col("end_date"), col("start_date"))).show()

# Array functions
df.withColumn("array_size", size(col("array_column"))).show()
df.withColumn("array_contains", array_contains(col("array_column"), "value")).show()
df.withColumn("exploded", explode(col("array_column"))).show()
df.withColumn("sorted_array", sort_array(col("array_column"))).show()

# Map functions
df.withColumn("map_keys", map_keys(col("map_column"))).show()
df.withColumn("map_values", map_values(col("map_column"))).show()
df.withColumn("map_value", col("map_column")["key"]).show()

# Null handling functions
df.withColumn("coalesced", coalesce(col("col1"), col("col2"), lit("default"))).show()
df.withColumn("is_null", isnull(col("column"))).show()
df.withColumn("is_not_null", isnan(col("column"))).show()
```

### User Defined Functions (UDFs)
```python
from pyspark.sql.types import *

# Python UDF
def square(x):
    return x ** 2 if x is not None else None

square_udf = udf(square, IntegerType())
df.withColumn("squared", square_udf(col("age"))).show()

# UDF with explicit return type
@udf(returnType=StringType())
def categorize_age(age):
    if age is None:
        return "Unknown"
    elif age < 30:
        return "Young"
    elif age < 60:
        return "Middle"
    else:
        return "Senior"

df.withColumn("age_category", categorize_age(col("age"))).show()

# Vectorized UDF (Pandas UDF) - much faster
from pyspark.sql.functions import pandas_udf
import pandas as pd

@pandas_udf(returnType=IntegerType())
def vectorized_square(series: pd.Series) -> pd.Series:
    return series ** 2

df.withColumn("vectorized_squared", vectorized_square(col("age"))).show()

# Grouped map UDF
@pandas_udf(returnType=StructType([
    StructField("department", StringType()),
    StructField("avg_salary", DoubleType())
]))
def compute_dept_avg(pdf: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([[pdf["department"].iloc[0], pdf["salary"].mean()]])

df.groupBy("department").apply(compute_dept_avg).show()
```

### Custom Transformations
```python
# Custom transformation functions
def add_derived_columns(df):
    return df.withColumn("full_name", concat(col("first_name"), lit(" "), col("last_name"))) \
             .withColumn("age_group", when(col("age") < 30, "Young").otherwise("Mature"))

def standardize_names(df):
    return df.withColumn("name", upper(trim(col("name"))))

def remove_outliers(df, column, threshold=3):
    stats = df.select(mean(column).alias("mean"), stddev(column).alias("std")).collect()[0]
    mean_val = stats["mean"]
    std_val = stats["std"]
    
    return df.filter(
        (col(column) >= (mean_val - threshold * std_val)) & 
        (col(column) <= (mean_val + threshold * std_val))
    )

# Apply custom transformations
df_transformed = df.transform(add_derived_columns) \
                   .transform(standardize_names) \
                   .transform(lambda x: remove_outliers(x, "salary"))
```

---

## Aggregations & Window Functions

### Aggregation Operations
```python
# Basic aggregations
df.agg(
    count("*").alias("total_count"),
    sum("salary").alias("total_salary"),
    avg("salary").alias("avg_salary"),
    min("age").alias("min_age"),
    max("age").alias("max_age"),
    stddev("salary").alias("salary_stddev")
).show()

# Group by aggregations
df.groupBy("department") \
  .agg(
      count("*").alias("employee_count"),
      avg("salary").alias("avg_salary"),
      collect_list("name").alias("employee_names"),
      collect_set("skill").alias("unique_skills")
  ).show()

# Multiple grouping columns
df.groupBy("department", "level") \
  .agg(avg("salary").alias("avg_salary")) \
  .orderBy("department", "level") \
  .show()

# Conditional aggregations
df.groupBy("department") \
  .agg(
      sum(when(col("gender") == "M", 1).otherwise(0)).alias("male_count"),
      sum(when(col("gender") == "F", 1).otherwise(0)).alias("female_count"),
      avg(when(col("performance") > 8, col("salary"))).alias("high_performer_avg_salary")
  ).show()

# Approximate aggregations (for large datasets)
df.agg(
    approx_count_distinct("customer_id").alias("approx_unique_customers"),
    expr("percentile_approx(salary, 0.5)").alias("median_salary")
).show()
```

### Window Functions
```python
from pyspark.sql.window import Window

# Define window specifications
dept_window = Window.partitionBy("department").orderBy("salary")
unbounded_window = Window.partitionBy("department") \
                         .orderBy("salary") \
                         .rowsBetween(Window.unboundedPreceding, Window.currentRow)
range_window = Window.partitionBy("department") \
                     .orderBy("salary") \
                     .rangeBetween(-1000, 1000)

# Ranking functions
df.withColumn("rank", rank().over(dept_window)) \
  .withColumn("dense_rank", dense_rank().over(dept_window)) \
  .withColumn("row_number", row_number().over(dept_window)) \
  .withColumn("percent_rank", percent_rank().over(dept_window)) \
  .show()

# Analytical functions
df.withColumn("running_total", sum("salary").over(unbounded_window)) \
  .withColumn("avg_dept_salary", avg("salary").over(Window.partitionBy("department"))) \
  .withColumn("salary_diff_from_avg", col("salary") - avg("salary").over(Window.partitionBy("department"))) \
  .show()

# Lead and lag functions
df.withColumn("next_salary", lead("salary", 1).over(dept_window)) \
  .withColumn("prev_salary", lag("salary", 1).over(dept_window)) \
  .withColumn("salary_change", col("salary") - lag("salary", 1).over(dept_window)) \
  .show()

# First and last value
df.withColumn("dept_min_salary", first("salary").over(dept_window)) \
  .withColumn("dept_max_salary", last("salary").over(dept_window)) \
  .show()

# N-tile functions
df.withColumn("quartile", ntile(4).over(dept_window)) \
  .withColumn("decile", ntile(10).over(dept_window)) \
  .show()

# Custom window aggregations
df.withColumn("salary_range_sum", sum("salary").over(range_window)) \
  .withColumn("nearby_avg", avg("salary").over(range_window)) \
  .show()
```

### Pivot and Rollup Operations
```python
# Pivot operations
pivot_df = df.groupBy("year") \
             .pivot("quarter") \
             .sum("sales")

# Pivot with explicit values (better performance)
pivot_df = df.groupBy("year") \
             .pivot("quarter", ["Q1", "Q2", "Q3", "Q4"]) \
             .sum("sales")

# Rollup (hierarchical aggregations)
rollup_df = df.rollup("region", "country", "city") \
              .agg(sum("sales").alias("total_sales")) \
              .orderBy("region", "country", "city")

# Cube (all possible combinations)
cube_df = df.cube("product", "region") \
            .agg(sum("sales").alias("total_sales"))

# Grouping sets (custom combinations)
grouping_sets_df = df.groupBy("year", "quarter") \
                     .agg(
                         sum("sales").alias("total_sales"),
                         grouping("year").alias("year_grouping"),
                         grouping("quarter").alias("quarter_grouping")
                     )
```

---

## Machine Learning (MLlib)

### Data Preparation
```python
from pyspark.ml.feature import *
from pyspark.ml.regression import LinearRegression
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import RegressionEvaluator, BinaryClassificationEvaluator

# String indexing for categorical variables
stringIndexer = StringIndexer(inputCol="category", outputCol="categoryIndex")
indexed_df = stringIndexer.fit(df).transform(df)

# One-hot encoding
oneHotEncoder = OneHotEncoder(inputCols=["categoryIndex"], outputCols=["categoryVec"])
encoded_df = oneHotEncoder.fit(indexed_df).transform(indexed_df)

# Vector assembler (combine features)
assembler = VectorAssembler(
    inputCols=["age", "salary", "categoryVec"],
    outputCol="features"
)
feature_df = assembler.transform(encoded_df)

# Feature scaling
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
scaler_model = scaler.fit(feature_df)
scaled_df = scaler_model.transform(feature_df)

# Train-test split
train_df, test_df = scaled_df.randomSplit([0.8, 0.2], seed=42)
```

### Regression
```python
# Linear regression
lr = LinearRegression(
    featuresCol="scaledFeatures",
    labelCol="target",
    maxIter=100,
    regParam=0.01
)

# Train model
lr_model = lr.fit(train_df)

# Make predictions
predictions = lr_model.transform(test_df)

# Evaluate model
evaluator = RegressionEvaluator(
    labelCol="target",
    predictionCol="prediction",
    metricName="rmse"
)
rmse = evaluator.evaluate(predictions)
print(f"RMSE: {rmse}")

# Model summary
print(f"Coefficients: {lr_model.coefficients}")
print(f"Intercept: {lr_model.intercept}")
print(f"R-squared: {lr_model.summary.r2}")
```

### Classification
```python
# Logistic regression
log_reg = LogisticRegression(
    featuresCol="scaledFeatures",
    labelCol="label",
    maxIter=100,
    regParam=0.01
)

# Train model
log_reg_model = log_reg.fit(train_df)

# Predictions
predictions = log_reg_model.transform(test_df)

# Evaluation
evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)
auc = evaluator.evaluate(predictions)
print(f"AUC: {auc}")

# Random Forest
from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(
    featuresCol="scaledFeatures",
    labelCol="label",
    numTrees=100,
    maxDepth=5
)
rf_model = rf.fit(train_df)
rf_predictions = rf_model.transform(test_df)

# Feature importance
feature_importance = rf_model.featureImportances
print(f"Feature Importance: {feature_importance}")
```

### Clustering
```python
# K-Means clustering
kmeans = KMeans(
    featuresCol="scaledFeatures",
    k=3,
    maxIter=100
)

# Train model
kmeans_model = kmeans.fit(scaled_df)

# Predictions (cluster assignments)
clustered_df = kmeans_model.transform(scaled_df)

# Cluster centers
centers = kmeans_model.clusterCenters()
print(f"Cluster Centers: {centers}")

# Within Set Sum of Squared Errors
wssse = kmeans_model.computeCost(scaled_df)
print(f"WSSSE: {wssse}")

# Silhouette analysis
from pyspark.ml.evaluation import ClusteringEvaluator
evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(clustered_df)
print(f"Silhouette Score: {silhouette}")
```

### ML Pipelines
```python
from pyspark.ml import Pipeline

# Create pipeline stages
stages = [
    stringIndexer,
    oneHotEncoder,
    assembler,
    scaler,
    lr
]

# Create and fit pipeline
pipeline = Pipeline(stages=stages)
pipeline_model = pipeline.fit(train_df)

# Transform test data
predictions = pipeline_model.transform(test_df)

# Cross-validation
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Parameter grid
paramGrid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.1, 0.01, 0.001]) \
    .addGrid(lr.maxIter, [50, 100, 200]) \
    .build()

# Cross-validator
crossval = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    numFolds=5
)

# Train with cross-validation
cv_model = crossval.fit(train_df)
best_model = cv_model.bestModel
```

---

## Streaming

### Structured Streaming Basics
```python
from pyspark.sql.streaming import *

# Read streaming data
streaming_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "topic1") \
    .load()

# File stream
file_stream = spark.readStream \
    .format("csv") \
    .option("header", "true") \
    .schema(schema) \
    .load("input_directory/")

# Socket stream (for testing)
socket_stream = spark.readStream \
    .format("socket") \
    .option("host", "localhost") \
    .option("port", 9999) \
    .load()
```

### Stream Processing
```python
# Basic transformations
processed_stream = streaming_df \
    .select(
        get_json_object(col("value").cast("string"), "$.name").alias("name"),
        get_json_object(col("value").cast("string"), "$.age").cast("int").alias("age")
    ) \
    .filter(col("age") > 18)

# Aggregations with windowing
windowed_counts = streaming_df \
    .withColumn("timestamp", current_timestamp()) \
    .groupBy(
        window(col("timestamp"), "10 minutes", "5 minutes"),
        col("category")
    ) \
    .count()

# Watermarking for late data
watermarked_stream = streaming_df \
    .withWatermark("timestamp", "10 minutes") \
    .groupBy(
        window(col("timestamp"), "5 minutes"),
        col("device_id")
    ) \
    .count()
```

### Stream Outputs
```python
# Console output (for testing)
query = processed_stream \
    .writeStream \
    .outputMode("append") \
    .format("console") \
    .trigger(processingTime="10 seconds") \
    .start()

# File output
file_query = processed_stream \
    .writeStream \
    .outputMode("append") \
    .format("parquet") \
    .option("path", "output_directory/") \
    .option("checkpointLocation", "checkpoint_directory/") \
    .start()

# Kafka output
kafka_query = processed_stream \
    .selectExpr("to_json(struct(*)) AS value") \
    .writeStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("topic", "output_topic") \
    .option("checkpointLocation", "checkpoint_directory/") \
    .start()

# Memory output (for testing)
memory_query = processed_stream \
    .writeStream \
    .outputMode("complete") \
    .format("memory") \
    .queryName("streaming_table") \
    .start()

# Query management
query.awaitTermination()
query.stop()
spark.sql("SELECT * FROM streaming_table").show()
```

### Advanced Streaming
```python
# Stream-to-stream joins
stream1 = spark.readStream.format("...").load()
stream2 = spark.readStream.format("...").load()

joined_stream = stream1.join(
    stream2,
    expr("""
        stream1.user_id = stream2.user_id AND
        stream2.timestamp >= stream1.timestamp AND
        stream2.timestamp <= stream1.timestamp + interval 1 hour
    """)
)

# Deduplication
deduplicated_stream = streaming_df \
    .withWatermark("timestamp", "10 minutes") \
    .dropDuplicates(["user_id", "event_id"])

# Foreach batch operations
def process_batch(batch_df, batch_id):
    # Custom processing logic
    batch_df.write.mode("append").saveAsTable("processed_data")
    print(f"Processed batch {batch_id} with {batch_df.count()} records")

foreach_query = streaming_df \
    .writeStream \
    .foreachBatch(process_batch) \
    .start()
```

---

## Performance Optimization

### Data Partitioning
```python
# Check current partitioning
print(f"Number of partitions: {df.rdd.getNumPartitions()}")

# Repartition data
df_repartitioned = df.repartition(4)  # 4 partitions
df_repartitioned_by_col = df.repartition("department")  # Partition by column

# Coalesce (reduce partitions without shuffling)
df_coalesced = df.coalesce(2)

# Custom partitioning for key-value operations
from pyspark import HashPartitioner
rdd_partitioned = key_value_rdd.partitionBy(4, HashPartitioner())

# Optimal partition size (128MB recommended)
optimal_partitions = max(2, int(df.count() / 1000000))  # ~1M records per partition
```

### Caching and Persistence
```python
# Cache frequently used DataFrames
df.cache()  # Memory only
df.persist()  # Same as cache()

# Different storage levels
from pyspark import StorageLevel
df.persist(StorageLevel.MEMORY_AND_DISK)
df.persist(StorageLevel.DISK_ONLY)
df.persist(StorageLevel.MEMORY_ONLY_SER)

# Check if DataFrame is cached
print(f"Is cached: {df.is_cached}")

# Remove from cache
df.unpersist()

# Cache with specific storage level
df.persist(StorageLevel.MEMORY_AND_DISK_SER)

# Best practices for caching
# 1. Cache DataFrames that are reused multiple times
# 2. Cache after expensive transformations
# 3. Consider serialization for memory efficiency
expensive_df = df.filter(...).groupBy(...).agg(...)
expensive_df.cache()

# Use count() to trigger caching
expensive_df.count()  # Forces caching
```

### Query Optimization
```python
# Predicate pushdown (filter early)
# Good
df.filter(col("date") >= "2024-01-01").select("name", "age")

# Less efficient
df.select("name", "age").filter(col("date") >= "2024-01-01")

# Column pruning (select only needed columns)
df.select("name", "age").groupBy("name").count()  # Better than df.groupBy(...)

# Broadcast joins for small tables
from pyspark.sql.functions import broadcast
large_df.join(broadcast(small_df), "key")

# Bucketing for better join performance
df.write.bucketBy(4, "user_id").saveAsTable("bucketed_table")

# Z-order optimization (Delta Lake)
# spark.sql("OPTIMIZE table_name ZORDER BY (col1, col2)")

# Adaptive Query Execution (AQE) settings
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
```

### Memory Management
```python
# Monitor memory usage
spark.sparkContext.statusTracker().getExecutorInfos()

# Tune memory fractions
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "10000")
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

# Kryo serialization for better performance
spark.conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

# Dynamic allocation
spark.conf.set("spark.dynamicAllocation.enabled", "true")
spark.conf.set("spark.dynamicAllocation.minExecutors", "1")
spark.conf.set("spark.dynamicAllocation.maxExecutors", "10")

# Garbage collection tuning
spark.conf.set("spark.executor.extraJavaOptions", "-XX:+UseG1GC")
```

### Monitoring and Debugging
```python
# Access Spark UI
print(f"Spark UI: {spark.sparkContext.uiWebUrl}")

# Explain query execution plan
df.explain()  # Logical plan
df.explain("extended")  # Physical plan
df.explain("cost")  # Cost-based optimizer info

# Query execution statistics
df.count()  # Triggers action and shows timing
df.cache().count()  # Cache and measure

# Monitor job progress programmatically
status = spark.sparkContext.statusTracker()
for stage in status.getActiveStageInfos():
    print(f"Stage {stage.stageId}: {stage.numActiveTasks} active tasks")

# Custom metrics and logging
import logging
logger = logging.getLogger(__name__)

def log_dataframe_stats(df, name):
    count = df.count()
    partitions = df.rdd.getNumPartitions()
    logger.info(f"{name}: {count} rows, {partitions} partitions")
    return df

# Use with transform
df.transform(lambda x: log_dataframe_stats(x, "processed_df"))
```

---

## Advanced Techniques

### Custom Data Sources
```python
# Custom file format reader
def read_custom_format(spark, path):
    # Custom logic to read proprietary format
    raw_df = spark.read.text(path)
    
    # Parse and transform
    parsed_df = raw_df.select(
        regexp_extract(col("value"), r"(\w+)\|(\d+)\|(.+)", 1).alias("name"),
        regexp_extract(col("value"), r"(\w+)\|(\d+)\|(.+)", 2).cast("int").alias("age"),
        regexp_extract(col("value"), r"(\w+)\|(\d+)\|(.+)", 3).alias("description")
    )
    
    return parsed_df

# Custom writer
def write_custom_format(df, path):
    formatted_df = df.select(
        concat_ws("|", col("name"), col("age"), col("description")).alias("formatted")
    )
    formatted_df.select("formatted").write.mode("overwrite").text(path)
```

### Complex Data Types
```python
# Working with arrays
df.withColumn("array_col", array(lit(1), lit(2), lit(3))).show()
df.withColumn("exploded", explode(col("array_col"))).show()
df.withColumn("array_size", size(col("array_col"))).show()
df.withColumn("array_contains", array_contains(col("array_col"), 2)).show()

# Working with maps
df.withColumn("map_col", create_map(lit("key1"), lit("value1"), lit("key2"), lit("value2"))).show()
df.withColumn("map_keys", map_keys(col("map_col"))).show()
df.withColumn("map_values", map_values(col("map_col"))).show()

# Working with structs
df.withColumn("struct_col", struct(col("name"), col("age"))).show()
df.withColumn("name_from_struct", col("struct_col.name")).show()

# JSON processing
json_df = df.withColumn("json_col", to_json(struct(col("*"))))
parsed_df = json_df.withColumn("parsed", from_json(col("json_col"), schema))

# Schema inference for complex types
complex_schema = spark.read.json("complex.json").schema
df_with_schema = spark.read.schema(complex_schema).json("complex.json")
```

### Performance Profiling
```python
import time
from functools import wraps

def time_spark_operation(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

@time_spark_operation
def complex_transformation(df):
    return df.filter(col("age") > 25) \
             .groupBy("department") \
             .agg(avg("salary").alias("avg_salary")) \
             .orderBy("avg_salary")

# Memory profiling
def profile_dataframe(df, name="DataFrame"):
    print(f"\n{name} Profile:")
    print(f"- Rows: {df.count():,}")
    print(f"- Columns: {len(df.columns)}")
    print(f"- Partitions: {df.rdd.getNumPartitions()}")
    print(f"- Cached: {df.is_cached}")
    df.printSchema()
    return df

# Resource monitoring
def monitor_cluster_resources():
    executor_infos = spark.sparkContext.statusTracker().getExecutorInfos()
    for executor in executor_infos:
        print(f"Executor {executor.executorId}: "
              f"Memory {executor.memoryUsed}/{executor.maxMemory} "
              f"Tasks {executor.activeTasks}")
```

### Integration Patterns
```python
# Integration with pandas (PyArrow)
pandas_df = spark_df.toPandas()  # Spark to Pandas
spark_df_from_pandas = spark.createDataFrame(pandas_df)  # Pandas to Spark

# Integration with NumPy
import numpy as np
numpy_array = np.array(spark_df.collect())

# Integration with scikit-learn
from sklearn.ensemble import RandomForestRegressor

# Collect data for sklearn (small datasets only)
X = spark_df.select("features").rdd.map(lambda row: row.features.toArray()).collect()
y = spark_df.select("label").rdd.map(lambda row: row.label).collect()

sklearn_model = RandomForestRegressor()
sklearn_model.fit(X, y)

# Distributed model serving
def predict_batch(batch_df):
    # Convert to format expected by sklearn
    X_batch = np.array([row.features.toArray() for row in batch_df.collect()])
    predictions = sklearn_model.predict(X_batch)
    
    # Create result DataFrame
    result = spark.createDataFrame(
        [(float(pred),) for pred in predictions],
        ["prediction"]
    )
    return result

# Apply to streaming data
streaming_predictions = streaming_df.transform(predict_batch)
```

---

## Pro Tips & Best Practices

### Code Organization
```python
# 1. Use configuration classes
class SparkConfig:
    APP_NAME = "MySparkApp"
    MASTER = "local[*]"
    CONFIGS = {
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        "spark.serializer": "org.apache.spark.serializer.KryoSerializer"
    }

def create_spark_session(config: SparkConfig):
    builder = SparkSession.builder.appName(config.APP_NAME).master(config.MASTER)
    for key, value in config.CONFIGS.items():
        builder = builder.config(key, value)
    return builder.getOrCreate()

# 2. Use dependency injection for data sources
class DataSource:
    def read_data(self, spark: SparkSession):
        raise NotImplementedError

class CSVDataSource(DataSource):
    def __init__(self, path: str):
        self.path = path
    
    def read_data(self, spark: SparkSession):
        return spark.read.csv(self.path, header=True, inferSchema=True)

# 3. Create reusable transformation functions
def standardize_column_names(df):
    """Convert all column names to lowercase with underscores."""
    for col_name in df.columns:
        new_name = col_name.lower().replace(" ", "_").replace("-", "_")
        df = df.withColumnRenamed(col_name, new_name)
    return df

def add_audit_columns(df):
    """Add standard audit columns."""
    return df.withColumn("created_at", current_timestamp()) \
             .withColumn("batch_id", monotonically_increasing_id())
```

### Error Handling
```python
# Robust error handling
def safe_transformation(df, transformation_func, fallback_value=None):
    try:
        return transformation_func(df)
    except Exception as e:
        print(f"Transformation failed: {e}")
        if fallback_value is not None:
            return df.withColumn("error", lit(fallback_value))
        else:
            raise

# Data quality validation
def validate_dataframe(df, required_columns, min_rows=0):
    # Check required columns
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check minimum rows
    row_count = df.count()
    if row_count < min_rows:
        raise ValueError(f"DataFrame has only {row_count} rows, minimum {min_rows} required")
    
    return df

# Graceful degradation
def robust_join(left_df, right_df, join_keys, join_type="inner"):
    try:
        result = left_df.join(right_df, join_keys, join_type)
        
        # Validate join didn't explode data
        left_count = left_df.count()
        result_count = result.count()
        
        if join_type == "inner" and result_count > left_count * 2:
            print(f"Warning: Join may have duplicated data. "
                  f"Left: {left_count}, Result: {result_count}")
        
        return result
    except Exception as e:
        print(f"Join failed: {e}")
        return left_df  # Return original data
```

### Testing Strategies
```python
# Unit testing with small datasets
def create_test_dataframe(spark):
    test_data = [
        ("Alice", 25, "Engineering"),
        ("Bob", 30, "Sales"),
        ("Charlie", 35, "Engineering")
    ]
    return spark.createDataFrame(test_data, ["name", "age", "department"])

def test_age_categorization():
    spark = SparkSession.builder.master("local[1]").getOrCreate()
    test_df = create_test_dataframe(spark)
    
    result = test_df.withColumn("age_category",
        when(col("age") < 30, "Young").otherwise("Senior")
    )
    
    categories = [row.age_category for row in result.collect()]
    assert categories == ["Young", "Senior", "Senior"]

# Property-based testing
def test_dataframe_properties(df):
    assert df.count() >= 0
    assert len(df.columns) > 0
    assert all(col_name.strip() == col_name for col_name in df.columns)  # No whitespace

# Performance testing
def benchmark_transformation(df, transformation_func, iterations=3):
    times = []
    for i in range(iterations):
        start = time.time()
        result = transformation_func(df)
        result.count()  # Force evaluation
        end = time.time()
        times.append(end - start)
    
    avg_time = sum(times) / len(times)
    print(f"Average time: {avg_time:.2f}s over {iterations} iterations")
    return avg_time
```

### Common Gotchas
```python
# 1. Lazy evaluation gotcha
def incorrect_iteration():
    df = spark.range(100)
    
    for i in range(10):
        df = df.filter(col("id") > i)  # Creates long lineage
        # df.cache()  # Solution: cache intermediate results
    
    return df.count()  # Very slow due to long lineage

# 2. Skewed data gotcha
def handle_skewed_joins(large_df, small_df, join_key):
    # Add salt to break up skewed keys
    salted_large = large_df.withColumn("salted_key", 
        concat(col(join_key), lit("_"), (rand() * 100).cast("int")))
    
    # Replicate small table
    salts = spark.range(100).select(col("id").alias("salt"))
    replicated_small = small_df.crossJoin(salts).withColumn("salted_key",
        concat(col(join_key), lit("_"), col("salt")))
    
    return salted_large.join(replicated_small, "salted_key").drop("salted_key", "salt")

# 3. Memory leak gotcha
def avoid_memory_leaks():
    # Bad: accumulating DataFrames without cleanup
    results = []
    for i in range(100):
        df = spark.range(1000)
        results.append(df.filter(col("id") > i))
    
    # Good: process and clean up
    for i in range(100):
        df = spark.range(1000)
        processed = df.filter(col("id") > i)
        processed.write.mode("append").parquet(f"output/batch_{i}")
        # DataFrame is garbage collected after this iteration

# 4. Schema evolution gotcha
def handle_schema_evolution(df_list):
    # Get union of all schemas
    all_columns = set()
    for df in df_list:
        all_columns.update(df.columns)
    
    # Standardize all DataFrames to same schema
    standardized = []
    for df in df_list:
        for col_name in all_columns:
            if col_name not in df.columns:
                df = df.withColumn(col_name, lit(None))
        standardized.append(df.select(*sorted(all_columns)))
    
    return standardized[0].union(*standardized[1:])
```

---

**📚 Additional Resources:**
- [PySpark Documentation](https://spark.apache.org/docs/latest/api/python/)
- [Spark SQL Guide](https://spark.apache.org/docs/latest/sql-programming-guide.html)
- [MLlib Guide](https://spark.apache.org/docs/latest/ml-guide.html)
- [Structured Streaming Guide](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html)

**🔗 Related Cheatsheets:**
- [Python Cheatsheet](PYTHON_CHEATSHEET.md)
- [NumPy Cheatsheet](NUMPY_CHEATSHEET.md)
- [Pandas Cheatsheet](PANDAS_CHEATSHEET.md)
- [PyTorch Cheatsheet](PYTORCH_CHEATSHEET.md)

---
*Last Updated: January 8, 2026*
