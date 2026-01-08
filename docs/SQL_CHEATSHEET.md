# SQL Comprehensive Cheatsheet

**Level:** SparkNotes/Unbound Quality Reference  
**Date:** January 8, 2026  
**SQL Standard:** ANSI SQL with PostgreSQL/MySQL/SQLite extensions  
**Coverage:** DDL, DML, DQL, TCL, DCL, Window Functions, CTEs, Performance

## Table of Contents

- [Quick Start & Core Concepts](#quick-start--core-concepts)
- [Data Definition Language (DDL)](#data-definition-language-ddl)
- [Data Manipulation Language (DML)](#data-manipulation-language-dml)
- [Data Query Language (DQL)](#data-query-language-dql)
- [Joins & Subqueries](#joins--subqueries)
- [Aggregations & Grouping](#aggregations--grouping)
- [Window Functions](#window-functions)
- [Common Table Expressions (CTEs)](#common-table-expressions-ctes)
- [String Functions](#string-functions)
- [Date & Time Functions](#date--time-functions)
- [Conditional Logic](#conditional-logic)
- [Indexes & Performance](#indexes--performance)
- [Transactions & Constraints](#transactions--constraints)
- [Advanced Techniques](#advanced-techniques)

---

## Quick Start & Core Concepts

### SQL Fundamentals
```sql
-- SQL is case-insensitive for keywords, but case-sensitive for data
-- Convention: UPPERCASE for keywords, lowercase for table/column names

-- Basic query structure
SELECT column1, column2
FROM table_name
WHERE condition
ORDER BY column1;

-- Query execution order (logical):
-- 1. FROM & JOINs
-- 2. WHERE
-- 3. GROUP BY
-- 4. HAVING
-- 5. SELECT
-- 6. DISTINCT
-- 7. ORDER BY
-- 8. LIMIT/OFFSET
```

### Data Types (Common across databases)
```sql
-- Numeric types
INTEGER, INT          -- Whole numbers (-2B to 2B)
BIGINT               -- Large integers (-9 quintillion to 9 quintillion)
SMALLINT             -- Small integers (-32K to 32K)
DECIMAL(p,s)         -- Exact decimal (precision, scale)
NUMERIC(p,s)         -- Same as DECIMAL
FLOAT, REAL          -- Floating-point (approximate)
DOUBLE PRECISION     -- Double-precision floating-point

-- String types
CHAR(n)              -- Fixed-length string
VARCHAR(n)           -- Variable-length string (max n)
TEXT                 -- Unlimited text (PostgreSQL, MySQL)

-- Date/Time types
DATE                 -- Date only (YYYY-MM-DD)
TIME                 -- Time only (HH:MM:SS)
TIMESTAMP            -- Date and time
DATETIME             -- Date and time (MySQL)
INTERVAL             -- Time interval (PostgreSQL)

-- Boolean
BOOLEAN, BOOL        -- TRUE/FALSE/NULL

-- Binary
BLOB                 -- Binary large object
BYTEA                -- Binary data (PostgreSQL)

-- JSON (Modern databases)
JSON                 -- JSON data
JSONB                -- Binary JSON (PostgreSQL, faster queries)
```

---

## Data Definition Language (DDL)

### Creating Tables
```sql
-- Basic table creation
CREATE TABLE employees (
    id SERIAL PRIMARY KEY,           -- Auto-increment (PostgreSQL)
    -- id INT AUTO_INCREMENT PRIMARY KEY,  -- MySQL
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    email VARCHAR(100) UNIQUE,
    hire_date DATE DEFAULT CURRENT_DATE,
    salary DECIMAL(10, 2) CHECK (salary > 0),
    department_id INT REFERENCES departments(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create table from query
CREATE TABLE high_earners AS
SELECT * FROM employees WHERE salary > 100000;

-- Create temporary table
CREATE TEMPORARY TABLE temp_results AS
SELECT * FROM employees WHERE department_id = 5;

-- Create table if not exists
CREATE TABLE IF NOT EXISTS logs (
    id SERIAL PRIMARY KEY,
    message TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### Altering Tables
```sql
-- Add column
ALTER TABLE employees ADD COLUMN middle_name VARCHAR(50);
ALTER TABLE employees ADD COLUMN is_active BOOLEAN DEFAULT TRUE;

-- Drop column
ALTER TABLE employees DROP COLUMN middle_name;

-- Rename column
ALTER TABLE employees RENAME COLUMN first_name TO fname;

-- Modify column type
ALTER TABLE employees ALTER COLUMN salary TYPE DECIMAL(12, 2);
-- MySQL: ALTER TABLE employees MODIFY COLUMN salary DECIMAL(12, 2);

-- Add constraint
ALTER TABLE employees ADD CONSTRAINT chk_salary CHECK (salary >= 0);
ALTER TABLE employees ADD CONSTRAINT fk_dept 
    FOREIGN KEY (department_id) REFERENCES departments(id);

-- Drop constraint
ALTER TABLE employees DROP CONSTRAINT chk_salary;

-- Rename table
ALTER TABLE employees RENAME TO staff;

-- Add default value
ALTER TABLE employees ALTER COLUMN is_active SET DEFAULT TRUE;

-- Remove default value
ALTER TABLE employees ALTER COLUMN is_active DROP DEFAULT;
```

### Dropping and Truncating
```sql
-- Drop table (removes table and data)
DROP TABLE employees;
DROP TABLE IF EXISTS employees;
DROP TABLE employees CASCADE;  -- Also drops dependent objects

-- Truncate table (removes all data, keeps structure)
TRUNCATE TABLE employees;
TRUNCATE TABLE employees RESTART IDENTITY;  -- Reset auto-increment
TRUNCATE TABLE employees CASCADE;  -- Also truncates dependent tables

-- Drop multiple tables
DROP TABLE table1, table2, table3;
```

### Views
```sql
-- Create view
CREATE VIEW employee_summary AS
SELECT 
    e.id,
    e.first_name || ' ' || e.last_name AS full_name,
    d.name AS department,
    e.salary
FROM employees e
JOIN departments d ON e.department_id = d.id;

-- Create or replace view
CREATE OR REPLACE VIEW employee_summary AS
SELECT * FROM employees WHERE is_active = TRUE;

-- Materialized view (PostgreSQL)
CREATE MATERIALIZED VIEW sales_summary AS
SELECT 
    date_trunc('month', sale_date) AS month,
    SUM(amount) AS total_sales
FROM sales
GROUP BY 1;

-- Refresh materialized view
REFRESH MATERIALIZED VIEW sales_summary;
REFRESH MATERIALIZED VIEW CONCURRENTLY sales_summary;  -- No lock

-- Drop view
DROP VIEW employee_summary;
DROP VIEW IF EXISTS employee_summary;
```

---

## Data Manipulation Language (DML)

### INSERT Operations
```sql
-- Insert single row
INSERT INTO employees (first_name, last_name, email, salary)
VALUES ('John', 'Doe', 'john@example.com', 75000);

-- Insert multiple rows
INSERT INTO employees (first_name, last_name, email, salary)
VALUES 
    ('Jane', 'Smith', 'jane@example.com', 80000),
    ('Bob', 'Johnson', 'bob@example.com', 70000),
    ('Alice', 'Williams', 'alice@example.com', 85000);

-- Insert from query
INSERT INTO high_earners (first_name, last_name, salary)
SELECT first_name, last_name, salary
FROM employees
WHERE salary > 100000;

-- Insert with returning clause (PostgreSQL)
INSERT INTO employees (first_name, last_name, email)
VALUES ('Tom', 'Wilson', 'tom@example.com')
RETURNING id, first_name;

-- Insert or update (UPSERT)
-- PostgreSQL
INSERT INTO employees (id, first_name, last_name, email)
VALUES (1, 'John', 'Doe', 'john@example.com')
ON CONFLICT (id) DO UPDATE SET
    first_name = EXCLUDED.first_name,
    last_name = EXCLUDED.last_name,
    email = EXCLUDED.email;

-- MySQL
INSERT INTO employees (id, first_name, last_name, email)
VALUES (1, 'John', 'Doe', 'john@example.com')
ON DUPLICATE KEY UPDATE
    first_name = VALUES(first_name),
    last_name = VALUES(last_name);

-- Insert ignore (MySQL - skip duplicates)
INSERT IGNORE INTO employees (id, first_name) VALUES (1, 'John');
```

### UPDATE Operations
```sql
-- Basic update
UPDATE employees
SET salary = 80000
WHERE id = 1;

-- Update multiple columns
UPDATE employees
SET 
    salary = salary * 1.10,
    last_updated = CURRENT_TIMESTAMP
WHERE department_id = 5;

-- Update with subquery
UPDATE employees
SET salary = (
    SELECT AVG(salary) 
    FROM employees 
    WHERE department_id = employees.department_id
)
WHERE performance_rating < 3;

-- Update with JOIN (PostgreSQL)
UPDATE employees e
SET salary = e.salary * 1.15
FROM departments d
WHERE e.department_id = d.id
  AND d.name = 'Engineering';

-- Update with JOIN (MySQL)
UPDATE employees e
JOIN departments d ON e.department_id = d.id
SET e.salary = e.salary * 1.15
WHERE d.name = 'Engineering';

-- Update with CASE
UPDATE employees
SET salary = CASE
    WHEN performance_rating >= 5 THEN salary * 1.20
    WHEN performance_rating >= 4 THEN salary * 1.15
    WHEN performance_rating >= 3 THEN salary * 1.10
    ELSE salary
END;

-- Update returning (PostgreSQL)
UPDATE employees
SET salary = salary * 1.10
WHERE department_id = 5
RETURNING id, first_name, salary;
```

### DELETE Operations
```sql
-- Basic delete
DELETE FROM employees WHERE id = 1;

-- Delete multiple rows
DELETE FROM employees WHERE department_id = 5;

-- Delete all rows (prefer TRUNCATE for performance)
DELETE FROM employees;

-- Delete with subquery
DELETE FROM employees
WHERE department_id IN (
    SELECT id FROM departments WHERE is_closed = TRUE
);

-- Delete with JOIN (PostgreSQL)
DELETE FROM employees e
USING departments d
WHERE e.department_id = d.id
  AND d.name = 'Closed Department';

-- Delete returning (PostgreSQL)
DELETE FROM employees
WHERE id = 1
RETURNING *;

-- Delete duplicates keeping one
DELETE FROM employees a
USING employees b
WHERE a.id < b.id
  AND a.email = b.email;

-- Alternative for deleting duplicates
DELETE FROM employees
WHERE id NOT IN (
    SELECT MIN(id)
    FROM employees
    GROUP BY email
);
```

---

## Data Query Language (DQL)

### SELECT Fundamentals
```sql
-- Select all columns
SELECT * FROM employees;

-- Select specific columns
SELECT first_name, last_name, salary FROM employees;

-- Column aliases
SELECT 
    first_name AS "First Name",
    last_name AS "Last Name",
    salary * 12 AS annual_salary
FROM employees;

-- Distinct values
SELECT DISTINCT department_id FROM employees;
SELECT DISTINCT ON (department_id) * FROM employees;  -- PostgreSQL

-- Limit results
SELECT * FROM employees LIMIT 10;
SELECT * FROM employees LIMIT 10 OFFSET 20;  -- Pagination

-- Top N (SQL Server)
SELECT TOP 10 * FROM employees;
SELECT TOP 10 PERCENT * FROM employees;
```

### WHERE Clause
```sql
-- Comparison operators
SELECT * FROM employees WHERE salary > 50000;
SELECT * FROM employees WHERE salary >= 50000;
SELECT * FROM employees WHERE salary < 100000;
SELECT * FROM employees WHERE salary <= 100000;
SELECT * FROM employees WHERE salary = 75000;
SELECT * FROM employees WHERE salary <> 75000;  -- Not equal
SELECT * FROM employees WHERE salary != 75000;  -- Not equal (alternative)

-- Logical operators
SELECT * FROM employees 
WHERE salary > 50000 AND department_id = 5;

SELECT * FROM employees 
WHERE salary > 100000 OR department_id = 5;

SELECT * FROM employees 
WHERE NOT department_id = 5;

-- BETWEEN (inclusive)
SELECT * FROM employees 
WHERE salary BETWEEN 50000 AND 100000;

SELECT * FROM employees 
WHERE hire_date BETWEEN '2020-01-01' AND '2023-12-31';

-- IN operator
SELECT * FROM employees 
WHERE department_id IN (1, 2, 3);

SELECT * FROM employees 
WHERE department_id IN (SELECT id FROM departments WHERE name LIKE 'Eng%');

-- NOT IN
SELECT * FROM employees 
WHERE department_id NOT IN (5, 6, 7);

-- LIKE pattern matching
SELECT * FROM employees WHERE first_name LIKE 'J%';      -- Starts with J
SELECT * FROM employees WHERE first_name LIKE '%son';    -- Ends with son
SELECT * FROM employees WHERE first_name LIKE '%oh%';    -- Contains oh
SELECT * FROM employees WHERE first_name LIKE 'J___';    -- J + 3 chars
SELECT * FROM employees WHERE email LIKE '%@gmail.com';

-- ILIKE (case-insensitive, PostgreSQL)
SELECT * FROM employees WHERE first_name ILIKE 'john';

-- Regular expressions (PostgreSQL)
SELECT * FROM employees WHERE first_name ~ '^J.*n$';     -- Starts J, ends n
SELECT * FROM employees WHERE first_name ~* '^j.*n$';    -- Case-insensitive

-- NULL handling
SELECT * FROM employees WHERE manager_id IS NULL;
SELECT * FROM employees WHERE manager_id IS NOT NULL;

-- COALESCE for null substitution
SELECT COALESCE(middle_name, 'N/A') FROM employees;
```

### ORDER BY
```sql
-- Ascending order (default)
SELECT * FROM employees ORDER BY last_name;
SELECT * FROM employees ORDER BY last_name ASC;

-- Descending order
SELECT * FROM employees ORDER BY salary DESC;

-- Multiple columns
SELECT * FROM employees ORDER BY department_id, salary DESC;

-- Order by expression
SELECT * FROM employees ORDER BY salary * 12 DESC;

-- Order by column position
SELECT first_name, last_name, salary FROM employees ORDER BY 3 DESC;

-- Order by alias
SELECT first_name, salary * 12 AS annual FROM employees ORDER BY annual DESC;

-- NULLS FIRST/LAST (PostgreSQL)
SELECT * FROM employees ORDER BY manager_id NULLS FIRST;
SELECT * FROM employees ORDER BY manager_id DESC NULLS LAST;

-- Custom ordering with CASE
SELECT * FROM employees
ORDER BY CASE department_id
    WHEN 1 THEN 1
    WHEN 5 THEN 2
    ELSE 3
END;
```

---

## Joins & Subqueries

### Join Types
```sql
-- INNER JOIN (only matching rows)
SELECT e.first_name, e.last_name, d.name AS department
FROM employees e
INNER JOIN departments d ON e.department_id = d.id;

-- LEFT JOIN (all from left, matching from right)
SELECT e.first_name, e.last_name, d.name AS department
FROM employees e
LEFT JOIN departments d ON e.department_id = d.id;

-- RIGHT JOIN (all from right, matching from left)
SELECT e.first_name, e.last_name, d.name AS department
FROM employees e
RIGHT JOIN departments d ON e.department_id = d.id;

-- FULL OUTER JOIN (all from both)
SELECT e.first_name, e.last_name, d.name AS department
FROM employees e
FULL OUTER JOIN departments d ON e.department_id = d.id;

-- CROSS JOIN (Cartesian product)
SELECT e.first_name, p.project_name
FROM employees e
CROSS JOIN projects p;

-- Self join
SELECT 
    e.first_name AS employee,
    m.first_name AS manager
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.id;

-- Multiple joins
SELECT 
    e.first_name,
    d.name AS department,
    p.name AS project
FROM employees e
JOIN departments d ON e.department_id = d.id
JOIN employee_projects ep ON e.id = ep.employee_id
JOIN projects p ON ep.project_id = p.id;

-- Join with multiple conditions
SELECT *
FROM orders o
JOIN order_items oi ON o.id = oi.order_id AND oi.quantity > 0;

-- NATURAL JOIN (joins on same-named columns)
SELECT * FROM employees NATURAL JOIN departments;

-- USING clause (when column names match)
SELECT * FROM employees e JOIN departments d USING (department_id);
```

### Subqueries
```sql
-- Scalar subquery (returns single value)
SELECT 
    first_name,
    salary,
    (SELECT AVG(salary) FROM employees) AS avg_salary
FROM employees;

-- Subquery in WHERE
SELECT * FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees);

SELECT * FROM employees
WHERE department_id IN (
    SELECT id FROM departments WHERE name LIKE 'Eng%'
);

-- Correlated subquery (references outer query)
SELECT e.first_name, e.salary
FROM employees e
WHERE e.salary > (
    SELECT AVG(salary) 
    FROM employees 
    WHERE department_id = e.department_id
);

-- EXISTS subquery
SELECT * FROM departments d
WHERE EXISTS (
    SELECT 1 FROM employees e WHERE e.department_id = d.id
);

SELECT * FROM departments d
WHERE NOT EXISTS (
    SELECT 1 FROM employees e WHERE e.department_id = d.id
);

-- Subquery in FROM (derived table)
SELECT department, avg_salary
FROM (
    SELECT department_id, AVG(salary) AS avg_salary
    FROM employees
    GROUP BY department_id
) AS dept_stats
WHERE avg_salary > 70000;

-- ALL and ANY operators
SELECT * FROM employees
WHERE salary > ALL (SELECT salary FROM employees WHERE department_id = 5);

SELECT * FROM employees
WHERE salary > ANY (SELECT salary FROM employees WHERE department_id = 5);

-- Lateral join (PostgreSQL)
SELECT e.first_name, recent_orders.*
FROM employees e
CROSS JOIN LATERAL (
    SELECT * FROM orders o
    WHERE o.employee_id = e.id
    ORDER BY o.order_date DESC
    LIMIT 3
) AS recent_orders;
```

---

## Aggregations & Grouping

### Aggregate Functions
```sql
-- Count functions
SELECT COUNT(*) FROM employees;                    -- Count all rows
SELECT COUNT(manager_id) FROM employees;           -- Count non-null
SELECT COUNT(DISTINCT department_id) FROM employees;  -- Count unique

-- Numeric aggregates
SELECT 
    SUM(salary) AS total_salary,
    AVG(salary) AS avg_salary,
    MIN(salary) AS min_salary,
    MAX(salary) AS max_salary
FROM employees;

-- String aggregates
SELECT STRING_AGG(first_name, ', ') FROM employees;  -- PostgreSQL
SELECT GROUP_CONCAT(first_name SEPARATOR ', ') FROM employees;  -- MySQL

-- Array aggregate (PostgreSQL)
SELECT ARRAY_AGG(first_name) FROM employees;

-- JSON aggregate (PostgreSQL)
SELECT JSON_AGG(row_to_json(e)) FROM employees e;

-- Statistical functions
SELECT 
    STDDEV(salary) AS standard_deviation,
    VARIANCE(salary) AS variance,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY salary) AS median
FROM employees;
```

### GROUP BY
```sql
-- Basic grouping
SELECT department_id, COUNT(*) AS employee_count
FROM employees
GROUP BY department_id;

-- Multiple grouping columns
SELECT department_id, EXTRACT(YEAR FROM hire_date) AS year, COUNT(*)
FROM employees
GROUP BY department_id, EXTRACT(YEAR FROM hire_date);

-- Grouping with aggregate functions
SELECT 
    department_id,
    COUNT(*) AS count,
    SUM(salary) AS total_salary,
    AVG(salary) AS avg_salary,
    MIN(salary) AS min_salary,
    MAX(salary) AS max_salary
FROM employees
GROUP BY department_id
ORDER BY avg_salary DESC;

-- HAVING clause (filter groups)
SELECT department_id, AVG(salary) AS avg_salary
FROM employees
GROUP BY department_id
HAVING AVG(salary) > 70000;

SELECT department_id, COUNT(*) AS employee_count
FROM employees
GROUP BY department_id
HAVING COUNT(*) >= 5;

-- GROUPING SETS (multiple groupings)
SELECT department_id, EXTRACT(YEAR FROM hire_date) AS year, SUM(salary)
FROM employees
GROUP BY GROUPING SETS (
    (department_id, EXTRACT(YEAR FROM hire_date)),
    (department_id),
    (EXTRACT(YEAR FROM hire_date)),
    ()
);

-- ROLLUP (hierarchical aggregation)
SELECT 
    department_id,
    job_title,
    SUM(salary) AS total_salary
FROM employees
GROUP BY ROLLUP (department_id, job_title);

-- CUBE (all combinations)
SELECT 
    department_id,
    job_title,
    SUM(salary) AS total_salary
FROM employees
GROUP BY CUBE (department_id, job_title);
```

---

## Window Functions

### Ranking Functions
```sql
-- ROW_NUMBER (unique sequential number)
SELECT 
    first_name,
    department_id,
    salary,
    ROW_NUMBER() OVER (ORDER BY salary DESC) AS row_num
FROM employees;

-- ROW_NUMBER within partition
SELECT 
    first_name,
    department_id,
    salary,
    ROW_NUMBER() OVER (PARTITION BY department_id ORDER BY salary DESC) AS dept_rank
FROM employees;

-- RANK (gaps for ties)
SELECT 
    first_name,
    salary,
    RANK() OVER (ORDER BY salary DESC) AS rank
FROM employees;
-- Result: 1, 2, 2, 4, 5 (gap after ties)

-- DENSE_RANK (no gaps)
SELECT 
    first_name,
    salary,
    DENSE_RANK() OVER (ORDER BY salary DESC) AS dense_rank
FROM employees;
-- Result: 1, 2, 2, 3, 4 (no gaps)

-- NTILE (divide into buckets)
SELECT 
    first_name,
    salary,
    NTILE(4) OVER (ORDER BY salary DESC) AS quartile
FROM employees;

-- PERCENT_RANK and CUME_DIST
SELECT 
    first_name,
    salary,
    PERCENT_RANK() OVER (ORDER BY salary) AS pct_rank,
    CUME_DIST() OVER (ORDER BY salary) AS cume_dist
FROM employees;
```

### Aggregate Window Functions
```sql
-- Running totals
SELECT 
    order_date,
    amount,
    SUM(amount) OVER (ORDER BY order_date) AS running_total
FROM orders;

-- Running average
SELECT 
    order_date,
    amount,
    AVG(amount) OVER (ORDER BY order_date) AS running_avg
FROM orders;

-- Partition aggregates
SELECT 
    first_name,
    department_id,
    salary,
    SUM(salary) OVER (PARTITION BY department_id) AS dept_total,
    AVG(salary) OVER (PARTITION BY department_id) AS dept_avg,
    salary - AVG(salary) OVER (PARTITION BY department_id) AS diff_from_avg
FROM employees;

-- Count window
SELECT 
    first_name,
    department_id,
    COUNT(*) OVER (PARTITION BY department_id) AS dept_count,
    COUNT(*) OVER () AS total_count
FROM employees;
```

### Navigation Functions
```sql
-- LAG (previous row value)
SELECT 
    order_date,
    amount,
    LAG(amount, 1) OVER (ORDER BY order_date) AS prev_amount,
    amount - LAG(amount, 1) OVER (ORDER BY order_date) AS change
FROM orders;

-- LAG with default value
SELECT 
    order_date,
    amount,
    LAG(amount, 1, 0) OVER (ORDER BY order_date) AS prev_amount
FROM orders;

-- LEAD (next row value)
SELECT 
    order_date,
    amount,
    LEAD(amount, 1) OVER (ORDER BY order_date) AS next_amount
FROM orders;

-- FIRST_VALUE and LAST_VALUE
SELECT 
    first_name,
    department_id,
    salary,
    FIRST_VALUE(first_name) OVER (
        PARTITION BY department_id 
        ORDER BY salary DESC
    ) AS highest_paid,
    LAST_VALUE(first_name) OVER (
        PARTITION BY department_id 
        ORDER BY salary DESC
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS lowest_paid
FROM employees;

-- NTH_VALUE
SELECT 
    first_name,
    salary,
    NTH_VALUE(first_name, 2) OVER (ORDER BY salary DESC) AS second_highest
FROM employees;
```

### Window Frame Specification
```sql
-- ROWS frame (physical rows)
SELECT 
    order_date,
    amount,
    AVG(amount) OVER (
        ORDER BY order_date
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS moving_avg_3
FROM orders;

-- RANGE frame (logical range)
SELECT 
    order_date,
    amount,
    SUM(amount) OVER (
        ORDER BY order_date
        RANGE BETWEEN INTERVAL '7 days' PRECEDING AND CURRENT ROW
    ) AS weekly_sum
FROM orders;

-- Frame options
ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW      -- All rows up to current
ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING      -- Current to end
ROWS BETWEEN 3 PRECEDING AND 3 FOLLOWING              -- Sliding window of 7
ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING  -- All rows

-- Named windows
SELECT 
    first_name,
    department_id,
    salary,
    SUM(salary) OVER w AS running_total,
    AVG(salary) OVER w AS running_avg
FROM employees
WINDOW w AS (PARTITION BY department_id ORDER BY salary);
```

---

## Common Table Expressions (CTEs)

### Basic CTEs
```sql
-- Simple CTE
WITH high_earners AS (
    SELECT * FROM employees WHERE salary > 100000
)
SELECT * FROM high_earners ORDER BY salary DESC;

-- Multiple CTEs
WITH 
dept_stats AS (
    SELECT 
        department_id,
        AVG(salary) AS avg_salary,
        COUNT(*) AS employee_count
    FROM employees
    GROUP BY department_id
),
high_avg_depts AS (
    SELECT * FROM dept_stats WHERE avg_salary > 75000
)
SELECT d.name, h.avg_salary, h.employee_count
FROM high_avg_depts h
JOIN departments d ON h.department_id = d.id;

-- CTE referencing another CTE
WITH 
employees_with_tenure AS (
    SELECT 
        *,
        EXTRACT(YEAR FROM AGE(CURRENT_DATE, hire_date)) AS tenure_years
    FROM employees
),
experienced_employees AS (
    SELECT * FROM employees_with_tenure WHERE tenure_years > 5
)
SELECT * FROM experienced_employees WHERE salary > 80000;
```

### Recursive CTEs
```sql
-- Organizational hierarchy
WITH RECURSIVE org_chart AS (
    -- Base case: top-level employees (no manager)
    SELECT 
        id, 
        first_name, 
        manager_id, 
        1 AS level,
        first_name::TEXT AS path
    FROM employees
    WHERE manager_id IS NULL
    
    UNION ALL
    
    -- Recursive case: employees with managers
    SELECT 
        e.id, 
        e.first_name, 
        e.manager_id, 
        oc.level + 1,
        oc.path || ' > ' || e.first_name
    FROM employees e
    JOIN org_chart oc ON e.manager_id = oc.id
)
SELECT * FROM org_chart ORDER BY level, first_name;

-- Number sequence
WITH RECURSIVE numbers AS (
    SELECT 1 AS n
    UNION ALL
    SELECT n + 1 FROM numbers WHERE n < 100
)
SELECT * FROM numbers;

-- Fibonacci sequence
WITH RECURSIVE fib AS (
    SELECT 1 AS n, 0::BIGINT AS fib_n, 1::BIGINT AS fib_n1
    UNION ALL
    SELECT n + 1, fib_n1, fib_n + fib_n1
    FROM fib
    WHERE n < 50
)
SELECT n, fib_n FROM fib;

-- Graph traversal (find all connected nodes)
WITH RECURSIVE connected AS (
    SELECT node_id, connected_to, 1 AS depth
    FROM graph
    WHERE node_id = 1
    
    UNION
    
    SELECT g.node_id, g.connected_to, c.depth + 1
    FROM graph g
    JOIN connected c ON g.node_id = c.connected_to
    WHERE c.depth < 10  -- Prevent infinite loops
)
SELECT DISTINCT node_id, connected_to, depth FROM connected;
```

### CTE with DML (PostgreSQL)
```sql
-- CTE with INSERT
WITH new_employees AS (
    INSERT INTO employees (first_name, last_name, salary, department_id)
    VALUES ('John', 'Doe', 75000, 1)
    RETURNING *
)
SELECT * FROM new_employees;

-- CTE with UPDATE
WITH updated AS (
    UPDATE employees
    SET salary = salary * 1.10
    WHERE department_id = 5
    RETURNING *
)
SELECT AVG(salary) FROM updated;

-- CTE with DELETE
WITH deleted AS (
    DELETE FROM inactive_employees
    WHERE last_login < CURRENT_DATE - INTERVAL '1 year'
    RETURNING *
)
INSERT INTO archived_employees SELECT * FROM deleted;
```

---

## String Functions

### String Manipulation
```sql
-- Concatenation
SELECT first_name || ' ' || last_name AS full_name FROM employees;
SELECT CONCAT(first_name, ' ', last_name) FROM employees;
SELECT CONCAT_WS(' ', first_name, middle_name, last_name) FROM employees;  -- With separator

-- Case conversion
SELECT UPPER(first_name), LOWER(last_name), INITCAP(first_name) FROM employees;

-- Trimming
SELECT TRIM(name) FROM employees;           -- Both sides
SELECT LTRIM(name) FROM employees;          -- Left side
SELECT RTRIM(name) FROM employees;          -- Right side
SELECT TRIM(BOTH ' ' FROM name) FROM employees;
SELECT TRIM(LEADING '0' FROM phone) FROM employees;

-- Substring
SELECT SUBSTRING(name FROM 1 FOR 3) FROM employees;  -- First 3 chars
SELECT SUBSTR(name, 1, 3) FROM employees;            -- Alternative syntax
SELECT LEFT(name, 3) FROM employees;                 -- First 3 chars
SELECT RIGHT(name, 3) FROM employees;                -- Last 3 chars

-- Position/Index
SELECT POSITION('o' IN name) FROM employees;         -- Position of 'o'
SELECT STRPOS(name, 'o') FROM employees;             -- PostgreSQL
SELECT INSTR(name, 'o') FROM employees;              -- MySQL

-- Length
SELECT LENGTH(name) FROM employees;
SELECT CHAR_LENGTH(name) FROM employees;

-- Replace
SELECT REPLACE(name, 'old', 'new') FROM employees;
SELECT TRANSLATE(name, 'abc', 'xyz') FROM employees;  -- Character mapping

-- Padding
SELECT LPAD(id::TEXT, 5, '0') FROM employees;        -- Left pad: '00001'
SELECT RPAD(name, 20, '.') FROM employees;           -- Right pad: 'John...........'

-- Repeat
SELECT REPEAT('*', 10);                              -- '**********'

-- Reverse
SELECT REVERSE(name) FROM employees;

-- Split and array
SELECT STRING_TO_ARRAY(tags, ',') FROM posts;        -- PostgreSQL
SELECT SPLIT_PART(email, '@', 2) FROM employees;     -- Domain from email
```

### Pattern Matching
```sql
-- LIKE patterns
SELECT * FROM employees WHERE name LIKE 'J%';        -- Starts with J
SELECT * FROM employees WHERE name LIKE '%son';      -- Ends with son
SELECT * FROM employees WHERE name LIKE '_o%';       -- Second char is o
SELECT * FROM employees WHERE name LIKE '%[aeiou]%'; -- Contains vowel (SQL Server)

-- Escape special characters
SELECT * FROM products WHERE name LIKE '%10\%%' ESCAPE '\';

-- Regular expressions (PostgreSQL)
SELECT * FROM employees WHERE name ~ '^J.*n$';       -- Case sensitive
SELECT * FROM employees WHERE name ~* '^j.*n$';      -- Case insensitive
SELECT * FROM employees WHERE name !~ '^J';          -- NOT matching

-- SIMILAR TO (SQL standard regex)
SELECT * FROM employees 
WHERE email SIMILAR TO '[a-z]+@[a-z]+\.[a-z]+';

-- Regex functions (PostgreSQL)
SELECT REGEXP_MATCHES(text, '\d+', 'g') FROM documents;  -- All matches
SELECT REGEXP_REPLACE(phone, '[^0-9]', '', 'g') FROM employees;  -- Numbers only
SELECT REGEXP_SPLIT_TO_TABLE(tags, ',') FROM posts;  -- Split to rows
```

---

## Date & Time Functions

### Current Date/Time
```sql
-- Current date/time
SELECT CURRENT_DATE;                    -- Date only
SELECT CURRENT_TIME;                    -- Time only
SELECT CURRENT_TIMESTAMP;               -- Date and time
SELECT NOW();                           -- PostgreSQL, MySQL
SELECT LOCALTIMESTAMP;                  -- Without timezone

-- Date/time at start of transaction
SELECT TRANSACTION_TIMESTAMP();         -- PostgreSQL
```

### Date Extraction
```sql
-- EXTRACT function
SELECT EXTRACT(YEAR FROM hire_date) FROM employees;
SELECT EXTRACT(MONTH FROM hire_date) FROM employees;
SELECT EXTRACT(DAY FROM hire_date) FROM employees;
SELECT EXTRACT(DOW FROM hire_date) FROM employees;    -- Day of week (0-6)
SELECT EXTRACT(DOY FROM hire_date) FROM employees;    -- Day of year
SELECT EXTRACT(WEEK FROM hire_date) FROM employees;
SELECT EXTRACT(QUARTER FROM hire_date) FROM employees;
SELECT EXTRACT(HOUR FROM created_at) FROM logs;
SELECT EXTRACT(MINUTE FROM created_at) FROM logs;
SELECT EXTRACT(SECOND FROM created_at) FROM logs;
SELECT EXTRACT(EPOCH FROM created_at) FROM logs;      -- Unix timestamp

-- DATE_PART (PostgreSQL)
SELECT DATE_PART('year', hire_date) FROM employees;

-- TO_CHAR for formatting
SELECT TO_CHAR(hire_date, 'YYYY-MM-DD') FROM employees;
SELECT TO_CHAR(hire_date, 'Month DD, YYYY') FROM employees;
SELECT TO_CHAR(created_at, 'HH24:MI:SS') FROM logs;
SELECT TO_CHAR(created_at, 'Day') FROM logs;          -- Weekday name
```

### Date Arithmetic
```sql
-- Add/subtract intervals
SELECT hire_date + INTERVAL '1 year' FROM employees;
SELECT hire_date - INTERVAL '30 days' FROM employees;
SELECT created_at + INTERVAL '2 hours' FROM logs;

-- Date difference
SELECT CURRENT_DATE - hire_date AS days_employed FROM employees;
SELECT AGE(CURRENT_DATE, hire_date) FROM employees;   -- PostgreSQL
SELECT DATEDIFF(CURRENT_DATE, hire_date) FROM employees;  -- MySQL

-- Date truncation
SELECT DATE_TRUNC('month', hire_date) FROM employees;  -- First of month
SELECT DATE_TRUNC('year', hire_date) FROM employees;   -- First of year
SELECT DATE_TRUNC('week', hire_date) FROM employees;   -- Start of week
SELECT DATE_TRUNC('hour', created_at) FROM logs;

-- Constructing dates
SELECT MAKE_DATE(2024, 1, 15);                        -- PostgreSQL
SELECT DATE('2024-01-15');                            -- Parse string
SELECT TO_DATE('15-01-2024', 'DD-MM-YYYY');          -- Parse with format
SELECT TO_TIMESTAMP('2024-01-15 14:30:00', 'YYYY-MM-DD HH24:MI:SS');
```

### Time Zones
```sql
-- Convert time zones
SELECT created_at AT TIME ZONE 'UTC' FROM logs;
SELECT created_at AT TIME ZONE 'America/New_York' FROM logs;

-- Current timezone
SELECT current_setting('TIMEZONE');                   -- PostgreSQL

-- Set session timezone
SET TIME ZONE 'America/Los_Angeles';                  -- PostgreSQL

-- Time zone aware timestamps
SELECT CURRENT_TIMESTAMP AT TIME ZONE 'UTC';
```

---

## Conditional Logic

### CASE Expressions
```sql
-- Simple CASE
SELECT 
    first_name,
    CASE department_id
        WHEN 1 THEN 'Engineering'
        WHEN 2 THEN 'Sales'
        WHEN 3 THEN 'Marketing'
        ELSE 'Other'
    END AS department
FROM employees;

-- Searched CASE
SELECT 
    first_name,
    salary,
    CASE 
        WHEN salary >= 100000 THEN 'High'
        WHEN salary >= 70000 THEN 'Medium'
        WHEN salary >= 40000 THEN 'Low'
        ELSE 'Entry'
    END AS salary_band
FROM employees;

-- CASE in ORDER BY
SELECT * FROM employees
ORDER BY CASE 
    WHEN status = 'active' THEN 1
    WHEN status = 'pending' THEN 2
    ELSE 3
END;

-- CASE in aggregate
SELECT 
    department_id,
    SUM(CASE WHEN gender = 'M' THEN 1 ELSE 0 END) AS male_count,
    SUM(CASE WHEN gender = 'F' THEN 1 ELSE 0 END) AS female_count
FROM employees
GROUP BY department_id;

-- CASE with NULL
SELECT 
    CASE WHEN manager_id IS NULL THEN 'Top Level' ELSE 'Has Manager' END
FROM employees;
```

### COALESCE and NULLIF
```sql
-- COALESCE (first non-null value)
SELECT COALESCE(phone, mobile, 'No phone') FROM contacts;
SELECT COALESCE(nickname, first_name) AS display_name FROM users;

-- NULLIF (return NULL if values equal)
SELECT NULLIF(value1, value2) FROM data;  -- NULL if value1 = value2
SELECT 100 / NULLIF(divisor, 0) FROM data;  -- Avoid division by zero

-- Combined usage
SELECT COALESCE(NULLIF(notes, ''), 'No notes') FROM tasks;
```

### IIF and Other Conditionals
```sql
-- IIF (SQL Server, some databases)
SELECT IIF(salary > 100000, 'High', 'Normal') FROM employees;

-- GREATEST and LEAST
SELECT GREATEST(val1, val2, val3) FROM data;
SELECT LEAST(val1, val2, val3) FROM data;

-- DECODE (Oracle, some databases)
SELECT DECODE(status, 'A', 'Active', 'I', 'Inactive', 'Unknown') FROM users;
```

---

## Indexes & Performance

### Index Management
```sql
-- Create index
CREATE INDEX idx_employees_email ON employees(email);
CREATE INDEX idx_employees_name ON employees(last_name, first_name);

-- Unique index
CREATE UNIQUE INDEX idx_employees_email_unique ON employees(email);

-- Partial index (PostgreSQL)
CREATE INDEX idx_active_employees ON employees(email) WHERE is_active = TRUE;

-- Expression index
CREATE INDEX idx_employees_lower_email ON employees(LOWER(email));

-- Covering index (include non-key columns)
CREATE INDEX idx_employees_dept_salary 
ON employees(department_id) INCLUDE (salary, first_name);

-- Drop index
DROP INDEX idx_employees_email;
DROP INDEX IF EXISTS idx_employees_email;

-- Rename index (PostgreSQL)
ALTER INDEX idx_old_name RENAME TO idx_new_name;

-- Rebuild index (PostgreSQL)
REINDEX INDEX idx_employees_email;
REINDEX TABLE employees;

-- Concurrent index creation (PostgreSQL, no locks)
CREATE INDEX CONCURRENTLY idx_employees_email ON employees(email);
```

### Query Analysis
```sql
-- EXPLAIN (show query plan)
EXPLAIN SELECT * FROM employees WHERE department_id = 5;

-- EXPLAIN ANALYZE (execute and show actual times)
EXPLAIN ANALYZE SELECT * FROM employees WHERE department_id = 5;

-- EXPLAIN with options (PostgreSQL)
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
SELECT * FROM employees WHERE department_id = 5;

-- Execution plan terminology:
-- Seq Scan: Full table scan
-- Index Scan: Using index
-- Index Only Scan: Index covers all needed columns
-- Bitmap Scan: Uses bitmap of matching rows
-- Hash Join/Merge Join/Nested Loop: Join strategies
```

### Performance Tips
```sql
-- 1. Select only needed columns
SELECT first_name, last_name FROM employees;  -- Not SELECT *

-- 2. Use WHERE to filter early
SELECT * FROM orders WHERE order_date >= '2024-01-01';

-- 3. Use indexes for WHERE, JOIN, ORDER BY columns
CREATE INDEX idx_orders_date ON orders(order_date);

-- 4. Avoid functions on indexed columns
-- BAD: WHERE YEAR(order_date) = 2024
-- GOOD: WHERE order_date >= '2024-01-01' AND order_date < '2025-01-01'

-- 5. Use EXISTS instead of IN for large subqueries
SELECT * FROM departments d
WHERE EXISTS (SELECT 1 FROM employees e WHERE e.department_id = d.id);

-- 6. Use LIMIT for large result sets
SELECT * FROM logs ORDER BY created_at DESC LIMIT 100;

-- 7. Batch large operations
DELETE FROM old_logs WHERE created_at < '2020-01-01' LIMIT 10000;

-- 8. Use appropriate data types
-- Use INT instead of VARCHAR for IDs
-- Use DATE instead of VARCHAR for dates

-- 9. Analyze and vacuum (PostgreSQL)
ANALYZE employees;
VACUUM ANALYZE employees;

-- 10. Update statistics (SQL Server)
UPDATE STATISTICS employees;
```

---

## Transactions & Constraints

### Transaction Control
```sql
-- Begin transaction
BEGIN;
-- Or: START TRANSACTION;

-- Commit transaction
COMMIT;

-- Rollback transaction
ROLLBACK;

-- Savepoints
BEGIN;
INSERT INTO accounts (id, balance) VALUES (1, 1000);
SAVEPOINT after_insert;
UPDATE accounts SET balance = 500 WHERE id = 1;
ROLLBACK TO after_insert;  -- Undo the update
COMMIT;

-- Transaction isolation levels
SET TRANSACTION ISOLATION LEVEL READ UNCOMMITTED;
SET TRANSACTION ISOLATION LEVEL READ COMMITTED;      -- Default for most
SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;
SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;
```

### Constraints
```sql
-- Primary key
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    -- Or: id INT PRIMARY KEY AUTO_INCREMENT
    name VARCHAR(100)
);

-- Add primary key to existing table
ALTER TABLE users ADD PRIMARY KEY (id);

-- Composite primary key
CREATE TABLE order_items (
    order_id INT,
    item_id INT,
    quantity INT,
    PRIMARY KEY (order_id, item_id)
);

-- Foreign key
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    customer_id INT REFERENCES customers(id),
    -- Or with explicit constraint
    customer_id INT,
    CONSTRAINT fk_customer FOREIGN KEY (customer_id) 
        REFERENCES customers(id) 
        ON DELETE CASCADE 
        ON UPDATE CASCADE
);

-- Foreign key actions
ON DELETE CASCADE      -- Delete child rows when parent deleted
ON DELETE SET NULL     -- Set to NULL when parent deleted
ON DELETE SET DEFAULT  -- Set to default when parent deleted
ON DELETE RESTRICT     -- Prevent deletion if children exist
ON DELETE NO ACTION    -- Same as RESTRICT (default)

-- Unique constraint
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE,
    -- Or named constraint
    email VARCHAR(255),
    CONSTRAINT uq_email UNIQUE (email)
);

-- Check constraint
CREATE TABLE employees (
    id SERIAL PRIMARY KEY,
    salary DECIMAL(10,2) CHECK (salary > 0),
    age INT,
    CONSTRAINT chk_age CHECK (age >= 18 AND age <= 120)
);

-- Not null constraint
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL
);

-- Default constraint
CREATE TABLE posts (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'draft'
);

-- Add/drop constraints
ALTER TABLE employees ADD CONSTRAINT chk_salary CHECK (salary > 0);
ALTER TABLE employees DROP CONSTRAINT chk_salary;
ALTER TABLE users ALTER COLUMN name SET NOT NULL;
ALTER TABLE users ALTER COLUMN name DROP NOT NULL;
```

### Locking
```sql
-- Table locks (PostgreSQL)
LOCK TABLE employees IN ACCESS EXCLUSIVE MODE;
LOCK TABLE employees IN SHARE MODE;

-- Row-level locks
SELECT * FROM accounts WHERE id = 1 FOR UPDATE;       -- Exclusive lock
SELECT * FROM accounts WHERE id = 1 FOR SHARE;        -- Shared lock
SELECT * FROM accounts WHERE id = 1 FOR NO KEY UPDATE;
SELECT * FROM accounts WHERE id = 1 FOR UPDATE NOWAIT;  -- Don't wait
SELECT * FROM accounts WHERE id = 1 FOR UPDATE SKIP LOCKED;  -- Skip locked rows

-- Advisory locks (PostgreSQL)
SELECT pg_advisory_lock(123);
SELECT pg_advisory_unlock(123);
SELECT pg_try_advisory_lock(123);  -- Non-blocking
```

---

## Advanced Techniques

### JSON Operations
```sql
-- JSON column type
CREATE TABLE events (
    id SERIAL PRIMARY KEY,
    data JSON,
    data_b JSONB  -- Binary JSON (PostgreSQL, faster queries)
);

-- Insert JSON
INSERT INTO events (data) VALUES ('{"name": "click", "count": 5}');

-- JSON access (PostgreSQL)
SELECT data->>'name' AS name FROM events;           -- Text extraction
SELECT data->'count' AS count FROM events;          -- JSON extraction
SELECT data#>>'{nested,key}' FROM events;          -- Nested path (text)
SELECT data#>'{nested,key}' FROM events;           -- Nested path (json)

-- JSON operators (PostgreSQL)
SELECT * FROM events WHERE data @> '{"name": "click"}';  -- Contains
SELECT * FROM events WHERE data ? 'name';                -- Has key
SELECT * FROM events WHERE data ?& array['name', 'count'];  -- Has all keys
SELECT * FROM events WHERE data ?| array['name', 'type'];   -- Has any key

-- JSON functions (PostgreSQL)
SELECT jsonb_array_elements(data->'items') FROM events;  -- Expand array
SELECT jsonb_object_keys(data) FROM events;              -- Get keys
SELECT jsonb_typeof(data->'count') FROM events;          -- Get type
SELECT jsonb_set(data, '{count}', '10') FROM events;     -- Set value
SELECT data || '{"new_key": "value"}' FROM events;       -- Merge

-- JSON aggregation
SELECT jsonb_agg(row_to_json(e)) FROM employees e;
SELECT jsonb_object_agg(name, value) FROM key_values;

-- JSON in MySQL
SELECT JSON_EXTRACT(data, '$.name') FROM events;
SELECT data->'$.name' FROM events;
SELECT JSON_UNQUOTE(data->'$.name') FROM events;
```

### Arrays (PostgreSQL)
```sql
-- Array column
CREATE TABLE posts (
    id SERIAL PRIMARY KEY,
    tags TEXT[]
);

-- Insert arrays
INSERT INTO posts (tags) VALUES (ARRAY['sql', 'database', 'tutorial']);
INSERT INTO posts (tags) VALUES ('{sql,database,tutorial}');  -- Alternative

-- Array access
SELECT tags[1] FROM posts;                  -- First element (1-indexed)
SELECT tags[1:2] FROM posts;               -- Slice (first 2 elements)

-- Array operators
SELECT * FROM posts WHERE tags @> ARRAY['sql'];  -- Contains
SELECT * FROM posts WHERE tags && ARRAY['sql', 'python'];  -- Overlap
SELECT * FROM posts WHERE 'sql' = ANY(tags);     -- Any element matches

-- Array functions
SELECT array_length(tags, 1) FROM posts;         -- Length
SELECT array_cat(tags, ARRAY['new']) FROM posts; -- Concatenate
SELECT array_append(tags, 'new') FROM posts;     -- Append
SELECT array_remove(tags, 'old') FROM posts;     -- Remove
SELECT unnest(tags) FROM posts;                  -- Expand to rows
SELECT array_agg(DISTINCT tag) FROM posts, unnest(tags) AS tag;  -- Aggregate

-- Array in WHERE
SELECT * FROM posts WHERE 'sql' = ANY(tags);
SELECT * FROM posts WHERE tags @> ARRAY['sql', 'database'];
```

### Full-Text Search
```sql
-- PostgreSQL full-text search
CREATE INDEX idx_posts_search ON posts USING GIN (to_tsvector('english', content));

SELECT * FROM posts
WHERE to_tsvector('english', content) @@ to_tsquery('database & performance');

-- Ranking results
SELECT 
    title,
    ts_rank(to_tsvector('english', content), to_tsquery('database')) AS rank
FROM posts
WHERE to_tsvector('english', content) @@ to_tsquery('database')
ORDER BY rank DESC;

-- Headline (snippets with highlighted terms)
SELECT 
    ts_headline('english', content, to_tsquery('database'),
        'StartSel=<b>, StopSel=</b>'
    ) AS snippet
FROM posts
WHERE to_tsvector('english', content) @@ to_tsquery('database');

-- MySQL full-text search
CREATE FULLTEXT INDEX idx_posts_content ON posts(content);

SELECT * FROM posts
WHERE MATCH(content) AGAINST('database performance' IN NATURAL LANGUAGE MODE);

SELECT *, MATCH(content) AGAINST('database') AS relevance
FROM posts
WHERE MATCH(content) AGAINST('database')
ORDER BY relevance DESC;
```

### Set Operations
```sql
-- UNION (removes duplicates)
SELECT city FROM customers
UNION
SELECT city FROM suppliers;

-- UNION ALL (keeps duplicates)
SELECT city FROM customers
UNION ALL
SELECT city FROM suppliers;

-- INTERSECT (common to both)
SELECT city FROM customers
INTERSECT
SELECT city FROM suppliers;

-- EXCEPT (in first but not second)
SELECT city FROM customers
EXCEPT
SELECT city FROM suppliers;

-- Combining with ORDER BY
(SELECT city, 'customer' AS source FROM customers)
UNION
(SELECT city, 'supplier' AS source FROM suppliers)
ORDER BY city;
```

---

## Pro Tips & Best Practices

### Query Writing
```sql
-- 1. Use meaningful aliases
SELECT 
    e.first_name,
    d.name AS department_name  -- Clear alias
FROM employees e
JOIN departments d ON e.department_id = d.id;

-- 2. Format for readability
SELECT 
    e.first_name,
    e.last_name,
    e.salary,
    d.name AS department
FROM employees e
JOIN departments d ON e.department_id = d.id
WHERE e.salary > 50000
  AND d.name IN ('Engineering', 'Sales')
ORDER BY e.salary DESC;

-- 3. Comment complex queries
-- Get top performers in each department
-- based on performance reviews from last year
WITH recent_reviews AS (
    SELECT *
    FROM performance_reviews
    WHERE review_date >= CURRENT_DATE - INTERVAL '1 year'
)
SELECT * FROM recent_reviews;

-- 4. Use CTEs for complex logic
WITH 
step1 AS (SELECT ...),
step2 AS (SELECT ... FROM step1),
step3 AS (SELECT ... FROM step2)
SELECT * FROM step3;

-- 5. Avoid SELECT * in production
SELECT id, name, email FROM users;  -- Explicit columns

-- 6. Use parameterized queries (prevent SQL injection)
-- In application code, never concatenate user input
-- Use prepared statements or parameterized queries
```

### Common Gotchas
```sql
-- 1. NULL comparisons
-- Wrong: WHERE column = NULL
-- Right: WHERE column IS NULL

-- 2. String vs numeric comparisons
-- Wrong: WHERE id = '123' (string comparison)
-- Right: WHERE id = 123 (numeric comparison)

-- 3. GROUP BY with non-aggregated columns
-- Wrong (in strict SQL mode):
SELECT department_id, first_name, AVG(salary)
FROM employees
GROUP BY department_id;

-- Right:
SELECT department_id, MIN(first_name), AVG(salary)
FROM employees
GROUP BY department_id;

-- 4. HAVING vs WHERE
-- WHERE: filters rows before grouping
-- HAVING: filters groups after grouping

-- 5. Division by zero
SELECT 100 / NULLIF(denominator, 0) FROM data;  -- Safe

-- 6. Date format consistency
-- Use ISO 8601: YYYY-MM-DD for portability

-- 7. OR in WHERE clauses can prevent index usage
-- Consider UNION instead for better performance
```

### Security Best Practices
```sql
-- 1. Use parameterized queries
-- Never concatenate user input into SQL strings

-- 2. Principle of least privilege
GRANT SELECT ON employees TO app_user;
GRANT SELECT, INSERT, UPDATE ON orders TO app_user;

-- 3. Avoid exposing error messages
-- Handle SQL errors in application layer

-- 4. Use views for restricted access
CREATE VIEW public_employee_info AS
SELECT id, first_name, last_name, department_id
FROM employees;
-- Hide salary and personal info

-- 5. Audit sensitive operations
CREATE TABLE audit_log (
    id SERIAL PRIMARY KEY,
    action TEXT,
    table_name TEXT,
    old_data JSONB,
    new_data JSONB,
    user_name TEXT DEFAULT CURRENT_USER,
    timestamp TIMESTAMP DEFAULT NOW()
);
```

---

**📚 Additional Resources:**
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [MySQL Documentation](https://dev.mysql.com/doc/)
- [SQLite Documentation](https://www.sqlite.org/docs.html)
- [SQL Tutorial (W3Schools)](https://www.w3schools.com/sql/)
- [Mode SQL Tutorial](https://mode.com/sql-tutorial/)

**🔗 Related Cheatsheets:**
- [Python Cheatsheet](PYTHON_CHEATSHEET.md)
- [Pandas Cheatsheet](PANDAS_CHEATSHEET.md)
- [PySpark Cheatsheet](PYSPARK_CHEATSHEET.md)

---
*Last Updated: January 8, 2026*
