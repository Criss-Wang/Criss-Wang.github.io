---
title: "SQL: Pick up the Basic within a day"
excerpt: "The best way to get the memory back is by looking at the code themselves"
date: 2020/08/15
updated: 2022/05/25
categories:
  - Blogs
tags: 
  - Database System
  - SQL
layout: post
mathjax: true
toc: true
---
### Overview
This blog is for people who have learnt SQL at some points of their study (just like me): We can quickly recap on various important concepts in SQL.
### Basic Concepts
1. Primary key:
	- Always unique for each row of the table, it must be NOT NULL (automatically set when `PRIMARY KEY` is specified)
	- Helps to identify each row even when row attributes are the same
	- A table must have and only have 1 primary key
	- Types:
        - Surrogate key : An artificial key that has no mapping to anything (or business value) in real world.
        - Natural key: A key that has mapping to real world thing: example: social security number/ NRIC / Passport number

2. composiite key: 2 column entries combined to form a key
	- Motivation: sometimes individuals of 2 entries cannot uniquely identify a row;
		
3. Foreign key:
	- Stores the primary key of a row in another database table
	- The foreign key\'s column name NOT necessary to coincide with the foreign table\'s primary key column name
	- A table can have more than 1 foreign key (or no foreign key at all)
	
4. advance concept:
	- **Q**: is it possible that TABLE A\'s foreign key is TABLE B\'s primary key and TABLE B\'s foreign key is TABLE A\'s primary key?

      **A**: Yes! cyclic dependency is valid in SQL.
      
      **Example**: `employee`\'s `emp_id` is `department`\'s `manager_id`; `department`\'s `branch_id` is `employee`\'s `department_id`.
	- **Q**: is it possible that TABLE A\'s foreign key relates to itself? 
			
      **A**: Yes! used to define relationships between rows within a table.
		
      **Example**: `employee`\'s `super_id` refers to a row in `employee`\'s table.
	
5. Data Types:
	1. <kbd>INT</kbd>: -- Whole number
	2. <kbd>DECIMAL(M,N)</kbd>: 	-- Decimal numbers - exact value, M digits, N after decimal point
	3. <kbd>VARCHAR(K)</kbd>:		-- Sring of text of length K
	4. <kbd>BLOB</kbd>:			-- Binary Large Object, stores large data
	5. <kbd>DATE</kbd>:				-- \'YYYY-MM-DD\'
	6. <kbd>TIMESTAMP</kbd>		-- \'YYYY-MM-DD HH:MM:SS\'

6. Difference between DROP and DELETE:
| DELETE      | DROP |
| ----------- | ----------- |
| Data Manipulation Language command      | Data Definition Language Command       |
| To remove tuples from a table   | To remove entire schema, table, domain or constraints from the database        |

### Basic Operations
0. Create Database
```sql
SHOW DATABASES;
CREATE DATABASE July_05;
USE July_05;
```
  Logical Query Processing (IMPT)
      - Step 1. <kbd>FROM (includes JOINS)</kbd> <br>
      - Step 2. <kbd>WHERE</kbd> <br>
      - Step 3. <kbd>GROUP BY</kbd> <br>
      - Step 4. <kbd>HAVING</kbd> <br>
      - Step 5. <kbd>SELECT</kbd> <br>
      - Step 6. <kbd>ORDER BY</kbd> <br>
  **CAUTION about column ordering**: columns evaluated at later steps must be created in earlier steps

1. Table Opeartions
```sql
CREATE TABLE student (
    student_id INT PRIMARY KEY,
    student_name VARCHAR(50),
    major VARCHAR(20),
    thr_id INT,
    -- can also remove the PRIMARY KEY above and add a line below
    -- PRIMARY KEY(student_id)
    FOREIGN KEY(thr_id) REFERENCES teacher(emp_id) ON DELETE SET NULL
);
-- Show all columns properties using the DESCRIBE keyword
DESCRIBE student;
-- Name          Null       Type
-- student_id               INT
-- name                     VARCHAR(50)
-- major                    VARCHAR(20)
```
Delete or modify a table
```sql
DROP TABLE student;
ALTER TABLE student ADD gpa DECIMAL(3,2);
ALTER TABLE student DROP COLUMN gpa;
ALTER TABLE student MODIFY COLUMN major TINYINT(1) UNSIGNED;
ALTER TABLE student ADD CONSTRAINT pk_id PRIMARY KEY (student_id);
ALTER TABLE student ADD CONSTRAINT fk_id FOREIGN KEY (thr_id) REFERENCES teacher(emp_id) ON DELETE SET NULL;
```
2. Row Insertion
```sql
-- Two ways of insertion
INSERT INTO student VALUES(2, 'Kate', 'Sociology');
INSERT INTO student(student_id, name) VALUES(3, 'Claire');
```
3. More properties of column
```sql
CREATE TABLE student (
    student_id INT AUTO_INCREMENT, -- id increase automaically if not specified
	student_id2 INT IDENTITY(1, 1) -- similar to AUTO_INCREMENT except
								   -- IDENTITY(seed, increment) enables one to self define the starting value (seed) and the increment amount (increment)
    name VARCHAR(50) NOT NULL, -- name value cannot be empty
    major VARCHAR(20) UNIQUE, -- each row's major value must be unique across the table
	info VARCHAR(10) DEFAULT 'undecided', --info has 'undecided' as default value
    PRIMARY KEY(student_id)
);
```
4. Update the table
```sql
-- Modify the content
UPDATE student
SET major = 'Biochemistry', name = 'What'
WHERE major = 'Biology' or major = 'Chemistry'; -- if no WHERE is applied, the set applies to all
-- Delete entries
DELETE FROM student
WHERE student_id = 5;
```
5. <kbd>SELECT</kbd> keyword
  ```sql
    -- partial selection
    SELECT student.name, student.major
    FROM student
    ORDER BY major, name DESC; -- by default ascending order, DESC change to descending

    -- or both major and name descending by 
    SELECT name, major 
    FROM student
    ORDER BY major DESC, student_id DESC
    
    - other optional selection technique
    FROM student
    ...
    WHERE major = 'chemistry' OR major = 'Bio';
    ...
    WHERE name IN ('kate', 'Claire', 'Jack'); -- the use of IN keyword
    ...
    WHERE birth_day BETWEEN '1970-01-01' AND '1975-01-01';
    ...
    WHERE (birth_day >= '1970-01-01' AND sex = 'F') OR salary > 80000;
    ...
    LIMIT 2 OFFSET 1;
    ...
    
    SELECT TOP(100) -- select the 100 rows in the front
    SELECT ... INTO samples -- select those columns into the "sample" table
  ```
6. comparison keyword
```sql
<, > , <=, >=, =, <> (means not equal to), AND, OR, ANY, ALL
```
7. Functions to call
```sql
SELECT COUNT(sex), sex
SELECT AVG(salary)
SELECT SUM(salary)
FROM employee
WHERE sex = 'F' AND birth_date > '1971-01-01';
GROUP BY sex;
```
8. Wildcard

```sql
-- It is often used to find the string containing certain characters;
SELECT *
FROM client
WHERE client_name LIKE '%LLC';
```
9. <kbd>UNION</kbd>
- Motivation: row combine (fixed columns)
- Used to combine the multiple select statement into 1;
- Vertical join (add rows of the latter `SELECT` below the rows of former `SELECT`)
- **Warning**: each entry within the same column must have the same data-type
  ```sql
  SELECT client.client_name AS Non_Employee_Entities, client.branch_id AS Branch_ID
  -- here the renaming using AS is very important to make the unioned row's column more logical
  -- e.g the client.branch_id and branch_supplier.branch_id unioned to be Branch_ID and branch_id separately in the table returned
  FROM client
  UNION
  SELECT branch_supplier.supplier_name, branch_supplier.branch_id
  FROM branch_supplier;
  ```
  ```sql
  SELECT * FROM
  (
      (SELECT CITY, LENGTH(CITY)
      FROM STATION
      WHERE LENGTH(CITY) = (SELECT MIN(LENGTH(CITY)) FROM STATION)
      ORDER BY CITY)
      UNION
      (SELECT CITY, LENGTH(CITY)
      FROM STATION    
      WHERE LENGTH(CITY) = (SELECT MAX(LENGTH(CITY)) FROM STATION)
      ORDER BY CITY)
  ) AS K -- note the use of AS is MUST included
  ORDER BY CITY
  ```
10.  <kbd>JOIN</kbd>
- Motivation: column combine (fixed row)
- The second table is used as an auxilary table for additional column entries in the first table
```sql
SELECT employee.emp_id, employee.first_name, branch.branch_name
FROM employee
JOIN branch    -- LEFT JOIN, RIGHT JOIN
ON employee.emp_id = branch.mgr_id;
```
```sql
SELECT * FROM
(
	(SELECT 
		*
	 FROM STATION AS P
	 ORDER BY LENGTH(P.CITY) DESC
	) AS A
	 LEFT JOIN
	 (SELECT 
		*
	  FROM STATION AS K
	  ORDER BY LENGTH(K.CITY) DESC
	) AS B
	ON A.ID = B.ID
)  -- here should not have AS 
ORDER BY A.CITY -- Must specify A or ambiguous warning
```
  Different types of join:
  - <kbd>INNER JOIN</kbd>: the usual type of JOIN;
  Only those rows that match the ON criteria in both tables will be included and joined
  - <kbd>LEFT JOIN</kbd>:
  All those rows in the left table are included but rows in the right table are included only when they match the `ON` criteria
  - <kbd>RIGHT JOIN</kbd>:
  the symmetric idea with <kbd>LEFT JOIN</kbd>
  - <kbd>OUTER JOIN</kbd>:
  All the rows in both tables are included (empty columns in the resultant table rows are treated with NULL)
11. Nested query
```sql
SELECT employee.first_name, employee.last_name
FROM employee
WHERE employee.emp_id IN (SELECT works_with.emp_id
                          FROM works_with
                          WHERE works_with.total_sales > 50000);
```
12. ON DELETE
```sql
ON DELETE SET NULL -- set the foreign key to null if the primary key which the foreign key refers to gets deleted
ON DELETE CASCADE -- delete the entire row if the primary key gets deleted, especially important if set null cannot be done (i.e the foreign key cannot be set to null)
```
13. Trigger test
    ```sql
      CREATE TABLE trigger_test (
          message VARCHAR(100)
      );
      -- the following code needs to be manually typed in mySQL code
      DELIMITER $$ -- change the delimiter to $$
      CREATE
          TRIGGER my_trigger BEFORE INSERT
          ON employee
          FOR EACH ROW BEGIN
              INSERT INTO trigger_test VALUES('added new employee'); -- note the use of ; delimiter here
          END$$ -- we need to use the $$ as delimiter which is declared in line 168
      DELIMITER ; -- change the delimiter back to ;

      -- Conditional trigger_test
      DELIMITER $$
      CREATE
          TRIGGER my_trigger BEFORE INSERT -- can also be UPDATE, DELETE
          ON employee
          FOR EACH ROW BEGIN
              IF NEW.sex = 'M' THEN
                    INSERT INTO trigger_test VALUES('added male employee');
              ELSEIF NEW.sex = 'F' THEN
                    INSERT INTO trigger_test VALUES('added female');
              ELSE
                    INSERT INTO trigger_test VALUES('added other employee');
              END IF;
          END$$
      DELIMITER ;
      -- possible to drop the trigger case (done in client terminal):
      DROP TRIGGER my_trigger
    ```
14. CTE: Common Table Expression
```sql
	WITH Number -- here Number is the name of the CTE, can be anything 
	AS
	(
	SELECT
	 CustomerId
	 , NTILE(1000) OVER(ORDER BY CustomerId) AS N
	FROM dbo.Customers
	)
	,
	TopCustomer -- here we define the second CTE here, notice the comma "," above, indicates that the WITH keyword is still effective
	AS
	(
	SELECT  
	 MAX(CustomerId) AS CustId
	FROM Number
	GROUP BY N
	)
	
	SELECT  -- this SELECT is together with the CTE Expression, not separate query
	 C2.*
	INTO dbo.CustomersSample 
	FROM TopCustomer AS C1
	INNER JOIN dbo.Customers AS C2
	 ON C1.CustId = C2.CustomerId

	SELECT * FROM dbo.CustomersSample -- with the above cte method, we created a randomized sample in the dbo.customers table
```
15. Functions and procedures
    - Procedure Creation and Execution
      ```sql
        DELIMITER $$
        CREATE PROCEDURE FizzBuzz()
        BEGIN
          DECLARE N INT DEFAULT 1;
            WHILE N <= 100 DO
            SET N = N + 1;
            END WHILE;
        END$$
        DELIMITER ;

        CALL FizzBuzz();
      ```
    - Function Creation
        ```sql
        DELIMITER $$

        CREATE FUNCTION multi( -- if function alrea exists, CREATE is changed to ALTER
          n INT
          , m INT
        ) 
        RETURNS INT
        DETERMINISTIC
        BEGIN
          DECLARE result INT;
          SET result = m * n;
          RETURN result;
        END$$
        DELIMITER ;

        SELECT your_db_name.multi(2,3) AS result;
      ```
    - Check if the function exists
        ```sql
        SHOW FUNCTION STATUS WHERE db = 'your database name';
        ```

### String and numeric operations on values
String operations
```sql
SELECt 
	UPPER(email) up
    , LOWER(last_name) low
    , CONCAT(first_name, ' ', last_name) full_name
    , LENGTH(email) email_len
    , CONCAT_WS(' | ', first_name, last_name) full_name_with_separator
    , TRIM(' hello ') AS trimmed
    , RIGHT(email, 3) AS right_three 
    , LPAD(customer_id, 5, '000') AS left_zero_padding
    , FORMAT(address_id, 3) AS formated_3_float_point
FROM customer
LIMIT 10;
```
Regex Matching[1]
```sql
SELECT CONCAT(first_name, ' ', last_name) FROM customer
WHERE last_name ~ '^[^aeiou]' AND last_name ~* '[aeiou]$'
ORDER BY right(first_name, 2);
-- ~ : Case-sensitive, compares two statements, returns true if the first string is contained in the second
-- ~* : Case-insensitive, compares two statements, returns true if the first string is contained in the second
-- !~ : Case-sensitive, compares two statements, returns false if the first string is contained in the second
-- !~* : Case-insensitive, compares two statements, return false if the first string is contained in the second
```
[1]:https://dataschool.com/how-to-teach-people-sql/how-regex-works-in-sql/
Numeric Functions
```sql
SELECT
	RAND() AS rand_num
    , ROUND(RAND() * 10, 2) AS rand_round_2_decimal
    , CEIL(RAND()) AS num_ceil
    , FLOOR(RAND()) as num_floor
    , RADIANS(180) AS pi_from_radian
    , DEGREES(3.141592653589793) AS pi_from_degree
    , ABS(-3) AS absolute_val
    , POWER(CUSTOMER_ID, 2) AS id_square
    , DATEDIFF(shop_date.date, return_date.date) AS usage_period
    , CONV(CUSTOMER_ID, 10, 16) AS to_hex
    , IFNULL(potential_NUll_column, 0) AS replacing_null_with_zero
FROM customer
LIMIT 10;
```
```sql
-- A Blunder
SELECT REPLACE(amount, 0, 1)
FROM payment
LIMIT 5;
```

### Some Advanced operations: Window functions
1. OVER clause determines window (the set of rows to operate on)
2. PARTITION BY splits the result set into partitions on which the window function is applied
3. Functions Available:
	- Aggregate - <kbd>COUNT, SUM, MIN, MAX, AVG</kbd>
	- Ranking - <kbd>ROW_NUMBER, RANK, DENSE_RANK, NTILE</kbd>
	- Offset - <kbd>FIRST_VALUE, LAST_VALUE, LEAD, LAG</kbd>
	- Statistical - <kbd>PERCENT_RANK, CUME_DIST, PERCENTILE_CONT, PERCENTILE_DIST</kbd>
4. Windows Functions also have FRAMES
	- ROWS
	- RANGE

#### 1. Demo on <kbd>PARTITION BY</kbd>
```sql
-- the non-window function way
WITH CTE
AS
(
SELECT
  Sales_Id
  , SUM(Line_Total) AS Total
FROM Sales_Details
GROUP BY Sales_Id
);

SELECT * FROM CTE AS A
INNER JOIN Sales_Details AS B
  ON A.Sales_Id = B.Sales_Id;	
  
-- the window function way
SELECT
  Sales_Id
  , Sales_Date
  , Item
  , Price
  , Quantity
  , Line_Total
  , COUNT(Line_Total) OVER(PARTITION BY Sales_Id) AS Line_Count
  , SUM(Line_Total) OVER(PARTITION BY Sales_Id) AS Sales_Total
  , SUM(Line_Total) OVER(PARTITION BY Sales_Date) AS Daily_Total
  , SUM(Line_Total) OVER() AS Total
FROM Sales_Details
ORDER BY Sales_Total;
```

#### 2. On Ranking Functions

Ranking functions are available as part of Window Functions:
  - <kbd>ROW_NUMBER()</kbd> unique incrementing integers

  - <kbd>RANK()</kbd> same rank for same values, but keep the counting rolling $\implies$ 1, 1 (duplicate), 3, 4, 5

  - <kbd>DENSE_RANK()</kbd>: same rank for same values, but only increase rank by 1 when values change $\implies$ 1, 1 (duplicate), 2, 3, 4
  - <kbd>RANK()</kbd> vs <kbd>DENSE_RANK()</kbd>: <kbd>RANK()</kbd> will have rows with identical rank/ gaps in rank if we get tied values
  - <kbd>NTILE(N)</kbd> assigns tile number based on the number of tiles required, just assign each row with a value from 0 - N ,in increase order
    - Example: 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, .... ,N, N, N, N
    - Usage: for partitioning/selective sampling of the data

```sql
SELECT   -- note that here we use ORDER BY instead of PARTITION BY because order/rank is sort of important rather than the fixed set of value
  Sales_Id
  , Sales_Total
  , ROW_NUMBER() OVER(ORDER BY Sales_Total DESC) AS rownum 
  , RANK() OVER(ORDER BY Sales_Total DESC) AS rnk
  , DENSE_RANK() OVER(ORDER BY Sales_Total DESC) AS dense
  , NTILE(3) OVER(ORDER BY Sales_Total DESC) AS ntle
FROM dbo.Sales_2

SELECT   -- This is the modified way, we rank individual set of rows by adding on the PARTITION BY 
  Sales_Id
  , Sales_Cust_Id
  , Sales_Total
  , ROW_NUMBER() OVER(PARTITION BY Sales_Cust_Id ORDER BY Sales_Total DESC) AS rownum 
  , RANK() OVER(PARTITION BY Sales_Cust_Id ORDER BY Sales_Total DESC) AS rnk
  , DENSE_RANK() OVER(PARTITION BY Sales_Cust_Id ORDER BY Sales_Total DESC) AS dense
  , NTILE(3) OVER(PARTITION BY Sales_Cust_Id ORDER BY Sales_Total DESC) AS ntle
FROM dbo.Sales_2
ORDER BY Sales_Cust_Id
```

#### 3. <kbd>GROUP BY</kbd>
```sql	
SELECT
  Sales_Cust_Id
  , SUM(Sales_Total) AS Total
  , RANK() OVER(ORDER BY SUM(Sales_Total) DESC) AS rnk -- note that we used SUM(Sales_Total) not Sales_Total or Total because we need the order of SUM(Sales_Total) for each customer and Total is not defined well
  , DENSE_RANK() OVER(ORDER BY SUM(Sales_Total) DESC) AS dnse
FROM dbo.Sales_2
WHERE Sales_Date >= '2019-03-01'
GROUP BY Sales_Cust_Id
ORDER BY rnk

-- special OVER clause operation
SELECT
  Sales_Customer_Id
  , SUM(Sales_Amount) AS Cust_Total
  , SUM(SUM(Sales_Amount)) -- this declaration will be wrong as the system says cannot aggregate over another aggregation
  , SUM(SUM(Sales_Amount)) OVER(ORDER BY (SELECT NULL)) AS Grand_Total -- this is the proper way as the aggregation is down to the OVER Clause not the SUM(Sales_Amount) function
  , AVG(SUM(Sales_Amount)) OVER(ORDER BY (SELECT NULL)) AS Average_Cust_Total
  , CAST((SUM(Sales_Amount) / SUM(SUM(Sales_Amount)) OVER(ORDER BY (SELECT NULL))) * 100 AS DECIMAL(6,2)) AS Pct
FROM dbo.Sales
GROUP BY Sales_Customer_Id
```
#### 4. Window FRAMES
```sql
SELECT
  Sales_Id
  , Sales_Date
  , Sales_Total
  , SUM(Sales_Total) OVER(ORDER BY Sales_Date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS [Running Total]
  -- note that SUM is a window function here
  -- ROWS BETWEEN ... AND CURRENT ROW gives FRAME that is the set of rows from UNBOUNDED PRECEDING to this CUR ROW
  -- [Runnig Total] => need to put [] between a phrase with empty space " "
  , SUM(Sales_Total) OVER(ORDER BY Sales_Date ROWS BETWEEN k PRECEDING AND CURRENT ROW) AS [Running Total]
  -- this line has the FRAME only between the CURRENT ROW and the k rows before it; 
  , SUM(Sales_Total) OVER(ORDER BY Sales_Date ROWS UNBOUNDED PRECEDING) AS [Running Total]
  -- this line is a simplified version for BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
  , 
FROM dbo.Sales_2
WHERE Sales_Cust_Id = 3
ORDER BY Sales_Date

SELECT
  Sales_Id
  , Sales_Date
  , Sales_Total
  , SUM(Sales_Total) OVER(ORDER BY Sales_Date ROWS UNBOUNDED PRECEDING) AS [Running Total]
  , CAST(AVG(Sales_Total) OVER(PARTITION BY Sales_Cust_Id ORDER BY Sales_Date ROWS UNBOUNDED PRECEDING) AS DECIMAL(8, 2)) AS [Running Average]
  -- this line enables running average for individual customers for all of them
  -- CAST .. AS DECIMAL(8,2) reduces the resultant running average into 2 decimal points
FROM dbo.Sales_2
ORDER BY Sales_Date
```
#### 5. Lag and Lead
- Useful for trend analysis
- <kbd>LAG</kbd> - return the value from the previous row
- <kbd>LEAD</kbd> - return the value from the next row
- Format: 
```sql
LAG([Column], [Offset], [Value if NULL])
```
- Demo:
```sql
SELECT 
  Sales_Customer_Id
  , Sales_Date
  , LAG(Sales_Amount, 2, 0) OVER(PARTITION BY Sales_Customer_Id ORDER BY Sales_Date) AS PrevValue
  -- get the Sales_Amount 2 days before, if no value is in the entry 2 days before, set it to 0 (default is NULL)
  , Sales_Amount
  , LEAD(Sales_Amount, 2, 0) OVER(PARTITION BY Sales_Customer_Id ORDER BY Sales_Date) AS NextValue
  -- idea is the same, just change it to later
FROM dbo.Sales
```
#### 6. Rolling window
```sql
SELECT 
  *
  , SUM(SalesAmount) OVER(ORDER BY [Date] ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) AS Total 
  -- the Window FRAME method and SUM function together makings the window "rolling"
  , SUM(SalesAmount) OVER(ORDER BY [Date] ROWS BETWEEN CURRENT ROW AND 9 FOLLOWING) AS Forward
  -- we use FOLLOWING for the future rows
FROM #TempSales -- nothing fancy about # sign here
ORDER BY [Date] -- here [] is needed because Date itself is a SQL keyword
```

#### 7. Variable Specification
```sql
SET GLOBAL some_global_variable = 1;
SET @n = 10;
SELECT @n AS num;

SET@id = (SELECT payment_id FROM payment WHERE customer_id = 2 LIMIT 1);
SELECT @id AS new_id;

WITH cte AS
(
SELECT customer_id, COUNT(payment_id) cc
FROM payment p
GROUP BY customer_id
),
cnt AS (SELECT cc, COUNT(*) AS tcc, MAX(cc) OVER() AS mcc FROM cte GROUP BY cc)
SELECT *
FROM cte
INNER JOIN cnt ON cnt.cc = cte.cc AND (cnt.tcc = 1 OR cnt.cc = cnt.mcc)
ORDER BY cte.cc DESC, customer_id ASC;
```

### Conclusion
The above codes demonstrate the majorities of the SQL codes formats an engineer would ever need in its daily CRUD operations already. Thanks for reading!
