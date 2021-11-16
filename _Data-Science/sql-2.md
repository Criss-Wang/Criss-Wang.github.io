---
title: "SQL: Index and Optimization"
date: 2020-08-31
layout: single
author_profile: true
categories:
  - Database
tags: 
  - SQL
  - Query Optimization
excerpt: "A short guide on optimizing query performances"
mathjax: "true"
---
## Overview
To be honest, I'm not a pro-SQL programmer. I'm still on my journey to learn more about database and query optimization. In this blog I will just give whatever I've learnt about indexing and optimization and its mostly based on MySQL. Hope it helps!
## Guidelines
1. Single Sheet >>> Multiple Sheet
2. If multiple sheet is needed, Use <kbd>JOIN</kbd> well:
    - Small Sheet drive Large Sheet (left join in this case for e.g.)
    - Establish proper indexing
    - Don't JOIN too many sheets as well
3. Try best NOT to use **subquery** or **Cartesian Product**
4. Window Funtions can be very helpful

## Indexes
- Allow faster retrieval of data
- **Question**: Why don't we just create loads of indexes?
- **Ansewr**: There is a trade-off, if loads of indexes exists on a table then those indexes need to be updated or maintained. In this case, DML operations suffer.

### Index operations
```sql
-- show indices
SHOW INDEX FROM your_db_name.customer;

-- Add index
ALTER TABLE payment
ADD INDEX idx_pay (payment_id);  -- [index] can be appended by [unique] to ensure each index is unique

CREATE FULLTEXT INDEX idx_staff ON customer (email); -- [fulltext] only applicable to string data

-- Drop Index
DROP INDEX idx_pay ON payment
```
For the full list of operations, you may refer to the official documentation of MySQL[[^1]]

[^1]: https://dev.mysql.com/doc/refman/8.0/en/create-index.html

### Clustered Indexes 
- ALTER TABLE Permission
- WHen a table does not have a clustered index then the table is stored as a heap, if the table has a clustered index it is stored as a B-tree
- Data is stored in order of clustered index
- Only one clustered index can exists on one table
- Clustered indexes are effective on columns that consistent of unique increasing integers (like identity_set)
- When a primary key is created a unique clustered index is automatically created - this can be beneficial for queries that involve joins on this column.