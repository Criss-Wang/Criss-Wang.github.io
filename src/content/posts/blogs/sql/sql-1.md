---
title: "SQL: Pick Up the Basics Within a Day"
date: 2020/08/15
categories:
  - Blogs
tags:
  - SQL
  - Database
excerpt: "A concise SQL primer covering SELECT, filtering, joins, aggregation, subqueries, inserts, updates, deletes, and practical query habits."
layout: post
toc: true
---

## Introduction

SQL is the standard language for querying relational databases. You can learn the basics quickly, but writing reliable SQL requires care with joins, filters, nulls, and aggregation.

This post is a compact primer.

## Select Rows

```sql
SELECT id, name, created_at
FROM users;
```

Use `WHERE` to filter:

```sql
SELECT id, name
FROM users
WHERE country = 'US'
  AND created_at >= '2024-01-01';
```

## Sort and Limit

```sql
SELECT id, total_amount
FROM orders
ORDER BY total_amount DESC
LIMIT 10;
```

`ORDER BY` controls result order. Without it, databases do not guarantee row order.

## Aggregation

```sql
SELECT country, COUNT(*) AS user_count
FROM users
GROUP BY country
ORDER BY user_count DESC;
```

Use `HAVING` to filter after aggregation:

```sql
SELECT user_id, COUNT(*) AS order_count
FROM orders
GROUP BY user_id
HAVING COUNT(*) >= 3;
```

## Joins

Joins combine tables.

```sql
SELECT users.id, users.name, orders.id AS order_id
FROM users
JOIN orders
  ON users.id = orders.user_id;
```

Common join types:

- `INNER JOIN`: keep matching rows only.
- `LEFT JOIN`: keep all left rows and matching right rows.
- `RIGHT JOIN`: keep all right rows and matching left rows.
- `FULL OUTER JOIN`: keep rows from both sides.

Most bugs come from unexpected row multiplication. Always know the grain of each table before joining.

## Nulls

`NULL` means missing or unknown. It does not behave like an ordinary value.

```sql
SELECT *
FROM users
WHERE deleted_at IS NULL;
```

Use `IS NULL`, not `= NULL`.

## Subqueries and CTEs

Common table expressions make queries easier to read:

```sql
WITH recent_orders AS (
  SELECT *
  FROM orders
  WHERE created_at >= '2024-01-01'
)
SELECT user_id, COUNT(*) AS order_count
FROM recent_orders
GROUP BY user_id;
```

Use CTEs to name steps in a query.

## Insert, Update, Delete

```sql
INSERT INTO users (id, name, country)
VALUES (1, 'Ada', 'US');
```

```sql
UPDATE users
SET country = 'CA'
WHERE id = 1;
```

```sql
DELETE FROM users
WHERE id = 1;
```

Be careful with `UPDATE` and `DELETE`. Always check the `WHERE` clause.

## Practical Habits

- Start with a small `LIMIT`.
- Select only needed columns.
- Check row counts before and after joins.
- Use CTEs for readability.
- Avoid ambiguous column names.
- Be explicit about time zones.
- Understand table grain before aggregation.
- Use transactions for risky changes.

SQL is simple to start and deep in practice. The best way to improve is to read query plans, inspect intermediate results, and keep asking what one row represents.
