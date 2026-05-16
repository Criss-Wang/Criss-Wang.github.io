---
title: "SQL: Index and Optimization"
date: 2020/08/31
categories:
  - Blogs
tags:
  - SQL
  - Database
excerpt: "A concise guide to SQL indexes, query plans, filtering, joins, aggregation, and practical optimization habits."
layout: post
toc: true
---

## Introduction

SQL optimization is about helping the database do less work. Indexes, query structure, and data layout all affect performance.

The first rule: measure before optimizing.

## Indexes

An index is a data structure that helps the database find rows faster.

Index columns used in:

- Frequent filters.
- Join keys.
- Sorting.
- Uniqueness constraints.

Example:

```sql
CREATE INDEX idx_orders_user_id ON orders(user_id);
```

Indexes are not free. They take storage and slow writes because the index must be updated when data changes.

## Composite Indexes

A composite index covers multiple columns:

```sql
CREATE INDEX idx_orders_user_date ON orders(user_id, created_at);
```

Column order matters. This index helps queries filtering by `user_id`, and it may help queries filtering by `user_id` plus `created_at`.

## Query Plans

Use `EXPLAIN` to inspect how the database plans to run a query:

```sql
EXPLAIN
SELECT *
FROM orders
WHERE user_id = 42;
```

Look for:

- Full table scans on large tables.
- Expensive sorts.
- Join order.
- Index usage.
- Estimated vs actual row counts.

## Optimization Habits

Useful habits:

- Filter early.
- Select only needed columns.
- Avoid unnecessary `DISTINCT`.
- Avoid functions on indexed columns in filters when possible.
- Check join keys.
- Avoid accidental many-to-many joins.
- Use pagination carefully.
- Archive or partition old data when appropriate.

## Common Mistakes

- Adding indexes without checking query plans.
- Indexing every column.
- Joining tables at incompatible grains.
- Using `SELECT *` in production queries.
- Sorting huge result sets unnecessarily.
- Filtering on transformed values instead of stored values.

Optimization is a feedback loop: inspect the query, read the plan, change one thing, measure again.
