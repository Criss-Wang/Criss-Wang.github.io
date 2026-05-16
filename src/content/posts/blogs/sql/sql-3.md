---
title: "SQL: Going Into Applications With MySQL and MongoDB"
date: 2020/09/04
categories:
  - Blogs
tags:
  - SQL
  - Database
excerpt: "A practical comparison of relational and document databases in application development, with notes on schemas, queries, transactions, and data modeling."
layout: post
toc: true
---

## Introduction

Application databases should be chosen by access pattern. MySQL and MongoDB represent two common choices: a relational database and a document database.

Neither is universally better. Each fits different modeling and operational needs.

## MySQL

MySQL stores data in relational tables.

Use it when:

- Data has clear structure.
- Relationships matter.
- Joins are important.
- Transactions are important.
- Reporting queries are needed.
- Constraints should be enforced by the database.

Example:

```sql
SELECT users.id, users.email, orders.total_amount
FROM users
JOIN orders
  ON users.id = orders.user_id
WHERE orders.created_at >= '2024-01-01';
```

Relational modeling encourages normalized tables and explicit relationships.

## MongoDB

MongoDB stores document records, usually JSON-like documents.

Use it when:

- Records are naturally document-shaped.
- Schema varies across records.
- Reads usually need the whole document.
- The application benefits from embedding related data.
- Fast iteration on schema is valuable.

Example document:

```json
{
  "user_id": "u_123",
  "email": "ada@example.com",
  "settings": {
    "theme": "dark",
    "language": "en"
  }
}
```

Document modeling often embeds data that would be joined in a relational design.

## Data Modeling Difference

In relational databases, ask:

```text
What entities exist, and how do they relate?
```

In document databases, ask:

```text
What document shape does the application need to read and write?
```

If the app constantly reads a user and their settings together, a document can be convenient. If the app needs flexible joins across many entities, relational design is often cleaner.

## Transactions and Consistency

Relational databases are strong defaults for transactional workloads. MongoDB also supports transactions, but document databases are often best when the data model avoids frequent cross-document transactions.

If the system handles money, inventory, or strong consistency requirements, start with a relational design unless there is a clear reason not to.

## Choosing Between Them

Use MySQL when:

- The schema is structured.
- Joins are central.
- Data integrity constraints matter.
- SQL reporting is important.

Use MongoDB when:

- The data is document-shaped.
- Schema flexibility matters.
- Related data is usually read together.
- The app does not need many joins.

The best database is the one whose data model matches the application, not the one with the most fashionable label.
