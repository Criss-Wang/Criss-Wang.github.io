---
title: "Database Intro"
date: 2020/07/01
categories:
  - Blogs
tags:
  - Data
  - Database
excerpt: "A compact introduction to databases, covering relational systems, NoSQL systems, transactions, schemas, indexes, and when each storage pattern fits."
layout: post
toc: true
---

## Introduction

A database stores data so applications can read, write, update, query, and protect it reliably. The right database depends on the shape of the data and the workload.

The most important question is not "SQL or NoSQL?" It is:

```text
What access patterns must this system support?
```

## Relational Databases

Relational databases store data in tables with rows and columns. They are a strong default when data has clear structure and relationships.

Examples:

- PostgreSQL.
- MySQL.
- SQLite.
- SQL Server.

Use relational databases when:

- Data has a stable schema.
- Transactions matter.
- Joins are useful.
- Consistency matters.
- Reporting and ad hoc queries are needed.

## NoSQL Databases

NoSQL systems relax some relational assumptions to support different access patterns.

Common types:

- **Document databases:** JSON-like documents, useful for flexible records.
- **Key-value stores:** fast lookup by key.
- **Wide-column stores:** large-scale distributed tables.
- **Graph databases:** relationships are the core data.
- **Time-series databases:** metrics and events over time.

NoSQL is not automatically more scalable. It is better when its data model matches the problem.

## Transactions

A transaction groups operations so they succeed or fail together.

The classic ACID properties are:

- **Atomicity:** all operations complete or none do.
- **Consistency:** constraints remain valid.
- **Isolation:** concurrent transactions do not corrupt each other.
- **Durability:** committed data survives failures.

Financial systems, order systems, and inventory systems usually need strong transaction guarantees.

## Schema

A schema defines the structure of data.

Strong schemas help:

- Validate data early.
- Prevent inconsistent records.
- Make queries predictable.
- Improve documentation.

Flexible schemas help when data changes often, but they move more validation responsibility into application code.

## Indexes

Indexes speed up reads by storing searchable structures for selected columns or fields.

They are not free:

- They consume storage.
- They slow writes.
- They need maintenance.
- Poor indexes can mislead query planning.

Index the queries that matter, not every column.

## Choosing a Database

Ask:

- What are the read and write patterns?
- Does the system need transactions?
- Is the schema stable?
- How large will the data become?
- What latency is required?
- Do queries need joins?
- Is horizontal scaling required?
- What failure modes are acceptable?

Start with the simplest database that satisfies the system requirements. Complexity is easy to add and hard to remove.
