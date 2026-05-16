---
title: "Apache Spark: Only the Simple Answer"
date: 2020/04/13
categories:
  - Blogs
tags:
  - Spark
  - Data Engineering
excerpt: "A simple explanation of Apache Spark: distributed data processing, DataFrames, lazy evaluation, transformations, actions, partitioning, and common performance mistakes."
layout: post
toc: true
---

## Introduction

Apache Spark is a distributed processing engine for large-scale data workloads. It lets you process data across many machines using APIs in Python, SQL, Scala, Java, and R.

Use Spark when the data or computation is too large for one machine, or when the pipeline already lives in a distributed data platform.

## The Basic Model

Spark splits work across a cluster.

Key pieces:

- **Driver:** coordinates the job.
- **Executors:** run tasks.
- **Partitions:** chunks of data processed in parallel.
- **Transformations:** build a logical plan.
- **Actions:** trigger execution.

Spark is lazily evaluated. Transformations do not run immediately; Spark builds a plan and executes it when an action is called.

## DataFrames

Spark DataFrames are the main API for structured data.

```python
df = spark.read.parquet("s3://bucket/events/")

result = (
    df.filter(df.event_type == "purchase")
      .groupBy("country")
      .count()
)

result.show()
```

The `filter` and `groupBy` calls are transformations. The `show` call is an action.

## Transformations and Actions

Common transformations:

- `select`
- `filter`
- `withColumn`
- `groupBy`
- `join`
- `orderBy`

Common actions:

- `show`
- `count`
- `collect`
- `write`

Be careful with `collect()`. It brings data back to the driver and can crash the job if the dataset is large.

## Partitioning

Partitioning affects parallelism and shuffle cost.

Too few partitions:

- Not enough parallelism.
- Slow tasks.

Too many partitions:

- Scheduling overhead.
- Many small files.

Partition by fields commonly used for filtering, such as date, when it matches the workload.

## Shuffles

A shuffle moves data across the network. Joins, group-bys, and repartitions often cause shuffles.

Shuffles are expensive because they involve disk, network, and coordination. Many Spark performance problems are really shuffle problems.

Reduce shuffle cost by:

- Filtering early.
- Selecting only needed columns.
- Joining on well-partitioned keys.
- Broadcasting small tables.
- Avoiding unnecessary repartitioning.

## Common Mistakes

- Calling `collect()` on large data.
- Writing too many small files.
- Joining huge tables without understanding keys.
- Ignoring skewed keys.
- Recomputing the same expensive transformation.
- Using Python UDFs when built-in functions would work.

## When Not to Use Spark

Spark is not always the answer.

Do not use it when:

- Data fits comfortably on one machine.
- A database can answer the query directly.
- Low-latency request serving is required.
- Operational complexity outweighs benefits.

Spark is powerful, but it is a distributed system. Use it when you need distributed processing.
