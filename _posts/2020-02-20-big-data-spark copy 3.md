---
title: "Apache Spark: Only the simple answer"
date: 2020-04-11
layout: single
author_profile: true
categories:
  - Big Data
tags: 
  - Software Development
  - Cloud Computing
  - Distributed Computing
excerpt: "Some conceptual understanding of spark and big data. Honestly, this just scratches the surface"
mathjax: "true"
---
## Overview
In this post, I'm just gonna discuss some fundational things I learned about big data with Apache Spark. Personally, I'm just a bit interested in this topic, and do not aim to really become a big data professional (not yet~). It does take tremendous effort to learn Spark well, not to mention the entire big data ecosystem. I'll update this post if I try out some new projects that really apply Spark and its APIs in a deep manner, but for now, let's just talk about some basics of Spark.

## 1. Apache Spark vs Hadoop MapReduce
- MapReduce is a __programming model__, Spark is a __processing framework__ 

|                  | Apache Spark                                           | MapReduce | 
| --------         | --------                                               | ------    | 
| Processing Type  | Process in batches and in real-time                    | Process in batches only| 
| Speed            | nearly 100x faster                                     | slower due to large scale data processing  | 
| Storage          | store data in RAM i.e. in-memeory (easier to retrieve) | Store in HDFS, longer time to retrieve| 
| Memory dependence| caching (for RDD) and in-memory data storage           | disk-dependent|

## Important Components of Spark Ecosystem
- Core components[MOST IMPORTANT]:
    - Spark Core (Caching, RDD, DataFrames, Datasets \|\| Transformations and Actions) 
      1. Memory management
      2. Fault recovery
      3. Task dispatching
      4. Scheduling and monitoring jobs
    - Spark SQL (Data Query)
      - Used for structured and semi-structured data processing
      - Usually used for in-memory processing
      - Top level:  DataFrame DSL(domain specific language) ; Spark SQl and HQL(Hive)
      - Level 2: DataFrame API
      - Level 3: Data Source API
      - Base Level: CSV + JSON + JDBC(Java Database connectivity) + etc storage/query
    - Spark Streaming (Stream data)
    - Spark MLlib (Machine Learnig toolkits)
    - GraphX (Graph Processing models)
- Langauge Support
    - Java
    - Python
    - Scala
    - R
- Spark Cluster Managers
	- Standalone mode: Default choice, run in FIFO order, and each application will try to use all available nodes
	- Apache YARN (Hadoop Integration): This is the resource manager of Hadoop, use this will help spark to connect to hdfs better
	- Kubernetes: For deployment, scaling and management of containerized applications

## RDD
- Resilient Distributed Datasets
- RDDs are immutable, fault-tolerant distributed collections of objects that can be operated in parallel (split into partition and executed on different nodes of a cluster)
- 2 major types of operations:
    - Transformation: map, filter, join, union, etc. Yield a new RDD containing the result
    - Action: reduce, count, first, etc. Return a value after running a computation on RDD
- Works in a similar style as java Stream

## How Spark runs applications with the help of its architecture
__START EXECUTION__
- In driver program
    - Spark applications runs as independent processes (i.e. split tasks) running across different machines
    - Spark sessions/context as the entry point of the application
    - Driver: Record the flow of the application
- Resource manager/Cluster manager (DAG Scheduler at the backend)
    - The driver program __request__ resources from the clusters 
    - Assign task to workers, one task per partition
    - Knows each step of the application for in-memory processing
- Worker node
    - processing slave (node manager) grands the request to usage of resources from resource manager
    - The request is called `Container`, within the `Container`, executor process is launched to apply tasks to its unit of work to the dataset in its partition and outputs a new partition dataset
    - because iterative algorithms apply operations repeated to the data, the benefit from caching datasets across iterations

__END EXECUTION__
- Results are sent back: worker node -> container -> manager -> driver program/disk
- The entire execution is lazily evaluated (transformations not evaluated until an action is called)

## What is a Parquet file and what are its advantages
- Parquet is a columnar format that is supported byh several data processing systems (default data type for spark)
- Advisable to use if not all fields/columns in the data are used
- Advantages
    - able to fetch specific columns for access
    - consumes less space
    - follow type-specific encoding
    - limited I/O operations

## What is shuffling in Spark? When does it occur?
- Shuffling is the process of redistributing data across partitions that may lead to data movement across executors
- Occurs while joining two tables or while performing `byKey` operations such as `GroupByKey` or `ReduceByKey`


## Notes on Big Data Learning Journey (for those who truly want a Big Data job and for my future )
To excel in the Big Data domain, you should master the following skills:
  1. Java & Scala --> Understand source code for related package/API development
  2. Linux --> Everyone should know about shell scripts, bash and linux commands
  3. Hadoop --> It's a broad topic, but first of all, the ability to read whatever source code for an API is a must
  4. Hive --> Know how to use it, understand how the SQL is converted in base code and how to optimize the query process or MapReduce/Spark Operations
  5. Spark --> The core developement process (But honestly, most of the time it is still SQL)
  6. Kafka --> High-volumn stream data processing; Good to use when you have high concurrency
  7. Flink --> Faster than Spark sometimes. However, you should not discard Spark. Learn based on what you need.
  8. HBase --> Know your database knowledge. Understand its fundamental knowledge
  9. Zookeeper --> Distributation cluster data coordination services; Know how to use, better to understand the basic
  10. YARN --> Cluster resources management; Know how to use
  11. Sqoop, Flume, Oozie/Azkaban --> Know how to use

### Different cluster managers 
1. Spark Standalone mode
    - by default, applications submitted to the standalone mode cluster will run in FIFO order, and each application will try to use all available nodes
2. Apache Mesos
    - an open sources project to manage computer clusters, and can also run Hadoop applications
3. YARN
    - Apache YARN is the cluster resource manager of Hadoop 2.
4. Kubernetes
    - an open-source system for automating deployment, scaling and management of containerized applications