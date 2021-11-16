---
title: "SQL: Going into Applications with MySQL and MongoDB"
date: 2020-09-04
layout: single
author_profile: true
categories:
  - Database
tags: 
  - SQL and NoSQL
excerpt: "Going back to the technical details of what and how for popular SQL and NoSQL dbms"
mathjax: "true"
---
## Introduction
This is a blog to note down some important concepts revolving MongoDB and MySQL, two of the most popular databases reprensentative of their respective domains: NoSQL and SQL. Many people know how to use these DBMS, but fail to appreciate their characteristics, when and why they are used in certain business solutions. I try to give as much high level comparisons as possible. This ensures that People can at least answer some basic interview questions when they look for a job using these tools.

## MongoDB
- A NoSQL database for high volumn data storage
- Dynamic schemas: creating entries without prior restriction of the data structure
- Represent data as of JSON documents and use JSON Query (JavaScript)
- Supports sharding and replication: it partitions data across multiple servers
### Sharding
The components of a Shard include:
1. A Shard – A MongoDB instance which holds the subset of the data. In production environments, **ALL** shards need to be part of replica sets.
2. Config server – A mongodb instance which holds **metadata** about the cluster, basically information about the various mongodb instances which will hold the shard data.
3. A Router – A mongodb instance responsible to re-directing the commands send by the client to the right servers.
### The benefits of NoSQL in MongoDB
- Schema Free: MongoDB has a pre-defined structure that can be defined and adhered to, but also, if you need different documents in a collection, it can have different structures.
- Scaled both **Horizontal** and **Vertical**: Improve system's processing power via
	- **Horizontal**: Adding more machines to expand the pool of resources
	- **Vertical**: Adding more power to a single machine (CPU/Storage)
- Optimized for WRITE performances

### The disadvantages of Non-SQL (without fixed schema) in MongoDB
- Does not support use of Foreign Keys
- Does not support optimization of JOIN operations 
- MongoDB is not strong ACID (Atomic, Consistency, Isolation & Durability)
- No Stored Procedure or functions, business logic must be implemented in the backend after data is retrieved (like Node.js). This may cause the operations to slow down.

## MySQL
- Relational Database (RDBMS)
- Represents data in tables and rows
- Predefine the Schema for the tables in the database
- Use SQL
- Supports Master-slave replication and master-master replication, i.e. copy data from one server to another 
- Optimized for high performance JOIN across multiple tables

### Disadvantages of MySQL (or traditional RDBMS)
- Scaled Only Vertically
- Transactions related to system catalog are not ACID compliant
- Sometimes a server crash can corrupt the system catalog
- Stored procedures are not cacheable
- MYSQL tables which is used for the procedure or trigger are most pre-locked.
- Risk of SQL injection attacks (if there is no predefined schema design, there is less of such a problem)

## Which to choose

| Characteristics  | MongDB                                                       | MySQL                                                                         |
| ---              | ---                                                          | ---                                                                           |
| Data nature      | A lot of unstructured data                                   | Mostly Structured data                                                        |
| Application      | Real-time analytics, content management, various mobile apps |Applications that requires multi-row transactions such as an accounting system |
| Service priority | Cloud Based                                                  | Security and ACID/BASE rules are very improtant                               |
| Data Volumn      | Large, high-speed volumn of data                             | Stable data flow                                                              |
