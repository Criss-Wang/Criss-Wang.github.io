---
title: "A fundamental course for Data Engineering"
date: 2020-05-21
layout: single
author_profile: true
categories:
  - Data Mining
tags: 
  - Data Engineering
  - ETL
  - Data storage
excerpt: "All you need for understanding industrial-level data"
mathjax: "true"
---

## Overview
In this blog, I'll discuss about a variety of fundamental concepts in data processing. While remembering them is pointless (since the true data engineers learn by practice), I try to give a clear view of different aspects in handling the data, this includes:
1. Data Storage --> Where data is stored
2. Data ETL --> Extracting / Transforming / Loading
3. Data Properties --> Source types and information system types
4. Data Processing Types --> How different data are processed differently
5. Data Cleansing --> What if the data is not well-organized
6. Database Compliance --> ACID or BASE?

## Data Storage
1. Data lake: a centralized repository that allows you to store structured, semistructured, and unstructured data at any scale.
- Single source of truth
- Store any type of data, regardless of structure 
- Example: AWS Data Lake
  :  Harness the power of purpose-built analytic services for a wide range of use cases, such as interactive analysis, data processing using Apache Spark and Apache Hadoop, data warehousing, real-time analytics, operational analytics, dashboards, and visualizations.

2. Data warehouse: A data warehouse is a central repository of structured data from many data sources. This data is transformed, aggregated, and prepared for business reporting and analysis.
- ETL (Extract, Transform, Load) operations before stored into data warehouse
- Data is stored within the data warehouse using a schema.
	
3. Data Mart: A subset of data from a data warehouse is called a data mart. 
- Data marts only focus on one subject or functional area. 
- A Data Warehouse might contain all relevant sources for an enterprise, but a Data Mart might store only a single department’s sources. Because data marts are generally a copy of data already contained in a data warehouse, they are often fast and simple to implement.

4. Traditional data warehousing: pros and cons

    | Pros                          | Cons                           | 
    | --------                      | ------                         | 
    | Fast data retrieval           | Costly to implement            | 
    | Curated data sets             | Maintenance can be challenging | 
    | Centralized storage	        | Security concerns              | 
    | Better business intelligence  | Hard to scale to meet demand   |

5. Data Warehouse vs Data Lake

    | Factors       	            | Data warehouse                                   | Data Lake                                    |
    | --------                      | ------                                           | ----------                                   |
    | Data						    | Relational from transactional systems, operational databases, and line of business applications		   | Non-relational and relational from IoT devices, websites, mobile apps, social media, and corporate applications |
    | Schema						| Designed prior to implementation (schema-on-write)							| Written at the time of analysis (schema-on-read) |
    | Price/performance			    | Fastest query, higher cost storage						| Fast, low-cost storage |
    | Data quality				    | Highly curated data that serves as the central version of the truth	| Any data, which may or may not be curated (e.g., raw data) |
    | Users						    | Business analysts									| Data scientists, data developers, and BA
    | Analytics					    | Batch reporting, BI, and visualizations			| Machine learning, predictive analytics, data discovery, and profiling.
                                                                                        				
## ETL Basics
1. Extracting data

	There are four key areas you must plan for.
	1. You must identify "where" all of the source data resides. This may be data stored on-premises by your company but can also include data that is found online.
	
	2. You must carefully plan "when" the extraction will take place due to the potential impact of the copy process on the source system.
	
	3. You must plan for "where" the data will be stored during processing. This is generally referred to as a staging location.
	
	4. You must plan for "how often" the extraction must be repeated.
	
2. Transforming data

    This phase involves using a series of rules and algorithms to massage the data into its final form. Data cleansing also occurs during this part of the process.
	
	It can be basic or advanced: This could be replacing NULL values with a zero or replacing the word female with the letter F; Or applying business rules to the data to calculate new values. Filtering, complex join operations, aggregating rows, splitting columns
	
3. Loading data
   
	The planning steps you took in the transformation phase will dictate the form the final data store must take. This could be a database, data warehouse, or data lake. 


## Data Properties

1. Data Source:
   
	In each of these data sources, data is stored in a specific way. Some data sources use a schema to organize content and indexes to improve performance. Others organize data in a more flexible way and are called schemaless. Schemaless data sources still use indexes to improve performance.

2. Types of data source:
	- Structured data:
		- stored in a tabular format, often within a database management system (DBMS). 
		- Organized based on a relational data model
		- Defines and standardize data elements
		
		- The *downside* to structured data is its lack of flexibility: you must reconfigure the schema to allow for this new data, and you must account for all records that don’t have a value for this new field. 
		
	- Semistructured data (NoSQL)
		- Stored in the form of elements within a file. (CSV, JSON, XML, etc)
		- Organized based on elements and the attributes that define them. 
		- No pre-defined schemas. Semistructured data is considered 
		- Have a self-describing structure: Each element is a single instance of a thing, such as a conversation. The attributes within an element define the characteristics of that conversation. Each conversation element can track different attributes.
		- The trade-off is with analytics. It can be more difficult to analyze semistructured data when analysts cannot predict which attributes will be present in any given data set.
		
	- Unstructured data 
		- Stored in the form of files. 
		- This data doesn't conform to a predefined data model and isn't organized in a predefined manner. 
		- Can be text-heavy, photographs, audio recordings, or even videos. 
		- Need to be preprocessed to perform meaningful analysis.
		
3. Types of information systems
   
	There are two main ways—known as information systems—of organizing data within a relational database. The data can be organized to focus on the storage of transactions or the process of analyzing transactions.
	
	- Online transaction processing (OLTP) databases:
		- operational databases,  primary focus being on the speed of data entry
		- These databases are characterized by a large number of insert, update, and delete operations.
		- based on ensuring rapid data entry and updates. The effectiveness of an OLTP system is often measured by the number of transactions per second.
		
	- Online analytical processing (OLAP) databases:
		- data warehouses, primary focus being the speed of data retrieval through queries.
		- These databases are characterized by a relatively low number of write operations and the lack of update and delete operations
		- based on the types of queries and other analytics that will be performed using the data. The effectiveness of an OLAP system is often measured by the response time of query results.
		
	- OLTP VS OLAP:

        | Characteristic	| OLTP          										| OLAP                                  |
        | ----------------  | ----------------------------                          | ----------------------------          |    
        | Nature			| Constant transactions (queries/updates)				| Periodiclarge updates, complex queries|
        | Examples			| Accounting database, online retail transactions       | Reporting, decision support           |
        | Type				| Operational data										| Consolidated data                     |
        | Data retention	| Short-term (2-6 months)								| Long-term (2-5 years)                 |
        | Storage			| Gigabytes (GB)										| Terabytes (TB)/petabytes (PB)         |
        | Users				| Many													| Few                                   |
        | Protection		| Robust, constant data protection and fault tolerance	| Periodic protection                   |
	
## Processing Types
1. Categories and types:
   
	__By Collection__:
	- Batch: Velocity is very predictable with batch processing. It amounts to large bursts of data transfer at scheduled intervals.

	- Periodic: Velocity is less predictable with periodic processing. The loss of scheduled events can put a strain on systems and must be considered.

	- Near real-time: Velocity is a huge concern with near real-time processing. These systems require data to be processed within minutes of the initial collection of the data. This can put tremendous strain on the processing and analytics systems involved.

	- Real-time: Velocity is the paramount concern for real-time processing systems. Information cannot take minutes to process. It must be processed in seconds to be valid and maintain its usefulness.
	
	__By Processing__:
	- Batch and periodic: Once the data has been collected, processing can be done in a controlled environment. There is time to plan for the appropriate resources.

	- Near real-time and real-time: Collection of the data leads to an immediate need for processing. Depending on the complexity of the processing (cleansing, scrubbing, curation), this can slow down the velocity of the solution significantly. Plan accordingly.
	
2. Data acceleration
   
	Another key characteristic of velocity on data is data acceleration, which means the rate at which large collections of data can be ingested, processed, and analyzed. Data acceleration is not constant. It comes in bursts. Take Twitter as an example. Hashtags can become hugely popular and appear hundreds of times in just seconds, or slow down to one tag an hour. That's data acceleration in action. Your system must be able to efficiently handle the peak of hundreds of tags a second and the lows of one tag an hour. 
	
3. Attributes of batch and stream processing

	|				| Batch data processing   | Stream data processing |
    | ------------- | ------------------      | ----------             | 
	| Data scope	| over all or most of the data 				| over data within a rolling time window, or on just the most recent data record|
	| Data size		| Large batches of data						| Individual records/ micro batches consisting of a few records					|		
	| Latency       | Minutes to hours							| Seconds or milliseconds |
	| Analysis      | Complex analytics							| Simple response functions, aggregates, and rolling metrics |
	
4. Processing big data streams

	There are many reasons to use streaming data solutions. In a batch processing system, processing is always asynchronous, and the collection system and processing system are often grouped together. With streaming solutions, the collection system (producer) and the processing system (consumer) are always separate. Streaming data uses what are called data producers. Each of these producers can write their data to the same endpoint, allowing multiple streams of data to be combined into a single stream for processing. Another huge advantage is the ability to preserve client ordering of data and the ability to perform parallel consumption of data. This allows multiple users to work simultaneously on the same data.

## Data Cleansing
1. Curation: the action or process of selecting, organizing, and looking after the items in a collection.

2. Data integrity: the maintenance and assurance of the accuracy and consistency of data over its entire lifecycle.
   
	Different types of integrity:
	
	- Referential integrity: the process of ensuring that the constraints of table relationships are enforced.
	- Domain integrity: the process of ensuring that the data being entered into a field matches the data type defined for that field..
	- Entity integrity: the process of ensuring that the values stored within a field match the constraints defined for that field.
	
	Maintaining Integrity accross steps of a data lifecycle:
	- Creation phase: Ensure data accuracy. Mainly involves software audits/data generation audits/data
	- Aggregation phase: Ensure the metrics computed are well-defined. Bad practice such as poor naming of metrics
	- Storage phase: Ensure stable data are not changed and volatile data are only changed by authorized personels
	- Access phase: System should be read-only and audited regularly for anomalies in access pattern
	- Share pahse: The phase where veracity get truly examined
	- Archive phase: Security of the data is the most important factor. Ensure limited access and read-only

3. Data veracity: the degree to which data is accurate, precise, and trusted.

4. A few best practices to help you identify data integrity issues
	- Know what clean looks like

		: Before you do anything else, you must come to a consensus on what clean looks like. Some businesses deem clean data to be data in its raw format with business rules applied. Some businesses deem clean data as data that has been normalized, aggregated, and had value substitutions applied to regulate all entries. These are two very different understandings of clean. Be sure to know which one you are aiming for.
		
	- Know where the errors are coming from
		: As you find errors in the data, trace them back to their likely source. This will help you to predict workloads that will have integrity issues. 
		
	- Know what acceptable changes look like
		: From a purely data-centric view, entering a zero in an empty column may seem like an easy data cleansing decision to make, but beware the effects of this change.
		
	- Know if the original data has value
		: In some systems, the original data is no longer valuable once it has been transformed. However, in highly regulated data or highly volatile data, it is important that both the original data and the transformed data are maintained in the destination system.
		
5. Database Schemas

	A data schema is the set of metadata used by the database to organize data objects and enforce integrity constraints. The schema defines attributes of the database, providing descriptions of each object and how it interacts with other objects within the database. One or more schemas can reside on the same database.
	
	- Logical schemas: focus on the constraints to be applied to the data within the database. This includes the organization of tables, views, and integrity checks.
	
	- Physical schemas: focus on the actual storage of data on disk or in a cloud repository. These schemas include details on the files, indices, partitioned tables, clusters, and more.

6. Information Schemas:
	
    An information schema is a database of metadata that houses information on the data objects within a database.

    Given the proper permissions on the database, you can query the information schema to learn about the objects within the database. When queries are executed, this information is used to ensure the best optimization for the query. The information schema can also be used in maintenance of the database itself.
	
	
## Database Compliance
1. ACID:
	- ACID is an acronym for *Atomicity*, *Consistency*, *Isolation*, and *Durability*
       - Atomicity: Ensures that your transactions either completely succeed or completely fail. No one statement can succeed without the others
   	
       - Consistency: Ensures that all transactions provide valid data to the database.  If any single statement violates these checks, the whole transaction will be rolled back
   	
       - Isolation: Ensures that one transaction cannot interfere with another concurrent transaction
   	
       - Data durability: Ensures your changes actually stick. Once a transaction has successfully completed, durability ensures that the result of the transaction is permanent even in the event of a system failure. This means that all completed transactions that result in a new record or update to an existing record will be written to disk and not left in memory.
	- Mainly to ensure veracity in a structured database
	- The goal of an ACID-compliant database is to return the most recent version of all data and ensure that data entered into the system meets all rules and constraints that are assigned at all times. 
	
	
	
2. BASE:
	- BASE is an acronym for *Basically Available Soft state Eventually Consistent*.
       - Basically Available: allows for one instance to receive a change request and make that change available immediately. The system will always guarantee a response for every request. However, it is possible that the response may be a failure or stale data, if the change has not been replicated to all nodes.
       - Soft State: In a BASE system, there are allowances for partial consistency across distributed instances. For this reason, BASE systems are considered to be in a soft state, also known as a changeable state.
       - Eventually Consistency: The data will be eventually consistent. In other words, a change will eventually be made to every copy. However, the data will be available in whatever state it is during propagation of the change.
	- BASE supports data integrity in non-relational databases
	- This consistency is mostly concerned with the rapid availability of data
	- To ensure the data is highly available, changes to data are made available immediately on the instance where the change was made. However, it may take time for that change to be replicated across the fleet of instances. 
	
	
	
3. ACID vs BASE:
   
	| ACID      										| BASE                                  | 
    | -----------                                       | --------                              |
	| Strong consistency								| Weak consistency – stale data is OK   |
	| Isolation is key									| Availability is key                   |
	| Focus on committed results						| Best effort results                   |
	| Conservative (pessimistic) availability			| Aggressive (optimistic) availability  |