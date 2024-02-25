---
title: "An overview of Big Data Analytics"
excerpt: "Know what data analysis is and for"
date: 2020/01/02
updated: 2022/01/30
categories:
  - Blogs
tags: 
  - Big Data
  - Data Analytics
  - Data Mining/Data Engineering
layout: post
mathjax: true
toc: true
---

### Overview
Data analysis is the process of compiling, processing, and analyzing data so that you can use it to make decisions. Let\'s start with the well-known 5 V\'s and proceed down to more concepts in Data analytics.

### The 5 V\'s
1. __Volume__ – data storage:
   - When businesses have more data than they are able to process and analyze, they have a volume problem.
   - There are three broad classifications of data source types:
      1. Structured data (10%): 
         : Organized and stored in the form of values that are grouped into rows and columns of a table.
      2. Semistructured data (10%): 
         : Often stored in a series of key-value pairs that are grouped into elements within a file.
      3. Unstructured data (80%): 
         : Not structured in a consistent way. Some data may have structure similar to semi-structured data but others may only contain metadata.
   

2. __Velocity__ - data processing:
   - When businesses need rapid insights from the data they are collecting, but the systems in place simply cannot meet the need, there\'s a velocity problem.
   - 2 types of processing:
      - Batch processing (Large burst of data): For initial insights and real-time feedback
      - Stream Processing (Tiny burst of data): For deep insights using complex analysis
      
3. __Variety__ - data structure and types
   - When your business becomes overwhelmed by the sheer number of data sources to analyze and you cannot find systems to perform the analytics, you know you have a variety problem.
   
4. __Veracity__ - data cleasing and transformation
   - When you have data that is ungoverned, coming from numerous, dissimilar systems and cannot curate the data in meaningful ways, you know you have a veracity problem
      
5. __Value__ - data insight and business intelligence
   - When you have massive volumes of data used to support a few golden insights, you may be missing the value of your data
		
	
### Data Analytics Definitions

1. Information analytics
   Information analytics is the process of analyzing information to find the value contained within it. This term is often synonymous with data analytics.
	

2. Operational analytics
   Operational analytics is a form of analytics that is used specifically to retrieve, analyze, and report on data for IT operations. This data includes system logs, security logs, complex IT infrastructure events and processes, user transactions, and even security threats.
	
	
### 5 forms of analysis:
1. Descriptive (Data Mining):
   <div class="buttons" style="padding-top:0">
   <button class="button is-small is-info is-light">Human Judgement</button>
   <button class="button  is-small is-info is-light">Insight</button>
   </div>
   - Requires highest amount of human effort and interpretation
   - Focus on "Whatdunit"
		
2. Diagnostic
	<div class="buttons" style="padding-top:0">
   <button class="button is-small is-info is-light">Human Judgement</button>
   <button class="button  is-small is-info is-light">Insight</button>
   </div>
   - Used to compare historic data with other data to find dependencies and patterns that lead to answers
   - Focus on "Whydunit"
		
3. Predictive
	<div class="buttons" style="padding-top:0">
   <button class="button is-small is-info is-light">Human Judgement</button>
   <button class="button  is-small is-info is-light">Insight</button>
   </div>
   - Focus on Future prediction
   - Uses the results of descriptive and diagnostics analysis to predict future events and trends
   - Accuracy highly dependent on the quality of data and stability of environment setup
		
4. Prescriptive
	<div class="buttons" style="padding-top:0">
   <button class="button is-small is-info is-light">Human Judgement</button>
   <button class="button  is-small is-info is-light">Insight</button>
   <button class="button  is-small is-info is-light">Decision</button>
   <button class="button  is-small is-info is-light">Action</button>
   </div>
   - Focus on Solution finding
   - Used to prescribe actions to taken given the data provided
   - Requires input from all other forms of analytics, combined with rules and contraints-based optimization to make relevant suggestion. 
   - This part is likely to be automated via machine learning
		
5. Cognitive
	<div class="buttons" style="padding-top:0">
   <button class="button is-small is-info is-light">Human Judgement</button>
   <button class="button  is-small is-info is-light">Insight</button>
   <button class="button  is-small is-info is-light">Decision</button>
   <button class="button  is-small is-info is-light">Action</button>
   </div>
   - Focus on recommended actions (not to be confused with \"solution\" to problems)
   - Try to mimic what the human brain does in problem solving
   - Generates hypothesis from the existing data, connections and contraints. Answers are provided in the form of recommendation and a confidence ranking
		
		
### Analytic services and velocity
1. Batch analytics:
	Typically involves querying large amounts of “cold” data. Batch analytics are implemented on large data sets to produce a large number of analytical results on a regular schedule.
	
2. Interactive analytics:
	Typically involves running complex queries across complex data sets at high speeds. This type of analytics is interactive in that it allows a user to query and see results right away. Batch analytics are generally run in the background, providing analytics in the form of scheduled report deliveries.
		
3. Stream analytics:
	Requires ingesting a sequence of data and incrementally updating metrics, reports, and summary statistics in response to each arriving data record. This method is best suited for real-time monitoring and response functions. Streaming data processing requires two layers: a storage layer and a processing layer. The storage layer needs to support record ordering and strong consistency to enable fast, inexpensive, and re-playable reads and writes of large streams of data. The processing layer is responsible for consuming data from the storage layer, running computations on that data, and then notifying the storage layer to delete data that is no longer needed. 
		
### Best practices for writing reports
1. Gather the data, facts, action items, and conclusions. Identify problems and formats
2. Identify the audience, expectations they have, and the proper method of delivery.
3. Identify the visualization styles and report style that will best fit the needs of the audience.
4. Create the reports and dashboards.