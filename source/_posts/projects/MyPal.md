---
date: 2022-08-20
layout: post
title: "MyPal"
categories:
  - Projects
excerpt: "A Relationship Management System"
mathjax: true
toc: true
---

### Introduction
[MyPal](https://github.com/Criss-Wang/MyPal-App) is a web application for students in campus to manage the interpersonal connections online. It was built by me and Ren Hao from National University of Singapore (who is now also my school mate at CMU). Our initiative for building this app was to give college students the power to conveniently create, manage and safely store important details about the people they want to keep connected with in their ever-expanding social circle during their college life. 

### Tech stack
Our app used MERN stack (MongoDB, Express.js, ReactJS, Node.js) for development. 

### Major features
1. **\"Relationship Info\" creation template**
  - Template to create a new info sheet about a person in one\'s social network
  - Can be personlized by adding and removing elements in the template
  - Automatic integration of information from the person\'s social media and contact book (through uploaded or online documents)
2. **Categorization and tagging**
  - Allow tags and categories to be applied on each connection for fast group viewing and relationship management
  - Associate links to enable the user to switch to the social media to message a specific person
3. **Social Network Visualization**
  - Using network diagram to display various social circles the user is in （built with `d3.js`)
  - Using Timeline to help the user recollect the events he/she had with certain people/group
4. **Weekly/Monthly Report**
  - Automatically generate a summary about the activities recorded on the website this week
  - Produce reminder about friends\' birthday and people they haven\'t got in touch with for a long time.