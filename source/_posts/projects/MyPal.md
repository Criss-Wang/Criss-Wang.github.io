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
[MyPal](https://github.com/Criss-Wang/MyPal-App) is a web application for students in campus to manage the interpersonal connections online. It was built by me and Ren Hao from National University of Singapore (who is now also my school mate at CMU). Our initiative for building this app was to give college students the power to conveniently create, manage and safely store important details about the people they want to keep connected with in their ever-expanding social circle during their college life. Here is a glimpse of our application](https://mypal-app.heroku.com)

<figure align="center">
    <img src="/../../images/Projects/mypal1.png" width="300px">
    <img src="/../../images/Projects/mypal2.png" width="30px">
    <img src="/../../images/Projects/mypal3.png" width="30px">
    <img src="/../../images/Projects/mypal4.png" width="30px">
</figure>

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


### **Tech & Methodology**
<div>
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" width="40" height="40"/>&nbsp;
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pytorch/pytorch-original.svg"  width="40" height="40"/>&nbsp;
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/jupyter/jupyter-original.svg" width="40" height="40"/>&nbsp;
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pandas/pandas-original.svg"  width="40" height="40"/>&nbsp;
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/docker/docker-original.svg" width="40" height="40"/>&nbsp;
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/amazonwebservices/amazonwebservices-plain-wordmark.svg" width="40" height="40"/>&nbsp;
</div>
- Specifically MERN stack (MongoDB, Express.js, ReactJS, Node.js)
- REST/GraphQL APIs