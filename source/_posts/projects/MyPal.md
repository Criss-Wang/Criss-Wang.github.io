---
date: 2019-05-08
updated: 2019-08-02
layout: post
title: "MyPal"
categories:
  - Projects
  - Node.js
  - React
  - MongoDB
  - Fullstack Web Dev
excerpt: "A Relationship Management System"
link: "/../../images/Projects/mypal1.png"
mathjax: true
toc: true
---

### **Introduction**

[MyPal](https://github.com/Criss-Wang/MyPal-App) is a web application for students in campus to manage the interpersonal connections online. It was built by me and [Ren Hao](https://github.com/mononokehime14) from National University of Singapore as part of the [Orbital Program](https://orbital.comp.nus.edu.sg/). We managed to design, implement, deploy and iterate our app during the summer of 2019 to improve our product\'s rating from Gemini (intermediate) to Apollo 11 (Advanced). [[**Certificate**](https://github.com/Criss-Wang/TeamRichardoMLu/blob/master/Orbital_Certificate.pdf)]  

Our initiative for building this app was to give college students the power to conveniently create, manage and safely store important details about the people they want to keep connected with in their ever-expanding social circle during their college life. Here is a glimpse of our [application](<https://mypal-app.heroku.com>)

<figure align="center">
    <img src="/../../images/Projects/mypal1.png" width="400px">
    <img src="/../../images/Projects/mypal2.png" width="400px">
    <img src="/../../images/Projects/mypal3.png" width="400px">
    <img src="/../../images/Projects/mypal4.png" width="400px">
</figure>

### **Major Features**

1. **\"Relationship Info\" creation template**

- Template to create a new info sheet about a person in one\'s social network
- Can be personlized by adding and removing elements in the template
- Automatic integration of information from the person\'s social media and contact book (through uploaded or online documents)

2. **Categorization and tagging**

- Allow tags and categories to be applied on each connection for fast group viewing and relationship management
- Associate links to enable the user to switch to the social media to message a specific person

3. **Social Network Visualization**

- Using network diagram to display various social circles the user is in （built with `D3.js`)
- Using Timeline to help the user recollect the events he/she had with certain people/group

4. **Weekly/Monthly Report**

- Automatically generate a summary about the activities recorded on the website this week
- Produce reminder about friends\' birthday and people they haven\'t got in touch with for a long time.

### **Tech & Methodology**

<div>
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/react/react-original.svg" width="40" height="40"/>&nbsp;
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/javascript/javascript-original.svg" width="40" height="40"/>&nbsp;
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/html5/html5-original.svg"  width="40" height="40"/>&nbsp;
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/css3/css3-original.svg" width="40" height="40"/>&nbsp;
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/express/express-original-wordmark.svg"  width="40" height="40"/>&nbsp;
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/graphql/graphql-plain.svg" width="40" height="40"/>&nbsp;
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/d3js/d3js-original.svg" width="40" height="40"/>&nbsp;
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/mongodb/mongodb-original.svg" width="40" height="40"/>&nbsp;
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/nodejs/nodejs-original.svg" width="40" height="40"/>&nbsp;
</div>

- Specifically MERN stack (MongoDB, Express.js, ReactJS, Node.js)
- REST/GraphQL APIs
