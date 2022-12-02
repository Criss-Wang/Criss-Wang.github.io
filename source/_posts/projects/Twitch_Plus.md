---
date: 2022-09-20
layout: post
title: "Twitch+"
categories:
  - Projects
excerpt: "A Search & Recommendation Engine for Twitch Resources"
mathjax: true
toc: true
---

### **Introduction**
[Twitch+](https://twitch-plus.herokuapp.com/) is a web application to track favorite Twitch resources for users. I deployed it on both heroku (lightweight version) and on AWS (more scalable and stable). This is an ongoing project which I keep thinking of new featuers and iteratively update my work. Here are some features I have built

- Custom game search
- Multiple resource extraction (Stream/Video/Clip)
- Like/unlike & Favorite panel
- Recommendations & Hot games list

![](../../images/Projects/twitch%2B.png)

### **Tech & Methodology**
<div>
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/java/java-original.svg" width="40" height="40"/>&nbsp;
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/spring/spring-original-wordmark.svg"  width="40" height="40"/>&nbsp;
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/tomcat/tomcat-original.svg" width="40" height="40"/>&nbsp;
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/javascript/javascript-original.svg"  width="40" height="40"/>&nbsp;
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/html5/html5-original.svg" width="40" height="40"/>&nbsp;
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/css3/css3-original.svg"    width="40" height="40"/>&nbsp;
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/react/react-original.svg"  width="40" height="40"/>&nbsp;
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/mysql/mysql-original.svg"  width="40" height="40"/>&nbsp;
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/docker/docker-original.svg" width="40" height="40"/>&nbsp;
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/amazonwebservices/amazonwebservices-plain-wordmark.svg" width="40" height="40"/>&nbsp;
    <img src="https://svn.apache.org/repos/asf/comdev/project-logos/originals/maven.svg" width="100" height="60"/>
</div>

- REST APIs with Java servlets
- Twitch API
- MySQL query optimization
- Recommender system - Deep Factorization Machine 
- Amazon Web Service (EC2, RDS)
