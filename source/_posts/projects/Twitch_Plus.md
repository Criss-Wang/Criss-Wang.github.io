---
date: 2022-08-30
updated: 2022-12-04
layout: post
title: "Twitch+"
categories:
  - Projects
  - Recommendation System
  - Full Stack Web Dev
  - Java
link: "../../images/Projects/twitch2.gif"
excerpt: "A Search & Recommendation Engine for Twitch Streaming Video Resources"
mathjax: true
toc: true
---

### **Introduction**

[Twitch+](https://twitch-plus.herokuapp.com/) is a web application to track favorite Twitch resources for users. I deployed it on both heroku (lightweight version) and on GCP (more scalable and stable). This is an ongoing project which I keep thinking of new featuers and iteratively update my work. Here are some features I have built. [[**code**](https://github.com/Criss-Wang/twitch-plus/tree/main)]

- Custom game search
- Multiple resource extraction (Stream/Video/Clip)
- Like/unlike & Favorite panel
- Recommendations & Hot games list

![](../../images/Projects/twitch1.gif)
![](../../images/Projects/twitch2.gif)
![](../../images/Projects/twitch3.gif)

### **Tech Stack & Methodology**

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
- **API lib**: Twitch API
- **Security** :JWT Authentication
- **Data**: MySQL query optimization, HikariCP connection pool, Hibernate lazy loading
- **AI**: Recommender system - Content-based filtering, Matrix factorization and Approximate Nearest Neighbor (ANN)
- **Cloud**: GCP (Compute engine, Cloud Storage, BigQuery)
