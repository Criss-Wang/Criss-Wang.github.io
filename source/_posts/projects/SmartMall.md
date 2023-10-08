---
date: 2021-05-03
updated: 2022-01-20
layout: post
title: "SmartMall Discounted Electronic Shopping"
categories:
  - Projects
  - Java
  - Backend Devlopment
  - Microservices
excerpt: "A robust online shopping app with various middlewares serving the microservices architecture"
link: "https://cdn.statically.io/gh/Criss-Wang/image-host@master/Blog/SmartMall2.webp"
mathjax: true
toc: true
---
### **Introduction**

A Spring Boot 3-powered, fully dockerized microservices application. [[**Code**](https://github.com/Criss-Wang/Smart-Mall)]

### **Overall Structure**

![](https://cdn.statically.io/gh/Criss-Wang/image-host@master/Blog/SmartMall2.webp)

#### **Key services**

- API Gateway (Spring Cloud)
- Service Discovery (Netflix Eureka)
- Inventory Service
- Notificaiton Service
- Order Service
- Product Service
- [TODO] Cart Service
- [TODO] Payment Service
- [TODO] Membership/Discount Service

#### **Key Features & Technology**

- **Data Storage**: MongoDB + PostgreSQL + Hibernate
- **IPC**: RestTemplate + WebClient
- **Gateway**: Spring Cloud Gateway
- **Service Registration & Discovery**: Netflix Eureka
- **Security**: Keycloak + JWT
- **Inter-service Communication**: Kafka (Event-driven, asynchronous) + Resilience4J (Circuit Breaker, fault tolerance)
- **Distributed Tracing**: Zipkin + Sleuth
- **Containerization**: Docker + [TODO] Kubernetes
- **Monitoring**: Grafana + Prometheus
- **Logging**: Lombok SLF4J
