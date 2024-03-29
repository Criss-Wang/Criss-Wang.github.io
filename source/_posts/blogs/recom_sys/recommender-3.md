---
title: "Recommender Systems: III. Deep-learning Methods"
excerpt: "Applications of Deep learning in recommender systems"
layout: post
date: 2020/04/05
updated: 2022/3/30
categories:
  - Blogs
tags: 
  - Deep learning
  - Reinforcement Learning
  - Recommender Systems
mathjax: true
toc: true
---
## A brief intro
There are a wide variety of DL tools used for recommendation systems, we will outline a few below. We cite various information from the paper [**Deep Learning based Recommender System: A Survey and New Perspectives**](https://arxiv.org/abs/1707.07435). You may find more details from that paper.

- Multilayer Perceptron (MLP) is a feed-forward neural network with multiple (one or more) hidden layers between the input layer and output layer. Here, the perceptron can employ arbitrary activation function and does not necessarily represent strictly binary classier. MLPs can be intrepreted as stacked layers of nonlinear transformations, learning hierarchical feature representations. MLPs are also known to be universal approximators.
- Autoencoder (AE) is an unsupervised model aempting to reconstruct its input data in the output layer. In general, the bottleneck layer (the middle-most layer) is used as a salient feature representation of the input data. ere are many variants of autoencoders such as denoising autoencoder, marginalized denoising
autoencoder, sparse autoencoder, contractive autoencoder and variational autoencoder (VAE).
- Convolutional Neural Network (CNN) is a special kind of feedforward neural network with convolution layers and pooling operations. It can capture the global and local features and significantly enhancing the eciency and accuracy. It performs well in processing data with grid-like topology.
- Recurrent Neural Network (RNN) is suitable for modelling sequential data. Unlike feedforward neural network, there are loops and memories in RNN to remember former computations. Variants such as Long Short Term Memory (LSTM) and Gated Recurrent Unit (GRU) network are oen deployed in practice to overcome the vanishing gradient problem.
- Restricted Boltzmann Machine (RBM) is a two layer neural network consisting of a visible layer and a hidden layer. It can be easily stacked to a deep net. Restricted here means that there are no intra-layer communications in visible layer or hidden layer.
- Neural Autoregressive Distribution Estimation (NADE) is an unsupervised neural network built atop autoregressive model and feedforward neural networks. It is a tractable and efficient estimator for modelling data distribution and densities.
- Adversarial Networks (AN) is a generative neural network which consists of a discriminator and a generator. The two neural networks are trained simultaneously by competing with each other in a minimax game framework.
- Attentional Models (AM) are dierentiable neural architectures that operate based on soft content addressing over an input sequence (or image). Attention mechanism is typically ubiquitous and was incepted in Computer Vision and Natural Language Processing domains. However, it has also been an emerging trend in deep recommender system research.
- Deep Reinforcement Learning (DRL) . Reinforcement learning operates on a trial-and-error paradigm. The whole framework mainly consists of the following components: agents, environments, states, actions and rewards. The combination between deep neural networks and reinforcement learning formulate DRL which have achieved human-level performance across multiple domains such as games and selfdriving cars. Deep neural networks enable the agent to get knowledge from raw data and derive efficient representations without handcrafted features and domain heuristics.

## Pros and cons
**Pros**
- **Nonlinear Transformation**: Contrary to linear models, deep neural networks is capable of modelling the non-linearity in data with nonlinear activations such as relu, sigmoid, tanh, etc. This property makes it possible to capture the complex and intricate user item interaction patterns. The linear assumption, acting as the basis of many traditional recommenders, is oversimplified and will greatly limit their modelling expressiveness. It is well-established that neural networks are able to approximate any continuous function with an arbitrary precision by varying the activation choices and combinations.
- **Representation Learning**: Deep neural networks is efficacious in learning the underlying explanatory factors and useful representations from input data. In general, a large amount of descriptive information about items and users is available in real-world applications. Making use of this information provides a way to advance our understanding of items and users, thus, resulting in a better recommender. As such, it is a natural choice to apply deep neural networks to representation learning in recommendation models. The advantages of using deep neural networks to assist representation learning are in two-folds: 
  1. it reduces the efforts in hand-craft feature design. Feature engineering is a labor intensive work, deep neural networks enable automatically feature learning from raw data in unsupervised or supervised approach; 
  2. it enables recommendation models to include heterogeneous content information such as text, images, audio and even video. Deep learning networks have made breakthroughs in multimedia data processing and shown potentials in representations learning from various sources.
- **Sequence Modelling**: Deep neural networks have shown promising results on a number of sequential modelling tasks such as machine translation, natural language understanding, speech recognition, chatbots, and many others. RNN and CNN play critical roles in these tasks. RNN achives this with internal memory states while CNN achieves this with filters sliding along with time. Both of them are widely applicable flexible in mining sequential structure in data. Modelling sequential signals is an important topic for mining the temporal dynamics of user behaviour and item evolution. For example, next-item/basket prediction and session based recommendation are typical applications. As such, deep neural networks become a perfect fit for this sequential pattern mining task
- **Flexibility**： Deep learning techniques possess high flexibility, especially with the advent of many popular deep learning frameworks.

**Cons**
- **Interpretability**: Despite its success, deep learning is well-known to behave as black boxes, and providing explainable predictions seem to be a really challenging task. A common argument against deep neural networks is that the hidden weights and activations are generally non-interpretable, limiting explainability. However, this concern has generally been eased with the advent of neural attention models and have paved the world for deep neural models that enjoy improved interpretability. While interpreting individual neurons still pose a challenge for neural models (not only in recommender systems), present state-of-the-art models are already capable of some extent of interpretability, enabling
explainable recommendation. We discuss this issue in more detail in the open issues section.
- **Data Requirement**: A second possible limitation is that deep learning is known to be data-hungry, in the sense that it requires sufficient data in order to fully support its rich parameterization. However, as compared with other domains (such as language or vision) in which labeled data is scarce, it is relatively easy to garner a significant amount of data within the context of recommender systems research. Million/billion scale datasets are commonplace not only in industry but also released as academic datasets.
- **Extensive Hyperparameter Tuning**: A third well-established argument against deep learning is the need for extensive hyperparameter tuning. However, we note that hyperparameter tuning is not an exclusive problem of deep learning but machine learning in general (e.g., regularization factors and learning rate similarly have to be tuned for traditional matrix factorization etc) Granted, deep learning may introduce additional hyperparameters in some cases.