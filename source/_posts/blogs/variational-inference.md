---
title: "Variational Inference"
excerpt: "A useful inference method for distributional approximation"
layout: post
date: 2021/10/14
updated: 2021/12/12
categories:
  - Blogs
tags: 
  - Optimization
  - Bayesian Statistics
mathjax: true
toc: true
---
### Introduction
#### 1. Background of Bayesian methods

In the field of machine learning, most would agree that frequentist approaches played a critical role in the development of early classical models. Nevertheless, we are witnessing the increasing significance of Bayesian methods in modern study of machine learning and data modelling. The simple-looking Bayes\' rule $p(z \mid x)=\frac{p(x \mid z) p(z)}{p(x)}$ has inspired a lot wonderful models in areas like topic modelling, representation learning and hyperparameter optimization. In these models, the latent variables $z=\left(z_{1}, z_{2}, \ldots z_{n}\right)$ are the focus of the study. By analysing several data on the observed variables $x=\left(x_{1}, x_{2}, \ldots x_{m}\right)$, we hope to get some meaningful information (for example, a point estimate or an entire distribution) about these latent variables.

#### 2. Problem with Bayesian methods: intractable integral

While the rule looks easily understandable, the numerical computation is hard in reality. One major issue is the intractable integral $\int_{z} p(x \mid z) p(z) d z$ we need to compute in order to get the $p(x)$, which is often called the \"model evidence\". This is often because the search space for $Z$ is combinatorially too large, making the computation extremely expensive. A common approach to deal with this problem is to approximate the posterior probability $p(z \mid x)$ directly. Some popular choices include Monte Carlo Sampling methods and variational inference. In this report, we will introduce the variational methods, which are perhaps the most widely used inference technique in machine learning. We will analyse a particularly famous technique in variational methods, mean-field variational inference.

#### 3. Main idea of variational inference

In variational inference, we can avoid computing the intractable integral by magically modelling the posterior $p$ directly. The main trick here is to approximate the unknown distribution $p$ with some similar distribution q. Since we can choose the q to belong to a certain family of distribution (hence tractable), the problem is now transformed into an optimization problem about the parameters of $q$.

### Understanding Variational Bayesian method

In this section, we demonstrate the theory behind variational Bayesian methods. 

#### 1. Kullback-Leibler Divergence

As mentioned above, variational inference needs a distribution $q$ to approximate the posterior distribution $p$. Therefore we need to gauge how well a candidate $q$ approximates the posterior. A common measure is Kullback-Leibler Divergence (often called KL divergence).

KL divergence is defined as

$$
K L(q \| p)=\int_{z} q(z) \frac{q(z)}{p(z \mid x)}=E_{q}\left[\log \frac{q(z)}{p(z \mid x)}\right]
$$

Where $E_{q}$ means the expected value with respect to distribution $q$. The formula can be interpreted as follows:

- if $\mathrm{q}$ is low, the divergence is generally low.

- if $\mathrm{q}$ is high and $\mathrm{p}$ is high, the divergence is low.

- if $\mathrm{q}$ is high and $\mathrm{p}$ is low, the divergence is high, hence the approximation is not ideal.

Take note of the following about use of KL divergence in Variational Bayes:

1. KL divergence is not symmetric, it\'s easy to see from the formula that $K L(p \| q) \neq K L(q \| p)$ as the approximation distribution $q$ is usually different from the target distribution $p$.

2. In general, we focus on approximating some regions of $p$ as good as possible (Figure 1 (a)). It is not necessary for the $q$ to nicely approximate every part of $p .$ As a result $K L(p \| q)$ (usually called forward $\mathrm{KL}$ divergence) is not ideal. Because for some regions $p>0$ which we don\'t want to care, if $q \rightarrow 0$, the KL divergence will be very large, forcing $q$ to take a different form even if it fits well with other regions of $p$ (refer to Figure 1(b)). On the other hand, $K L(q \| p)$ (usually called reverse KL divergence) has the nice property that only regions where $q>0$ requires $p$ and $q$ to be similar. Consequently, reverse KL divergence is more commonly used in Variational Inference.

#### 2. Evidence lower bound

Usually we don\'t directly minimizing KL divergence to obtain a good approximated distribution. This is because computing $\mathrm{KL}$ divergence still depends on the posterior $p(z \mid x)$. The computation involves the \"evidence\" term $p(x)$ which is expensive to compute, as shown in the formula below:

$$
\begin{aligned}
K L(q \| p) &=E_{q}\left[\log \frac{q(z)}{p(z \mid x)}\right] \\\\
&=E_{q}[\log q(z)]-E_{q}[\log p(z \mid x)] \\\\
&=E_{q}[\log q(z)]-E_{q}[\log p(z, x)]+\log p(x) \\\\
&=-\left(E_{q}[\log p(z, x)]-E_{q}[\log q(z)]\right)+\log p(x)
\end{aligned}
$$

The approximation using reverse KL divergence usually gives good empirical results, even though some regions of $p$ may be compromised 
<figure class="half">
	<a href="https://agustinus.kristia.de/techblog/2016/12/21/forward-reverse-kl/"><img src="/images/Machine%20learning/reverse_kl.png"></a>
	<a href="https://agustinus.kristia.de/techblog/2016/12/21/forward-reverse-kl/"><img src="/images/Machine%20learning/forward_kl.png"></a>
	<figcaption>Figure 1: Reverse KL vs Forward KL divergence: The left image has a better approximation $\mathrm{Q}(\mathrm{x})$ on part of  $\mathrm{P}(\mathrm{x})$. (a) Reverse KL simulation (b) Forward KL simulation</figcaption>
</figure>

We can directly conclude by the fact $K L(q \| p) \geq 0$ that the term $E_{q}[\log p(z, x)]-E_{q}[\log q(z)]$ is $\operatorname{less}$ than the log of evidence. We can also proof this result using Jensen\'s inequality as follows:

- By the definition of marginal probability, we have $p(x)=\int_{z} p(x, z)$, take log on both side we have:

$$
\begin{aligned}
\log p(x) &=\log \int_{z} p(x, z) \\\\
&=\log \int_{z} p(x, z) \frac{q(z)}{q(z)} \\\\
&=\log \left(E_{q}\left[\frac{p(z, x)}{q(z)}\right]\right) \\\\
& \geq E_{q}[\log p(z, x)]-E_{q}[\log q(z)]
\end{aligned}
$$

- The last 2 lines follow from Jensen\'s Inequality which states that for a convex function $f$, we have $f(E[X]) \geq E[f(X)]$

This term $E_{q}[\log p(z, x)]-E_{q}[\log q(z)]$ is known as the Evidence Lower Bound, or $E L B O .$ Since log $p(x)$ does not depend on $q$, we can treat it as a constant from the perspective of optimizing $q$. Hence, minimizing $K L(q \| p)$ is now equivalent to maximizing $E L B O .$

### General procedure

In general, a variational inference starts with a family of variational distribution (such as the mean-field family described below) as the candidate for $q$. We can then use the manually chosen $q$ to compute the ELBO terms $E_{q}[\log p(z, x)]$ and $E_{q}[\log q(z)]$. Afterwards, we optimize the parameters in $q$ to maximize the ELBO value using some optimization techniques (such as coordinate ascent and gradient methods). 

### Mean Field Variational Family

#### 1. The \"Mean Field\" Assumptions

As shown above, the particular variational distribution family we use to approximate the posterior $p$ is chosen by ourselves. A popular choice is called the mean-field variational family. This family of distribution assumes joint approximation distribution $q(Z)$ to be factorized over some partition of the latent variables. This implies mutual independence among the n fractions in the partition. In particular: we have

$$
p(z \mid X) \approx q(z)=\prod_{i=1}^{n} q_{i}\left(z_{i}\right)
$$

where $z$ is factorized into $z_{1}, \ldots, z_{n}$. For simplicity, we assume that each fraction only contains 1 latent variable $(n=\|z\|)$, it is often referred as \"naive mean field\". This family is nice to analyse because we can model each distribution with a tractable distribution based on the problem set-up. Do note that a limitation of this family is that we cannot easily capture the interdependence among the latent variables.

#### 2. Derivation of optimal $q_{j}\left(z_{j}\right)$

Now in order to derive the the optimal form of distribution for $q_{j}\left(z_{j}\right)$ and thus the overall $q$, we need to go back to the ELBO optimization with this mean-field family assumption. Recall the formula for ELBO (we use $\mathcal{L}$ here as it is the convention): $\mathcal{L}=E_{q}[\log p(x, z)]-E_{q}[\log q(z)] .$ We express this formula in terms of $q_{j}\left(z_{j}\right)$ as using functional integral (see appendix A):

$$
\mathcal{L}=\int_{z_{j}} q_{j}\left(z_{j}\right) E_{q_{-j}}[\log p(x, z)] d z_{j}+E_{q_{j}}\left[\log q_{j}\left(z_{j}\right)\right]+G\left(q_{1}, \ldots, q_{j-1}, q_{j+1}, \ldots, q_{n}\right)
$$

With this new expression, we can consider maximizing $\mathcal{L}$ with respect to each of the $q_{j}\left(z_{j}\right)$. The optimal form of $q_{j}\left(z_{j}\right)$ is the one which maximizes $\mathcal{L}$, that is:

$$
\begin{aligned}
\underset{q_{j}}{\arg \max } \mathcal{L} &=\underset{q_{j}}{\arg \max } \int_{z_{j}} q_{j}\left(z_{j}\right) E_{q_{-j}}[\log p(x, z)] d z_{j}+E_{q_{j}}\left[\log q_{j}\left(z_{j}\right)\right]+G\left(q_{1}, \ldots, q_{j-1}, q_{j+1}, \ldots, q_{n}\right) \\\\
&=\underset{q_{j}}{\arg \max } \int_{z_{j}} q_{j}\left(z_{j}\right)\left(E_{q_{-j}}[\log p(x, z)]+\log q_{j}\left(z_{j}\right)\right) d z_{j}
\end{aligned}
$$

We take the derivative with respect to $q_{j}\left(z_{j}\right)$ using Lagrange multipliers $\lambda_{j}{ }^{2}$ and set to 0 yields:

$$
\begin{aligned}
\log q_{j}\left(z_{j}\right) &=E_{q_{-j}}[\log p(x, z)]-1-\lambda_{j} \\\\
q_{j}\left(z_{j}\right) &=\frac{\exp \left(E_{q_{-j}}[\log p(x, z)]\right)}{C_{j}}
\end{aligned}
$$

where $C_{j}$ is a normalization constant that plays minimal role in the variable update.

The funtional derivative of this expression actually requires some knowledge about calculus of variations, specifically [Euler-Lagrange equation](https://en.wikipedia.org/wiki/Calculus_of_variations).

#### 3. Variable update with Coordinate Ascent

From equation $(9)$ we found that $q_{j}\left(z_{j}\right) \propto \exp \left(E_{q_{-j}}[\log p(x, z)]\right)$. Therefore iterative optimization algorithms like [Coordinate Ascent](https://en.wikipedia.org/wiki/Coordinate_descent) can be applied to update the latent variables to reach their optimal form. Note that all the $q_{j}\left(z_{j}\right)$ \'s are interdependent during the update, hence in each iteration, we need to update all the $q_{j}\left(z_{j}\right)$ \'s. As short description for the coordinate ascent in this setup will be:

1. Compute values (if any) that can be directly obtained from data and constants

2. Initialize a particular $z_{j}$ to an arbitrary value

3. Update each variable with the step function $\left(\propto \exp \left(E_{q_{-j}}[\log p(x, z)]\right)\right)$

4. Repeat step 3 until the convergence of ELBO

A more detailed example of coordinate ascent will be shown in next section with the univariate gaussian distribution example. A point to take note that in general, we cannot guarantee the convexity of ELBO function. Hence, the convergence is usually to a local maximum.

### Example with Univariate Gaussian

We demonstrate the mean-field variational inference with a simple case of observations from univariate Gaussian model. We first assume there are $N$ observations $X=\left(x_{1}, \ldots x_{N}\right)$ from a Gaussian distribution satisfying:

$$
\begin{aligned}
&x_{i} \sim \mathcal{N}\left(\mu, \tau^{-1}\right), i=1, \ldots, N \\\\
&\text { where } \mu \sim \mathcal{N}\left(\mu_{0},\left(\kappa_{0} \tau\right)^{-1}\right) \& \tau \sim \operatorname{Gamma}\left(a_{0}, b_{0}\right)
\end{aligned}
$$

Here $\tau$ is inverse of variance $\sigma^{2}$ (hence one-to-one correspondence). From the derivation of $q_{j}\left(z_{j}\right)$ we know we need to compute the log joint probability $\log p(X, \mu, \tau)$. We will first derive an explicit formula for it by expanding the join probability into conditional probability:

$$
\begin{aligned}
\log p(X, \mu, \tau)=& \log p(X \mid \mu, \tau)+\log p(\mu \mid \tau)+\log p(\tau) \\\\
=& \log \left(\sqrt{\frac{\tau^{N}}{2 \pi}} e^{-\frac{\tau}{2} \sum_{i=1}^{N}\left(x_{i}-\mu\right)^{2}}\right)+\log \left(\sqrt{\frac{\kappa_{0} \tau}{2 \pi}} e^{-\frac{\kappa_{0} \tau}{2}\left(\mu-\mu_{0}\right)^{2}}\right)+\log \left(\frac{b_{0}^{a_{0}}}{\Gamma\left(a_{0}\right)} \tau^{a_{0}-1} e^{-b_{0} \tau}\right) \\\\
=& \frac{N}{2} \log \tau-\frac{\tau}{2} \sum_{i=1}^{N}\left(x_{i}-\mu\right)^{2}+\frac{1}{2} \log \left(\kappa_{0} \tau\right)-\frac{\kappa_{0} \tau}{2}\left(\mu-\mu_{0}\right)^{2} +\left(a_{0}-1\right) \log \tau-b_{0} \tau+C
\end{aligned}
$$

where $\mathrm{C}$ is a constant.


Note that sometimes some latent variable has higher priority that others. The choice of this variable depends on the exact question in hand. 

#### 1. Compute independent $q_{\mu}(\mu)$ and $q_{\tau}(\tau)$

Next, we apply approximation via $p(\mu, \tau \mid X) \approx q(\mu, \tau)$. By the mean-field assumption, we have $q(\mu, \tau)=$ $q_{\mu}(\mu) q_{\tau}(\tau)$. We proceed to find the optimal form of $q_{\mu}(\mu)$ and $q_{\tau}(\tau)$ :

- Compute the expression for $q_{\mu}(\mu)$ :

$$
\begin{aligned}
\log q_{\mu}(\mu) &=E_{\tau}[\log p(X, \mu, \tau)]+C_{1} \\\\
&=E_{\tau}\left[\frac{N}{2} \log \tau-\frac{\tau}{2} \sum_{i=1}^{N}\left(x_{i}-\mu\right)^{2}+\frac{1}{2} \log \left(\kappa_{0} \tau\right)-\frac{\kappa_{0} \tau}{2}\left(\mu-\mu_{0}\right)^{2}+\left(a_{0}-1\right) \log \tau-b_{0} \tau\right]+C_{2} \\\\
&=-\frac{E_{\tau}[\tau]}{2}\left[\kappa_{0}\left(\mu-\mu_{0}\right)^{2}+\sum_{i=1}^{N}\left(x_{i}-\mu\right)^{2}\right]+C_{3} \\\\\
&\text { (completing the square for the term inside the square bracket) } \\\\
&=-\frac{\left(\kappa_{0}+N\right) E_{\tau}[\tau]}{2}\left(\mu-\frac{\kappa_{0} \mu_{0}+\sum_{i=1}^{N} x_{i}}{\kappa_{0}+N}\right)^{2}+C_{4}
\end{aligned}
$$



Note that here $E_{\tau}$ is a shortcut representation for $E_{q_{\tau}(\tau)}$, and all $C_{1}, \ldots C_{k}$ are constant terms not involved in the optimization update. From the expression above, it\'s easy to observe that $q_{\mu}(\mu)$ follows a Gaussian distribution with $q_{\mu}(\mu) \sim \mathcal{N}\left(\hat{\mu}, \hat{\tau}^{-1}\right)$, where:

$$
\begin{aligned}
\hat{\mu} &=\frac{\kappa_{0} \mu_{0}+\sum_{i=1}^{N} x_{i}}{\kappa_{0}+N} \\\\
\hat{\tau} &=\left(\kappa_{0}+N\right) E_{\tau}[\tau]
\end{aligned}
$$

- Compute the expression for $q_{\tau}(\tau):$

$$
\begin{aligned}
\log q_{\tau}(\tau) &=E_{\mu}\left[\frac{N}{2} \log \tau-\frac{\tau}{2} \sum_{i=1}^{N}\left(x_{i}-\mu\right)^{2}+\frac{1}{2} \log \left(\kappa_{0} \tau\right)-\frac{\kappa_{0} \tau}{2}\left(\mu-\mu_{0}\right)^{2}+\left(a_{0}-1\right) \log \tau-b_{0} \tau\right]+C_{5} \\\\
&=\left(a_{0}-1\right) \log (\tau)-b_{0} \tau+\frac{1+N}{2} \log (\tau)-\frac{\tau}{2} E_{\mu}\left[\kappa_{0}\left(\mu-\mu_{0}\right)^{2}+\sum_{i=1}^{N}\left(x_{i}-\mu\right)^{2}\right]+C_{6} \\\\
&=\left(a_{0}-1+\frac{1+N}{2}\right) \log (\tau)-\left(b_{0}+\frac{1}{2} E_{\mu}\left[\kappa_{0}\left(\mu-\mu_{0}\right)^{2}+\sum_{i=1}^{N}\left(x_{i}-\mu\right)^{2}\right]\right) \tau+C_{6}
\end{aligned}
$$

A closer look at the result $(9)$ suggest that $q_{\mu}(\mu)$ follows a Gaussian distribution with $q_{\tau}(\tau) \sim$ Gamma $(\hat{a}, \hat{b})$, where:

$$
\begin{aligned}
&\hat{a}=a_{0}+\frac{1+N}{2} \\\\
&\hat{b}=b_{0}+\frac{1}{2} E_{\mu}\left[\kappa_{0}\left(\mu-\mu_{0}\right)^{2}+\sum_{i=1}^{N}\left(x_{i}-\mu\right)^{2}\right]
\end{aligned}
$$


#### 2. Variable update until ELBO convergence}

Now that we have $q_{\mu}(\mu) \sim \mathcal{N}\left(\hat{\mu}, \hat{\tau}^{-1}\right)$ and $q_{\tau}(\tau) \sim \operatorname{Gamma}(\hat{a}, \hat{b})$, we only need to update their parameters:

$$
\begin{aligned}
&\hat{\tau}=\left(\kappa_{0}+N\right) E_{\tau}[\tau]=\left(\kappa_{0}+N\right) \frac{\hat{a}}{\hat{b}} \\\\
&\hat{a}=a_{0}+\frac{1+N}{2} \\\\
&\hat{b}=b_{0}+\frac{1}{2}\left[\kappa_{0}\left(\frac{1}{\hat{\tau}}+\hat{\mu}^{2}+\mu_{0}^{2}-2 \hat{\mu} \mu_{0}\right)+\sum_{i=1}^{N}\left(x_{i}^{2}-2 \hat{\mu} x_{i}+\frac{1}{\hat{\tau}}+\hat{\mu}^{2}\right)\right] \\\\
&\hat{\mu}=\frac{\kappa_{0} \mu_{0}+\sum_{i=1}^{N} x_{i}}{\kappa_{0}+N}
\end{aligned}
$$

Using the updated $q_{\mu}(\mu)$ and $q_{\tau}(\tau)$, we can then compute $E L B O=E_{q}[\log p(X, \mu, \tau)]-E_{q}\left[\log q_{\mu}(\mu)+\right.$ $\left.\log q_{\tau}(\tau)\right]$ with $q=q_{\mu} q_{\tau} .$ Hence the coordinate ascent algorithm can be applied here:

1. Compute $\hat{\mu}$ and $\hat{a}$ as they can be derived directly from the data and constants based on their formula

2. Initialize $\hat{\tau}$ to some random value

3. Update $\hat{b}$ with current values of $\hat{\mu}, \hat{a}$ and $\hat{\tau}$

4. Update $\hat{\tau}$ with current values of $\hat{\mu}, \hat{a}$ and $\hat{b}$

5. Compute ELBO value with the variables $\mu$ \& $\tau$ updated with the parameters in step 1 - 4

6. Repeat the last 3 steps until ELBO value doesn\'t vary by much

As a result of the algorithm, we obtain an approximation $q=q_{\mu} q_{\tau}$ for the posterior distribution of $\mu$ and $\tau$ given observations $X$.

### Extension and Further result

In this section, we briefly outline some more theory and reflection about general variational Bayesian methods. Due to space limitations, we only provide a short discussion on each of these.

### Exponential family distributions in Varational Inference

A nice property of the exponential family distribution is the presence of conjugate priors in closed forms. This allows for less computationally intensive approaches when approximating posterior distributions (due to reasons like simpler optimization algorithm applicable and better analytical forms). Further more, Gharamani \& Beal even suggested in 2000 that if all the $q_{j}\left(z_{j}\right)$ belong to the same exponential family, the update of latent variables in the optimization procedure can be exact.

A great achievement in the field of variational inference is the generalized update formula for Exponentialfamily-conditional models. These models has conditional densities that are in exponential family. The nice property of exponential family leads to an amazing result that the optimal approximation form for posteriors are in the same exponential family as the conditional. This has benefits a lot of well-known models like Markov random field and Factorial Hidden Markov Model.

### Comparison to other Inference methods

The ultimate results of variational inference are the approximation for the entire posterior distribution about the parameters and variables in the target problem with some observations instead of just a single point estimate. This serves the purpose of further statistical study of these latent variables, even if their true distributions are analytically intractable. Another group of inference methods commonly used to achieve the similar aim is Markov chain Monte Carlo (MCMC) methods like Gibbs sampling, which seeks to produce reliable resampling of given observations that help to approximate latent variables well. Another common Bayesian method that has a similar iterative variable update procedure is Expectation Maximization (EM). For EM, however, only point estimates of posterior distribution are obtained. The estimates are \"Expectation maximizing\" points, which means any information about the distribution around these points (or the parameters they estimate) are not preserved. On the other hand, despite the advantage of \"entire distribution\" Variational inference has, its point estimates are often derived just by the mean value of the approximated distributions. Such point estimates are often less significant compared to those derived using EM, as the optimum is not directly achieved from the Bayesian network itself, but the optimal distributions inferred from the network.

### Popular algorithms applying variational inference

The popularity of variational inference has grown to even surpass the classical MCMC methods in recent years. It is particularly successful in generative modeling as a replacement for Gibbs sampling. The methods often show better empirical result than Gibbs sampling, and are thus more well-adopted. We here showcase some popular machine learning models and even deep learning models that heavily rely on variational inference methods and achieved great success:

- Latent Dirichlet Allocation: With the underlying Dirichlet distribution, the model applies both variational method (for latent variable distribution) and EM algorithm to obtain an optimal topic separation and categorization.

- variational autoencoder: The latent Gaussian space (a representation for the input with all the latent variables and parameters) is derived from observations, and fine-tuned to generate some convincing counterparts (a copy for instance) of the input.

These models often rely on a mixture of statistical learning theories, but variational inference is definitely one of the key function within them. 
