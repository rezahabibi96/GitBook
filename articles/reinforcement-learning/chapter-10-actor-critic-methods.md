# Chapter 10 Actor-Critic Methods

The term Actor-Critic emphasizes to incorporate policy gradient and value-based methods together. Here, actor refers to policy update. It is known as actor because we will apply policy to take actions.  Then for critic, it refers to policy evaluation or value estimation. It is called critic because it criticizes the policy by evaluating it.

Recall from last lecture, scalar metric $J(\theta)$ can either be to maximize $\bar{v}_\pi$ or $\bar{r}_\pi$, and the gradient ascent to maximize $J(\theta)$ as follow:

$$
\theta _{t+1} = \theta _t + \alpha \nabla_\theta J(\theta)
$$

$$
\theta _{t+1} = \theta _t + \alpha \mathbb{E}_{S\sim \eta, A\sim \pi} \left[\nabla _\theta \ln \pi(A|S,\theta)q_\pi(S,A)\right]
$$

with the accompany stochastic gradient ascent as follow:

$$
\theta_{t+1} = \theta _t + \alpha \nabla _\theta \ln \pi(a_t|s_t,\theta _t)q_t(s_t,a_t)
$$

This expression is the core idea of actor-critic and we can directly see the actor and the critic:

- the full expression corresponds to actor
- while the algorithm to estimate $q_t(s,a)$ corresponds to critic

There are two ways to estimate $q_t(s_t,a_t)$:

- If we use Monte Carlo, then the algorithm is known as REINFORCE
- If we use TD learning, then it is known as actor-critic

QAC

Basically, in actor-critic methods, there will be both policy function $\pi(a|s,\theta_0)$ where $\theta_0$ is the initial parameter and value function $q(s,a,w_0)$ where $w_0$ is the initial parameter. It consists of Actor (policy update) given by:

$$
\theta_{t+1} = \theta _t + \alpha _\theta\nabla _\theta \ln \pi(a_t|s_t,\theta _t)q(s_t,a_t, w_t)
$$

and Critic (value update) given by

$$
w_{t+1} = w_{t} + \alpha_{w}\left[ r_{t+1}+\gamma q(s_{t+1},a_{t+1},w_t) - q(s_t,a_t,w_t) \right] \nabla_w q(s_t,a_t,w_t)
$$

This particular actor-critic algorithm is known as Q Actor-Critic (QAC) and remarks:

- The critic corresponds to SARSA + value function approximation
- The actor corresponds to policy update + policy function approximation

A2C

The next actor-critic algorithm is Advantage Actor-Critic (A2C). Recall that the policy gradient methods require to compute the gradient $\nabla _\theta J(\theta) = \mathbb{E}[X]$ where

$$
X(S,A) =\nabla _\theta \ln \pi(A|S,\theta)q_\pi(S,A)
$$

with the stochastic gradient requires us to use random sample to approximate the true gradient $\mathbb{E}[X]$. However, it may introduce variance $\mathrm{var}(X)$. 

The objective of A2C algorithm is to minimize $\mathrm{var}(X)$ by adding baseline $b(S)$ to its gradient $\nabla_\theta J(\theta)$, where $b(S)$ is a scalar function of $S$.

From

$$
\nabla_\theta J(\theta) = \mathbb{E}_{S\sim \eta, A\sim \pi} \left[\nabla _\theta \ln \pi(A|S,\theta)q_\pi(S,A)\right]
$$

To:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{S\sim \eta, A\sim \pi} \left[\nabla _\theta \ln \pi(A|S,\theta)(q_\pi(S,A)-b(S))\right]
$$

If we chose the $b(s)=v_\pi(s)$ then the gradient ascent given by:

$$
\theta _{t+1} = \theta _t + \alpha \mathbb{E}_{S\sim \eta, A\sim \pi} \left[\nabla _\theta \ln \pi(A|S,\theta)(q_\pi(S,A)-v_\pi(S))\right]
$$

and we can write 

$$
\delta _\pi(S,A) = q_\pi(S,A)-v_\pi(S)
$$

where $\delta _\pi(S,A)$ is known as advantage function (hence the name is A2C) and it can be approximated by the TD error (recall from the chapter Temporal Difference):

placeholder

Off-Policy Actor-Critic

Up to now, the policy gradient is on-policy because the gradient is 

$$
\nabla_\theta J(\theta) = \mathbb{E}_{S\sim \eta, A\sim \pi}[*]
$$

Is it possible to convert policy gradient to off-policy?

- it is possible by importance sampling
- the technique of importance sampling is not limited to actor-critic or reinforcement learning. It applies to any algorithm that aims to estimate an expectation of a distribution without sampling directly from the underlying target distribution

DPG
