# Chapter 10 Actor-Critic Methods

The term Actor-Critic emphasizes the combination of policy gradient and value-based methods. Here, the actor refers to policy update. It is called the actor because we apply the policy to take actions. The critic refers to policy evaluation or value estimation. It is called the critic because it evaluates (criticizes) the policy.

## Table of Contents

* [Actor-Critic Framework](#actor-critic-framework)
* [Q Actor-Critic (QAC)](#q-actor-critic-qac)
* [Advantage Actor-Critic (A2C)](#advantage-actor-critic-a2c)
* [Off-Policy Actor-Critic](#off-policy-actor-critic)
* [Deterministic Policy Gradient](#deterministic-policy-gradient)

## Actor-Critic Framework

Recall from the previous chapter that the scalar metric $J(\theta)$ can either maximize $\bar{v}_\pi$ or $\bar{r}_\pi$, and the gradient ascent to maximize $J(\theta)$ is given as follows:

$$
\theta _{t+1} = \theta _t + \alpha \nabla_\theta J(\theta)
$$

$$
\theta _{t+1} = \theta _t + \alpha \mathbb{E}_{S\sim \eta, A\sim \pi} \left[\nabla _\theta \ln \pi(A|S,\theta)q_\pi(S,A)\right]
$$

with the corresponding stochastic gradient ascent:

$$
\theta_{t+1} = \theta _t + \alpha \nabla _\theta \ln \pi(a_t|s_t,\theta _t)q_t(s_t,a_t)
$$

This expression is the core idea of actor-critic, and we can directly identify the actor and the critic:

- The full expression corresponds to the actor
- The algorithm used to estimate $q_t(s,a)$ corresponds to the critic

There are two ways to estimate $q_t(s_t,a_t)$:

- If we use Monte Carlo, then the algorithm is known as REINFORCE
- If we use TD learning, then it is known as actor-critic

## Q Actor-Critic (QAC)

In actor-critic methods, there are both a policy function $\pi(a|s,\theta_0)$, where $\theta_0$ is the initial parameter, and a value function $q(s,a,w_0)$, where $w_0$ is the initial parameter.

It consists of:

**Actor (policy update):**

$$
\theta_{t+1} = \theta _t + \alpha _\theta\nabla _\theta \ln \pi(a_t|s_t,\theta _t)q(s_t,a_t, w_t)
$$

**Critic (value update):**

$$
w_{t+1} = w_{t} + \alpha_{w}\left[ r_{t+1}+\gamma q(s_{t+1},a_{t+1},w_t) - q(s_t,a_t,w_t) \right] \nabla_w q(s_t,a_t,w_t)
$$

This particular actor-critic algorithm is known as Q Actor-Critic (QAC), with the following remarks:

- The critic corresponds to SARSA with value function approximation
- The actor corresponds to policy update with policy function approximation

## Advantage Actor-Critic (A2C)

The next actor-critic algorithm is Advantage Actor-Critic (A2C). Recall that policy gradient methods require computing the gradient $\nabla _\theta J(\theta) = \mathbb{E}[X]$, where

$$
X(S,A) =\nabla _\theta \ln \pi(A|S,\theta)q_\pi(S,A)
$$

The stochastic gradient requires us to use random samples to approximate the true gradient $\mathbb{E}[X]$. However, this may introduce variance $\mathrm{var}(X)$.

The objective of the A2C algorithm is to minimize $\mathrm{var}(X)$ by adding a baseline $b(S)$ to the gradient $\nabla_\theta J(\theta)$, where $b(S)$ is a scalar function of $S$.

From

$$
\nabla_\theta J(\theta) = \mathbb{E}_{S\sim \eta, A\sim \pi} \left[\nabla _\theta \ln \pi(A|S,\theta)q_\pi(S,A)\right]
$$

to

$$
\nabla_\theta J(\theta) = \mathbb{E}_{S\sim \eta, A\sim \pi} \left[\nabla _\theta \ln \pi(A|S,\theta)(q_\pi(S,A)-b(S))\right]
$$

If we choose $b(s)=v_\pi(s)$, then the gradient ascent is given by:

$$
\theta _{t+1} = \theta _t + \alpha \mathbb{E}_{S\sim \eta, A\sim \pi} \left[\nabla _\theta \ln \pi(A|S,\theta)(q_\pi(S,A)-v_\pi(S))\right]
$$

and we can write

$$
\delta _\pi(S,A) = q_\pi(S,A)-v_\pi(S)
$$

where $\delta _\pi(S,A)$ is known as the advantage function (hence the name A2C), and it can be approximated by the TD error (recall from the chapter on Temporal Difference):

placeholder

## Off-Policy Actor-Critic

Up to now, policy gradient methods are on-policy because the gradient is

$$
\nabla_\theta J(\theta) = \mathbb{E}_{S\sim \eta, A\sim \pi}[*]
$$

Is it possible to convert policy gradient methods to off-policy?

- It is possible by using importance sampling
- The technique of importance sampling is not limited to actor-critic or reinforcement learning. It applies to any algorithm that aims to estimate an expectation of a distribution without sampling directly from the target distribution

## Deterministic Policy Gradient

DPG
