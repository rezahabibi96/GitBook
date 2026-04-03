# Chapter 9 Policy Gradient Methods

So far, policies have been represented using tables. However, similar to state value and action value, policies can also be represented by a parameterized function $$\pi(a|s,\theta)$$, where $$\theta \in \mathbb{R}^m$$ is a parameter vector.

## Table of Contents

* [From Value-Based to Policy-Based Methods](#from-value-based-to-policy-based-methods)
* [Performance Metrics](#performance-metrics)
* [Policy Gradient Theorem](#policy-gradient-theorem)
* [Stochastic Policy Gradient](#stochastic-policy-gradient)
* [REINFORCE and Actor-Critic](#reinforce-and-actor-critic)
* [Remarks on Policy Gradient](#remarks-on-policy-gradient)

## From Value-Based to Policy-Based Methods

For value-based methods, our goal is to find a parameterized function $$\hat{v}(s,w)$$ that best approximates the true state value $$v(s)$$, where $$w\in \mathbb{R}^m$$ is the parameter vector. In contrast, for policy-based methods, the objective of the parameterized policy $$\pi(a|s,\theta)$$ is to best approximate the optimal policy $$\pi^*$$.

Note that in the tabular case, a policy $$\pi$$ is optimal if it can maximize every state value. However, in the function approximation case, a policy $$\pi$$ is optimal if it can maximize certain scalar metrics.

## Performance Metrics

These scalar metrics can be:

1. Average value $$\bar{v}_\pi$$.

There are several ways to express this:

- Expression 1: $$\sum_{s\in S}{d(s)v_\pi (s)}$$
- Expression 2: $$\mathbb{E}_{S\sim d}[v_\pi (S)]$$
- Expression 3: $$\lim _{n \rightarrow \infty}{\mathbb{E}\left[\sum_{t=0}^{n} \gamma ^t R_{t+1}\right]}$$

where $$\bar{v}_\pi$$ is a weighted average of the state value, also called the average value. The $$d(s)$$ is a probability distribution over state $$s$$ that can either be uniform (independent of $$\pi$$) or the long-run stationary distribution under $$\pi$$.

2. Average reward $$\bar{r}_\pi$$.

- Expression 1: $$\sum_{s\in S}{d_\pi(s)r_\pi (s)}$$
- Expression 2: $$\mathbb{E}_{S\sim d_\pi}[r_\pi (S)]$$
- Expression 3: $$\lim _{n \rightarrow \infty}{\frac{1}{n} \mathbb{E}\left[\sum_{t=0}^{n-1} \gamma ^t R_{t+1}\right]}$$

where $$\bar{r}_\pi$$ is a weighted average of the immediate reward, also called the average reward; $$r_\pi(s)$$ is the average immediate reward obtained from state $$s$$; and $$d_\pi$$ is the long-run stationary distribution under $$\pi$$.

The relationship between $$\bar{v}_\pi$$ and $$\bar{r}_\pi$$ is given by $$\bar{r}_\pi=(1-\gamma) \bar{v}_\pi$$, so they can be maximized simultaneously.

After identifying the possible metrics for the optimal policy (in the case of function approximation), we can search for the optimal values of $$\theta$$ such that $$\pi(a|s,\theta)$$ maximizes these metrics.

## Policy Gradient Theorem

Knowing this, we can apply gradient-based optimization methods to maximize $$J(\theta)$$. The corresponding gradient expression is given below (see the textbook for the detailed proof):

$$
\nabla _\theta J(\theta) = \sum_{s\in S} \eta(s) \sum_{a\in A} \nabla _\theta \pi(a|s,\theta)q_\pi(s,a)
$$

where $$J(\theta)$$ can either correspond to maximizing $$\bar{v}_{\pi _\theta}$$ or $$\bar{r}_{\pi _\theta}$$, and $$\eta$$ is a distribution or weighting over the states.

We can also express it in expectation form (again, see the textbook for the detailed proof) so that we can apply stochastic gradient methods:

$$
\nabla _\theta J(\theta) = \mathbb{E}_{S\sim \eta, A\sim \pi} \left[\nabla _\theta \ln \pi(A|S,\theta)q_\pi(S,A)\right]
$$

## Stochastic Policy Gradient

By using stochastic gradients, we can approximate the gradient:

$$
\nabla _\theta J(\theta) = \nabla _\theta \ln \pi(a|s,\theta)q_\pi(s,a)
$$

where $$s, a$$ are samples. It is required by $$\ln \pi (a|s,\theta)$$ that $$\pi(a,|s,\theta)>0$$ for any $$s,a,\theta$$. We can use the softmax function at the last layer to satisfy this condition.

Therefore, we can update the values of $$\theta$$ using stochastic gradient ascent as follows:

$$
\theta_{t+1} = \theta _t + \alpha \nabla _\theta \ln \pi(a_t|s_t,\theta _t)q_\pi(s_t,a_t)
$$

However, since $$q_\pi$$ is unknown, we can replace it with an estimate:

$$
\theta_{t+1} = \theta _t + \alpha \nabla _\theta \ln \pi(a_t|s_t,\theta _t)q_t(s_t,a_t)
$$

## REINFORCE and Actor-Critic

If $$q_\pi(s_t,a_t)$$ is estimated using Monte Carlo, it is known as REINFORCE, one of the earliest and simplest policy gradient algorithms. If $$q_\pi(s_t,a_t)$$ is estimated using TD learning, it is known as Actor-Critic, which will be introduced next.

REINFORCE is an on-policy gradient algorithm because $$A \sim \pi(A|S,\theta)$$, hence it requires $$a_t$$ to be sampled according to $$\pi (s_t, \theta_t)$$.

Since $$\nabla _\theta \ln \pi(a_t|s_t,\theta _t) = \frac{\nabla _\theta \pi(a_t|s_t,\theta _t)}{\pi(a_t|s_t,\theta _t)}$$, we can also rewrite:

$$
\theta_{t+1} = \theta _t + \alpha \nabla _\theta \ln \pi(a_t|s_t,\theta _t)q_t(s_t,a_t)
$$

as follows:

$$
\theta_{t+1} = \theta _t + \alpha \beta_t \nabla _\theta \pi(a_t|s_t,\theta _t)
$$

where $$\beta _t = \frac{q_t(s_t,a_t)}{\pi(a_t|s_t,\theta _t)}$$, which can be interpreted as balancing exploration and exploitation.

## Remarks on Policy Gradient

Since $$\pi(a,|s,\theta)>0$$ and $$a_t$$ is sampled according to $$\pi (s_t, \theta_t)$$, the parameterized policy is stochastic. There also exist deterministic policy gradient methods, which will be discussed in the next lecture.

Unlike REINFORCE (a stochastic policy gradient method), deterministic policy gradient methods are off-policy.
