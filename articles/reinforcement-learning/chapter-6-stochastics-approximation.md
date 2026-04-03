# Chapter 6 Stochastic Approximation

This chapter introduces the concept of stochastic approximation (SA), with two examples: the Robbins–Monro (RM) algorithm and stochastic gradient descent. It serves as the foundation for the next RL algorithm: Temporal Difference (TD).

## Table of Contents

* [Introduction to Stochastic Approximation](#introduction-to-stochastic-approximation)
* [Robbins-Monro Algorithm](#robbins-monro-algorithm)
* [Stochastic Gradient Descent](#stochastic-gradient-descent)
* [Connection to Reinforcement Learning](#connection-to-reinforcement-learning)

## Introduction to Stochastic Approximation

Stochastic approximation (SA) refers to a broad class of stochastic iterative algorithms for solving **root finding** or **optimization problems**.

Many problems can eventually be converted into root finding problems. Suppose we are given the problem of how to calculate the mean $\bar x$, that is, $\mathbb{E}[X]\approx \bar x$.

- The first way, which is trivial, is to collect all the samples and then calculate the average, similar to what MC does.

    $$
    \bar x = \frac{1}{N}\sum_{i=1}^{N}x_i
    $$

    The drawback is that we have to wait until all the samples are collected. In fact, this is a disadvantage of MC compared to TD, which supports incremental or iterative estimation.

- The second way is by using the Robbins-Monro (RM) algorithm, but how?

## Robbins-Monro Algorithm

Suppose we would like to find the root of the equation

$$
g(w)=0,
$$

where $g:\R \rightarrow \R$ is a function and $w \in \R$ is the variable to be solved (the root).

It is clearly a root finding problem, and the Robbins-Monro (RM) algorithm can solve the problem as follows:

$$
w_{k+1}=w_k - \alpha_k \~{g}(w_k,\eta_k) \quad k=1,2,3,...
$$

where:

- $w_k$ is the $k-$th estimate of the root
- $\~{g}(w_k,\eta_k)= g(w_k) + \eta_k$ is the $k-$th noisy observation. Why $\~{g}$ instead of $g$? Because we do not always know the true form of $g$, and many times we only have data/samples of $g$.
- $\alpha_k$ is a positive coefficient

To make it clear, consider the case where we need to estimate $\mathbb{E}[X]$. We do not know $X$, and we only have samples/data of $X$, that is $x \sim X$.

We can rephrase the problem into $g(w)=w-\mathbb{E}[X]$, where $w$ is our estimate of $\mathbb{E}[X]$. Then, finding $\mathbb{E}[X]$ is equivalent to finding the root of $g$, that is $w$ such that

$$
g(w)=0 \iff w=\mathbb{E}[X]
$$

Please keep in mind that we do not know $g$, since we do not know $X$. However, since we have samples of $X$, we may obtain the noisy observation $\~{g}$ as follows:

$$
\~{g}(w,\eta)=w-x
$$

$$
\~{g}(w,\eta)=w-x=(w-\mathbb{E}[X])+(\mathbb{E}[X]-x)=g(w)+\eta
$$

Then, by using the Robbins-Monro (RM) algorithm, we can obtain its estimation (incrementally/iteratively) as follows:

$$
w_{k+1}=w_k - \alpha_k \~{g}(w_k,\eta_k)=w_k-\alpha_k(w_k-x_k)
$$

Therefore, we have shown that the mean estimation problem can eventually be converted into a root finding problem. We will observe later that the TD algorithm takes a very similar form.

## Stochastic Gradient Descent

Suppose we would like to find the optimization of $J(w)$

$$
\min_{w} J(w) = \mathbb{E}[f(w,X)]
$$

where:

- $w$ is the parameters to be optimized
- $X$ is a random variable, and the expectation is with respect to $X$
- $w$ and $X$ can be either scalars or vectors, and the function $f(.)$ is a scalar

It is clearly an optimization problem, and the SGD algorithm can solve the problem as follows:

$$
w_{k+1}=w_k - \alpha_k \nabla_wf(w_k, x_k) \quad k=1,2,3,...
$$

Many problems can eventually be converted into optimization problems. To make it clear, consider the same case where we need to estimate $\mathbb{E}[X]$. We do not know $X$, and we only have samples/data of $X$, that is $x \sim X$.

We can rephrase the problem into

$$
\min_{w} J(w) = \mathbb{E}[f(w,X)]
$$

where $f(w,X)= \frac{1}{2} \Vert w-X \Vert ^2$, and $w$ is our estimate of $\min J$.

To find its optimized value, we must find the derivative of $f$, and we know that the derivative of $f$ is $\nabla_w f(w,X)=w-X$, and the optimal solution $w^*$ must satisfy

$$
\nabla_w J(w)=0
$$

$$
\nabla_w \mathbb{E}[f(w,X)] = \mathbb{E}[{\nabla_w f(w,X)}] = \mathbb{E}[w-X] = 0
$$

Then, finding $\mathbb{E}[X]$ is equivalent to solving the optimization problem of $J(w)$, that is $w$ such that

$$
w=\min J \iff w=\mathbb{E}[X]
$$

Therefore, we have shown that the mean estimation problem can eventually be converted into an optimization problem.

## Connection to Reinforcement Learning

Why are both examples mean estimation problems? Recall that in model-free reinforcement learning, we need to estimate the state (or action) value, which is the expected value of the discounted return starting from a state (and taking an action). For instance, in MC-based RL, we must complete sampling over an entire trajectory/episode before estimating the mean, which makes it non-incremental. In the next chapter, we revise this approach by using Temporal Difference (TD), which is very similar to the RM algorithm discussed in this chapter.
