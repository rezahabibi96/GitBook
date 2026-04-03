# Chapter 3 Optimal State Values and Bellman Optimality Equation

This chapter introduces the concept of the optimal policy and the Bellman optimality equation. Recall that the ultimate goal of reinforcement learning is to find the optimal policy.

## Table of Contents

* [Optimal Policy](#optimal-policy)
* [Bellman Optimality Equation](#bellman-optimality-equation)
* [Solving the Bellman Optimality Equation](#solving-the-bellman-optimality-equation)

## Optimal Policy

A policy $$\pi^*$$ is optimal if $$v_{\pi^*}(s) \geq v_{\pi}(s)$$ for all $$s$$ and for any other policy $$\pi$$.

## Bellman Optimality Equation

The Bellman optimality equation (elementwise form) is defined as:

$$
v_\pi (s) = \max_\pi \sum_a{\pi (a|s) \left(\sum_r{p(r|s,a)r + \gamma \sum_{s'}{p(s'|s,a)v_\pi (s')}}\right)}
$$

and its matrix-vector form is defined as:

$$
v_\pi=\max_\pi (r_\pi + \gamma P_\pi v_\pi)
$$

The Bellman optimality equation is important because its solution corresponds to the optimal state value and the optimal policy. The existence and uniqueness of the equation are guaranteed by the contraction mapping theorem.

## Solving the Bellman Optimality Equation

How can the Bellman optimality equation be solved? By using an iterative algorithm:

$$
v_{k+1}=\max_\pi {(r_\pi+\gamma P_\pi v_k)}
$$

In fact, this is known as value iteration, which will be explained in Chapter 4.
