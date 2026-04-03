# Chapter 2 State Values and Bellman Equation

This chapter introduces the concept of the Bellman equation. The Bellman equation is important because it is used to evaluate a policy by calculating the state value and action value.

## Table of Contents

* [State Value](#state-value)
* [Action Value](#action-value)
* [Bellman Equation](#bellman-equation)
* [Solving the Bellman Equation](#solving-the-bellman-equation)

## State Value

State value is the expectation/mean of all possible returns that can be obtained starting from a state $$s$$:

$$
v_\pi(s)=\mathbb{E}[G_t=R_{t+1}+\gamma R_{t+2}+\gamma ^2 R_{t+3}+...|S_t=s]
$$

where the first term, $$R_{t+1}$$, is the immediate reward, while the remaining terms are the future rewards. The return is the sum of the rewards obtained along a trajectory, discounted by a discount factor $$\lambda$$. Therefore, $$G_t$$ is the (discounted) return obtained along a trajectory, and we can also write $$G_t$$ as:

$$
G_t=R_{t+1}+\gamma R_{t+2}+\gamma ^2 R_{t+3}+...
$$

$$
G_t=R_{t+1}+\gamma (R_{t+2}+\gamma  R_{t+3}+\gamma ^2 R_{t+4}+...)
$$

$$
G_t=R_{t+1}+\gamma G_{t+1}
$$

Since the state value is the average return that the agent can obtain starting from a state, it can also be expressed as:

$$
v_\pi(s)=\mathbb{E}[G_t|S_t=s]=\sum_a\mathbb{E}[G_t|s_t=s,A_t=a]\pi(a|s)
$$

where $$\sum_a\mathbb{E}[G_t|s_t=s,A_t=a]$$ is the action value.

## Action Value

Action value is the expectation/mean of all possible returns that can be obtained starting from a state $$s$$ and taking an action $$a$$:

$$
q_\pi(s,a)=\mathbb{E}[G_t|S_t=s, A_t=a]
$$

## Bellman Equation

The Bellman equation (elementwise form) is given by:

$$
v_\pi (s)=\sum_a{\pi (a|s) \left[\sum_r{p(r|s,a)r + \gamma \sum_{s'}{p(s'|s,a)v_\pi (s')}}\right]}
$$

$$
v_\pi (s) = \sum_a{\pi (a|s) q_\pi (s,a)}
$$

It also shows the relationship between the state value and the action value.

The matrix-vector form of the Bellman equation is given by:

$$
v_\pi=r_\pi + \gamma P_\pi v_\pi
$$

## Solving the Bellman Equation

How to solve the Bellman equation? By using an iterative algorithm:

$$
v_{k+1}=r_\pi+\gamma P_\pi v_k
$$

In fact, this is known as policy iteration, which will be explained in Chapter 4.
