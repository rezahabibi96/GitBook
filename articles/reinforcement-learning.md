---
description: >-
  This is my summary and study notes based on the book Mathematical Foundations
  of Reinforcement Learning
---

# Reinforcement Learning

## Chapter 1 Basic Concepts

This chapter introduces the concept of Markov Decision Process (MDP) with its key elements:

Sets:

* State set: the set of states $S$
* Action set: the set of actions $A(s)$ is associated for every state
* Reward set: the set of rewards $R(s, a)$

Probability distribution:

* State transition probability: at state $s$, taking action $a$, the probability to transit to state $s’$ is $p(s’|s,a)$. The state transition depends on the state and action
* Reward probability: at state $s$, taking action $a$, the probability to get reward $r$ is $p(r|s,a)$
* Policy probability: at state $s$, the probability to choose action $a$ is $\pi(a|s)$

The state transition and reward depend on the state and action, and the action to choose from the state depends on the policy.

The ultimate goal of reinforcement learning is to find the optimal policy.

## Chapter 2 State Values and Bellman Equation

This chapter introduces the concept of Bellman equation. The Bellman equation is important since it is used to evaluate the policy by calculating state value &/ action value.

State value is the expectation/mean of all possible returns that can be obtained starting from a state $s$:

$$
v_\pi(s)=\mathbb{E}[G_t=R_{t+1}+\gamma R_{t+2}+\gamma ^2 R_{t+3}+...|S_t=s]
$$

where the first term, $R\_{t+1}$, is the immediate reward, while the rest terms are the future rewards. Return is the sum of the rewards obtained along a trajectory, discounted by a discount factor $\lambda$. Therefore, $$G_t$$  is the (discounted) return obtained along a trajectory and we can also write $G\_t$:&#x20;

$$
G_t=R_{t+1}+\gamma R_{t+2}+\gamma ^2 R_{t+3}+...
$$

$$
G_t=R_{t+1}+\gamma (R_{t+2}+\gamma  R_{t+3}+\gamma ^2 R_{t+4}+...)
$$

$$
G_t=R_{t+1}+G_{t+1}
$$

Since state value is the average return, the agent can get starting from a state, then it can also be expressed through

$$
v_\pi(s)=\mathbb{E}[G_t|S_t=s]=\sum_a\mathbb{E}[G_t|s_t=s,A_t=a]\pi(a|s)
$$

where $\sum\_a\mathbb{E}\[G\_t|s\_t=s,A\_t=a]$ is the action value.

Action value is the expectation/mean of all possible returns that can be obtained starting from a state $s$ and taking an action $a$:

$$
q_\pi(s,a)=\mathbb{E}[G_t|S_t=s, A_t=a]
$$

The Bellman equation (elementwise form) given by:

$$
v_\pi (s)=\sum_a{\pi (a|s) \left[\sum_r{p(r|s,a)r + \gamma \sum_{s'}{p(s'|s,a)v_\pi (s')}}\right]}
$$

$$
v_\pi (s) = \sum_a{\pi (a|s) q_\pi (s,a)}
$$

It also shows the relationship between state and action value. The matrix-vector form of the Bellman equation given by: $v\_\pi=r\_\pi + \gamma P\_\pi v\_\pi$.

How to solve the Bellman equation? By iterative algorithm:

$$
v_{k+1}=r_\pi+\gamma P_\pi v_k
$$

In fact, it is known as Policy iteration which will be explained in Chapter 4.

## Chapter 3 Optimal State Values and Bellman Optimality Equation

This chapter introduces the concept of optimal policy and the Bellman optimality equation. Recall that the ultimate goal of reinforcement learning is to find the optimal policy.

A policy $\pi^_$ is optimal if $v\_{\pi^_}(s) \geq v\_{\pi}(s)$ for all $s$ and for any other policy $\pi$.

The Bellman optimality equation (elementwise form) defined as:

$$
v_\pi (s) = \max_\pi \sum_a{\pi (a|s) \left(\sum_r{p(r|s,a)r + \gamma \sum_{s'}{p(s'|s,a)v_\pi (s')}}\right)}
$$

and its matrix-vector form defined as: $v\_\pi=\max\_\pi (r\_\pi + \gamma P\_\pi v\_\pi)$

The Bellman optimality equation is important because its solution corresponds to the optimal state value and optimal policy. The existence and uniqueness of the equation is guaranteed by the contraction mapping theorem.

How to solve the Bellman optimality equation? By iterative algorithm:

$$
v_{k+1}=\max_\pi {(r_\pi+\gamma P_\pi v_k)}
$$

In fact, it is known as Value iteration which will be explained in Chapter 4

## Chapter 4 Value Iteration and Policy Iteration

This chapter introduces the concept of dynamic programming to solve the Bellman equation and the Bellman optimality equation.

The first is Value iteration. It is an iterative algorithm to find optimal policy by solving the Bellman optimality equation, given an initial value $v\_0$. There are two steps in Value iteration:

1.  Policy update (the matrix-vector form)

    $$
    \pi_{k+1}=\arg \max_\pi (r_\pi+\gamma P_\pi v_k)
    $$
2.  value update (the matrix-vector form)

    $$
    v_{k+1}=r_{\pi_{k+1}}+\gamma P_{\pi_{k+1}}v_k
    $$

    Please, note that $v\_{k+1}$ is not state value since it is not ensured that $v\_{k+1}$ satisfies the Bellman equation. Also, the lhs index is $k+1$, while the rhs is $k$

The second is Policy iteration. It is an iterative algorithm to find optimal policy by solving the Bellman equation, given an initial value $\pi\_0$. There are two steps in Value iteration:

1.  Policy evaluation (the matrix-vector form)

    $$
    v_{\pi_k}=r_{\pi_{k}}+\gamma P_{\pi_{k}}v_k
    $$

    It satisfies the Bellman equation. In fact, the policy evaluation itself is an iterative method solving the Bellman equation.
2. Policy improvement (the matrix-vector form)

$$
\pi_{k+1}=\arg \max_\pi (r_\pi+\gamma P_\pi v_k)
$$

In an actual implementation of the algorithm, the element-wise form is used rather than its matrix-vector form. Here, the matrix-vector form used is for simplicity only.

Both of them is model-based reinforcement learning since they require to know the model (probability distribution) of $p(r|s,a)$ and $p(s’|s,a)$, which is not suitable for real word case.

## Chapter 5 Monte Carlo Methods

This chapter introduces the concept of the earliest and most basic model-free reinforcement learning, Monte Carlo or MC-based RL. The key idea is to convert the policy iteration (chapter 3) to be model-free.

Recall that the policy improvement step in policy iteration (element-wise form), based on chapter 2 and 4, given by:

$$
\pi_{k+1}=\arg \max_\pi \sum_a{\pi (a|s) q_{\pi_k} (s,a)}
$$

As we can see, to find the better policy, the key is to calculate $q\_{\pi\_k} (s,a)$.

There are two ways to express action value:

1.  Expression 1 requires the model $p(r|s,a)$ and $p(s'|s,a)$

    $$
    q_{\pi_k} (s,a) = \sum_r{p(r|s,a)r + \gamma \sum_{s'}{p(s'|s,a)v_{\pi_k} (s')}}
    $$

    We can’t use expression 1.
2.  Expression 2 does not require the model

    $$
    q_{\pi_k} (s,a)=\mathbb{E}[G_t|S_t=s, A_t=a]
    $$

    We can use expression 2 to obtain $q$ based on the data (samples/experiences)!

Since we don’t have the model and don’t know the true value of the expectation, and we only have the data, how do we obtain $q\_{\pi\_k} (s,a)$?

* Starting from $(s,a)$, following policy $\pi\_k$, generate an episode
* The return of this episode is $g(s,a)$, which is a sample of $G\_t$ in $q\_{\pi\_k} (s,a)=\mathbb{E}\[G\_t|S\_t=s, A\_t=a]$
*   Suppose we have a set of episodes ${g^{(j)}(s,a)}$ then

    $$
    q_{\pi_k} (s,a)=\mathbb{E}[G_t|S_t=s, A_t=a]\approx \frac{1}{N}\sum_{i=1}^{N}{g^{(i)}(s,a)}
    $$

There are several variants of MC-based RL, such as MC basic, MC exploring start, MC $\epsilon$-greedy, but the underlying idea is similar among all of them. So, understanding MC basic is enough based on my perspective.

However, it is important to know about the greedy policies vs $\epsilon$-greedy policies:

* In policy improvement or policy update step, the greedy policy always chose the policy with the highest action value for every state, leaving other actions underrepresented (less exploration, more exploitation)
* On the other hand, the $\epsilon$-greedy policy seeks to balance the tradeoff between exploitation vs exploration through the lens of $\epsilon$

If $\epsilon \rightarrow 0$, it becomes greedy (exploitation). However, when $\epsilon \rightarrow 1$, it becomes uniform (exploration)

$$
\pi(a|s) = 
\begin{cases}
1-\frac{\epsilon}{|A(s)|}(|A(s)|-1) &\quad \text{for the greedy action} \\
\frac{\epsilon}{|A(s)|} &\quad \text{for the other} |A(s)|-1 \text{ actions}
\end{cases}
$$

In MC-based RL, we need state, action, and reward set with an initial policy $\pi\_0$.

## Chapter 6 Stochastics Approximation

This chapter introduces the concept of stochastic approximation (SA), with two examples: the Robbins–Monro (RM) algorithm and stochastic gradient descent. It serves as the foundation for the next RL algorithm: Temporal Difference (TD).

Stochastic approximation (SA) refers to a broad class of stochastic iterative algorithms solving **root finding** or **optimization problems**.

Many problems can eventually be converted to root finding problem. Suppose we are given the problem of how to calculate the mean $\bar x$, that is, $\mathbb{E}\[X]\approx \bar x$ ?

*   the first way, which is trivial, is to collect all the samples then calculate the avg, similar to what MC do.

    $$
    \bar x = \frac{1}{N}\sum_{i=1}^{N}x_i
    $$

    The drawback is that we have to wait until all the samples to be collected. In fact, it is the disadvantage of MC compared to TD, which supports incremental or iterative estimation
* the second way, by using Robbins-Monro (RM) algorithm, but how?

RM

Suppose we would like to find the root of the equation

$$
g(w)=0,
$$

where $g:\R \rightarrow \R$ is a function and $w \in \R$ is the variable to be solved (the root).

It is clearly a root finding problem, and the Robbins-Monro (RM) algorithm can solve the problem as follows:

$$
w_{k+1}=w_k - a_k \~{g}(w_k,\eta_k) \quad k=1,2,3,...
$$

where:

* $w\_k$ is the $k-$th estimate of the root
* $\~{g}(w\_k,\eta\_k)= g(w\_k) + \eta\_k$ is the $k-$th noisy observation. Why $\~{g}$ instead of $g$? Because we don’t always know the true form of $g$, and many times we only have the data/samples of the $g$.
* $a\_k$ is a positive coefficient

To make it clear, consider the case where we need to estimate $\mathbb{E}\[X]$. We don’t know $X$ and we only have samples/data of $X$, that is $x \sim X$.

We can rephrase the problem into $g(w)=w-\mathbb{E}\[X]$ where $w$ is our estimate of $\mathbb{E}\[X]$. Then, finding $\mathbb{E}\[X]$ is equal to finding root of $g$, that is $w$ such that

$$
g(w)=w-\mathbb{E}[X]=0 \iff w=\mathbb{E}[X]
$$

Please, keep in mind that we don’t know $g$ since we don’t know $X$. However, since we have the samples of $X$, we may get the noisy observation $\~{g}$ as follows:

$$
\~{g}(w,\eta)=w-x
$$

$$
\~{g}(w,\eta)=w-x=(w-\mathbb{E}[X])+(\mathbb{E}[X]-x)=g(w)+\eta
$$

Then by using Robbins-Monro (RM) algorithm, we can get its estimation (incrementally/iteratively) as follows:

$$
w_{k+1}=w_k - a_k \~{g}(w_k,\eta_k)=w_k-\alpha_k(w_k-x_k)
$$

Therefor we have shown that mean estimation problem can eventually be converted to root finding problem. We will observe later that the TD algorithm takes a very similar form.

SGD

Next consider the case where we need to optimize
