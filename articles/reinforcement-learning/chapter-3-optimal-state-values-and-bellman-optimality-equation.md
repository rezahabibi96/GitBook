# Chapter 3 Optimal State Values and Bellman Optimality Equation

This chapter introduces the concept of optimal policy and the Bellman optimality equation. Recall that the ultimate goal of reinforcement learning is to find the optimal policy.

A policy $\pi^_$ is optimal if $v\_{\pi^_}(s) \geq v\_{\pi}(s)$ for all $s$ and for any other policy $\pi$.

The Bellman optimality equation (elementwise form) defined as:

\$$ v\_\pi (s) = \max\_\pi \sum\_a{\pi (a|s) \left(\sum\_r{p(r|s,a)r + \gamma \sum\_{s'}{p(s'|s,a)v\_\pi (s')\}}\right)} \$$

and its matrix-vector form defined as: $v\_\pi=\max\_\pi (r\_\pi + \gamma P\_\pi v\_\pi)$

The Bellman optimality equation is important because its solution corresponds to the optimal state value and optimal policy. The existence and uniqueness of the equation is guaranteed by the contraction mapping theorem.

How to solve the Bellman optimality equation? By iterative algorithm:

\$$ v\_{k+1}=\max\_\pi {(r\_\pi+\gamma P\_\pi v\_k)} \$$

In fact, it is known as Value iteration which will be explained in Chapter 4
