# Chapter 3 Optimal State Values and Bellman Optimality Equation

This chapter introduces the concept of optimal policy and the Bellman optimality equation. Recall that the ultimate goal of reinforcement learning is to find the optimal policy. 

A policy $$\pi^*$$ is optimal if $$v_{\pi^*}(s) \geq v_{\pi}(s)$$ for all $$s$$ and for any other policy $$\pi$$.

The Bellman optimality equation (elementwise form) defined as: 

$$
v_\pi (s) = \max_\pi \sum_a{\pi (a|s) \left(\sum_r{p(r|s,a)r + \gamma \sum_{s'}{p(s'|s,a)v_\pi (s')}}\right)}
$$

and its matrix-vector form defined as: $$v_\pi=\max_\pi (r_\pi + \gamma P_\pi v_\pi)$$

The Bellman optimality equation is important because its solution corresponds to the optimal state value and optimal policy. The existence and uniqueness of the equation is guaranteed by the contraction mapping theorem.

How to solve the Bellman optimality equation? By iterative algorithm:

$$
v_{k+1}=\max_\pi {(r_\pi+\gamma P_\pi v_k)}
$$

In fact, it is known as Value iteration which will be explained in Chapter 4
