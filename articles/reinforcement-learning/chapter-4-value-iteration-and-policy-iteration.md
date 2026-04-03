# Chapter 4 Value Iteration and Policy Iteration

This chapter introduces the concept of dynamic programming to solve the Bellman equation and the Bellman optimality equation.

## Table of Contents

* [Value Iteration](#value-iteration)
* [Policy Iteration](#policy-iteration)
* [Practical Considerations](#practical-considerations)

## Value Iteration

The first is Value iteration. It is an iterative algorithm to find the optimal policy by solving the Bellman optimality equation, given an initial value $$v_0$$. There are two steps in Value iteration:

1. Policy update (the matrix-vector form)

    $$
     \pi_{k+1}=\arg \max_\pi (r_\pi+\gamma P_\pi v_k)
    $$

2. Value update (the matrix-vector form)

    $$
    v_{k+1}=r_{\pi_{k+1}}+\gamma P_{\pi_{k+1}}v_k
    $$

    Please note that $$v_{k+1}$$ is not the state value, since it is not ensured that $$v_{k+1}$$ satisfies the Bellman equation. Also, the LHS index is $$k+1$$, while the RHS is $$k$$.

## Policy Iteration

The second is Policy iteration. It is an iterative algorithm to find the optimal policy by solving the Bellman equation, given an initial value $$\pi_0$$. There are two steps in Policy iteration:

1. Policy evaluation (the matrix-vector form)

    $$
    v_{\pi_k}=r_{\pi_{k}}+\gamma P_{\pi_{k}}v_k
    $$

    It satisfies the Bellman equation. In fact, policy evaluation itself is an iterative method for solving the Bellman equation.

2. Policy improvement (the matrix-vector form)

$$
 \pi_{k+1}=\arg \max_\pi (r_\pi+\gamma P_\pi v_k)
$$

## Practical Considerations

In an actual implementation of the algorithm, the elementwise form is used rather than the matrix-vector form. Here, the matrix-vector form is used for simplicity only.

Both methods are model-based reinforcement learning approaches, since they require knowledge of the model (probability distributions) $$p(r|s,a)$$ and $$p(s’|s,a)$$, which is not suitable for real-world cases.
