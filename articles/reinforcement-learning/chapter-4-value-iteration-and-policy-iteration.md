# Chapter 4 Value Iteration and Policy Iteration

This chapter introduces the concept of dynamic programming to solve the Bellman equation and the Bellman optimality equation.

The first is Value iteration. It is an iterative algorithm to find optimal policy by solving the Bellman optimality equation, given an initial value $$v_0$$. There are two steps in Value iteration:

1. Policy update (the matrix-vector form) 
    
    $$
     \pi_{k+1}=\arg \max_\pi (r_\pi+\gamma P_\pi v_k)
    $$
    
2. value update (the matrix-vector form)
    
    $$
    v_{k+1}=r_{\pi_{k+1}}+\gamma P_{\pi_{k+1}}v_k
    $$
    
    Please, note that $$v_{k+1}$$ is not state value since it is not ensured that $$v_{k+1}$$ satisfies the Bellman equation. Also, the lhs index is $$k+1$$, while the rhs is $$k$$
    

The second is Policy iteration. It is an iterative algorithm to find optimal policy by solving the Bellman equation, given an initial value $$\pi_0$$. There are two steps in Value iteration:

1. Policy evaluation (the matrix-vector form)
    
    $$
    v_{\pi_k}=r_{\pi_{k}}+\gamma P_{\pi_{k}}v_k
    $$
    
    It satisfies the Bellman equation. In fact, the policy evaluation itself is an iterative method solving the Bellman equation. 
    
2. Policy improvement (the matrix-vector form)

$$
 \pi_{k+1}=\arg \max_\pi (r_\pi+\gamma P_\pi v_k)
$$

In an actual implementation of the algorithm, the element-wise form is used rather than its matrix-vector form. Here, the matrix-vector form used is for simplicity only.

Both of them is model-based reinforcement learning since they require to know the model (probability distribution) of $$p(r|s,a)$$ and $$p(s’|s,a)$$, which is not suitable for real word case.
