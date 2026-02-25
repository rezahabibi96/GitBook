# Chapter 2 State Values and Bellman Equation

This chapter introduces the concept of Bellman equation. The Bellman equation is important since it is used to evaluate the policy by calculating state value &/ action value.

State value is the expectation/mean of all possible returns that can be obtained starting from a state $$s$$:

$$
v_\pi(s)=\mathbb{E}[G_t=R_{t+1}+\gamma R_{t+2}+\gamma ^2 R_{t+3}+...|S_t=s]
$$

where the first term, $$R_{t+1}$$, is the immediate reward, while the rest terms are the future rewards. Return is the sum of the rewards obtained along a trajectory, discounted by a discount factor $$\lambda$$. Therefore, $$G_t$$ is the (discounted) return obtained along a trajectory and we can also write $$G_t$$:

$$
G_t=R_{t+1}+\gamma R_{t+2}+\gamma ^2 R_{t+3}+...
$$

$$
G_t=R_{t+1}+\gamma (R_{t+2}+\gamma  R_{t+3}+\gamma ^2 R_{t+4}+...)
$$

$$
G_t=R_{t+1}+\gamma G_{t+1}
$$

Since state value is the average return, the agent can get, starting from a state, then it can also be expressed through

$$
v_\pi(s)=\mathbb{E}[G_t|S_t=s]=\sum_a\mathbb{E}[G_t|s_t=s,A_t=a]\pi(a|s)
$$

where $$\sum_a\mathbb{E}[G_t|s_t=s,A_t=a]$$ is the action value.

Action value is the expectation/mean of all possible returns that can be obtained starting from a state $$s$$ and taking an action $$a$$:

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

It also shows the relationship between state and action value. The matrix-vector form of the Bellman equation given by: $$v_\pi=r_\pi + \gamma P_\pi v_\pi$$.

How to solve the Bellman equation? By iterative algorithm:

$$
v_{k+1}=r_\pi+\gamma P_\pi v_k
$$

In fact, it is known as Policy iteration which will be explained in Chapter 4.
