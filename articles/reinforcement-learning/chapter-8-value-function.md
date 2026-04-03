# Chapter 8 Value Function

So far, we use the tabular representation of state value and action value. However, we know that the tabular representation can have some limitations such as:

1. difficult to handle when state value and action value become large
2. difficult to handle continuous spaces

Therefore, the whole idea is to have parameterized functions: $\hat{v}(s,w)$ to approximate $v_\pi(s)$, where $w\in \mathbb{R}^m$ is the parameter vector.

To find the optimal $w$, we need the objective function, which can be defined:

$$
J(w)=\mathbb{E}_{S \sim d_\pi}\left[ \left( v_\pi(S) - \hat{v}(S,w) \right)^2 \right]
$$

where $d_\pi$ is the stationary distribution, the distribution of the states under policy $\pi$ after long-run behavior. Then, we can apply stochastic gradient descent as:

$$
w_{t+1} = w_{t} + \alpha_{t}\left( v_\pi(s_t) - \hat{v}(s_t,w_t) \right) \nabla_w \hat{v}(s_t,w_t)
$$

For more details of its derivation, please refer to the slide number 22&44.

However, note that it is not feasible since it requires the true state value $v_\pi$, which is the unknown to be estimated.

We can replace $v_\pi(s_t)$ with an approximation:

1. First, Monte Carlo learning with function approximation
We can use the discounted return $g_t$ to approximate $v_\pi(s_t)$ and the algorithm becomes 
$w_{t+1} = w_{t} + \alpha_{t}\left( g_t - \hat{v}(s_t,w_t) \right) \nabla_w \hat{v}(s_t,w_t)$
2. Second, TD learning with function approximation
We can use the TD target $r_{t+1}+\gamma \hat{v}(s_{t+1},w_t)$ to approximate $v_\pi(s_t)$ and the algorithm becomes 
$w_{t+1} = w_{t} + \alpha_{t}\left( r_{t+1}+\gamma \hat{v}(s_{t+1},w_t) - \hat{v}(s_t,w_t) \right) \nabla_w \hat{v}(s_t,w_t)$
The second approach is more preferable

However, remember that to find optimal policy in model-free RL, we need to estimate action value, rather than state value. Fortunately, we can have very similar form of TD learning with function approximation to estimate action value:

$$
w_{t+1} = w_{t} + \alpha_{t}\left( r_{t+1}+\gamma \hat{q}(s_{t+1},a_{t+1},w_t) - \hat{q}(s_t,a_t,w_t) \right) \nabla_w \hat{q}(s_t,a_t,w_t)
$$

which in fact is also Sarsa algorithm with value function approximation.

Similar to Sarsa, tabular Q-learning can also be extended to the case of value function approximation:

$$
w_{t+1} = w_{t} + \alpha_{t}\left( r_{t+1}+\gamma \max_{a\in A(s_{t+1})} \hat{q}(s_{t+1},a,w_t) - \hat{q}(s_t,a_t,w_t) \right) \nabla_w \hat{q}(s_t,a_t,w_t)
$$

which can also be done either in on-policy or off-policy.

So far, the algorithms discussed use linear function, such as polynomial basis or fourier basis as the value approximation function. We can extend it to nonlinear function approximator such as neural networks. One of the earliest and most successful algorithms that introduce deep neural networks into RL is Deep Q-learning or Deep Q-network (DQN). 

The algorithms discussed so far have the objective function $J(w)=\mathbb{E}_{S \sim d_\pi}[\left( v_\pi(S) - \hat{v}(S,w) \right)^2]$, however, Deep Q-learning has different objective function:

$$
J(w)=\mathbb{E}_{}\left[ \left( R + \gamma \max_{a\in A(S')} \hat{q}(S',a,w) - \hat{q}(S,A,w) \right)^2 \right]
$$

where $(S,A,R,S’)$ are random variables. Unlike in the former $J(w)$ where $S \sim d_\pi$, in the later $J(w)$ we do not know any prior regarding $(S,A,R,S’)$. Therefore, we assume it is uniform distribution and use **experience replay buffer** to guarantee it is satisfied.

However, the objective function of Deep Q-learning is a bit tricky, since the parameter $w$ not only appears in $\hat{q}(S,A,w)$ but also in

$$
y = R + \gamma \max_{a\in A(S')} \hat{q}(S',a,w)
$$

To solve the problem, they introduce **two neural networks**:

- One is a main network representing $\hat{q}(s,a,w)$
- The other is a target network $\hat{q}(s,a,w_T)$

The objective function in this case degenerates to:

$$
J=\mathbb{E}_{}\left[ \left( R + \gamma \max_{a\in A(S')} \hat{q}(S',a,w_T) - \hat{q}(S,A,w) \right)^2 \right]
$$

where $w_T$ is the target network param and $w$ is the main network param. 

When $w_T$ is fixed, we can now easily compute the gradient of $J$ with respect to parameter $w$ as:

 

$$
\nabla_w J=\mathbb{E}_{}\left[ \left( R + \gamma \max_{a\in A(S')} \hat{q}(S',a,w_T) - \hat{q}(S,A,w) \right) \nabla_w \hat{q}(S,A,w) \right]
$$

Then, we can apply stochastic gradient descent to update the parameter.

If we take a look more closely, we emphasize value-based methods and value function methods where we commonly are required to estimate state or action value that would be used to update the policy. From the next chapter, we will have move:

1. from value-based methods to policy-based methods
2. from value function methods to policy function methods (or called as policy gradient methods)
