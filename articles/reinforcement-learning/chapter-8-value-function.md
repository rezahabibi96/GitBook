# Chapter 8 Value Function

So far, we have used the tabular representation of the state value and action value. However, we know that the tabular representation has some limitations, such as:

1. difficulty in handling large state and action spaces
2. difficulty in handling continuous spaces

## Table of Contents

* [Function Approximation](#function-approximation)
* [Objective Function](#objective-function)
* [Monte Carlo and TD with Function Approximation](#monte-carlo-and-td-with-function-approximation)
* [Action-Value Function Approximation](#action-value-function-approximation)
* [Deep Q-Learning](#deep-q-learning)
* [Transition to Policy-Based Methods](#transition-to-policy-based-methods)

## Function Approximation

Therefore, the main idea is to use parameterized functions: $$\hat{v}(s,w)$$ to approximate $$v_\pi(s)$$, where $$w\in \mathbb{R}^m$$ is the parameter vector.

## Objective Function

To find the optimal $$w$$, we need an objective function, which can be defined as:

$$
J(w)=\mathbb{E}_{S \sim d_\pi}\left[ \left( v_\pi(S) - \hat{v}(S,w) \right)^2 \right]
$$

where $$d_\pi$$ is the stationary distribution, i.e., the distribution of states under policy $$\pi$$ after long-run behavior. Then, we can apply stochastic gradient descent as:

$$
w_{t+1} = w_{t} + \alpha_{t}\left( v_\pi(s_t) - \hat{v}(s_t,w_t) \right) \nabla_w \hat{v}(s_t,w_t)
$$

For more details of its derivation, please refer to slides 22 and 44.

However, note that this is not feasible since it requires the true state value $$v_\pi$$, which is unknown and needs to be estimated.

## Monte Carlo and TD with Function Approximation

We can replace $$v_\pi(s_t)$$ with an approximation:

1. First, Monte Carlo learning with function approximation  
   We can use the discounted return $$g_t$$ to approximate $$v_\pi(s_t)$$, and the algorithm becomes  
   $$w_{t+1} = w_{t} + \alpha_{t}\left( g_t - \hat{v}(s_t,w_t) \right) \nabla_w \hat{v}(s_t,w_t)$$

2. Second, TD learning with function approximation  
   We can use the TD target $$r_{t+1}+\gamma \hat{v}(s_{t+1},w_t)$$ to approximate $$v_\pi(s_t)$$, and the algorithm becomes  
   $$w_{t+1} = w_{t} + \alpha_{t}\left( r_{t+1}+\gamma \hat{v}(s_{t+1},w_t) - \hat{v}(s_t,w_t) \right) \nabla_w \hat{v}(s_t,w_t)$$  

The second approach is more preferable.

## Action-Value Function Approximation

However, remember that to find the optimal policy in model-free RL, we need to estimate the action value rather than the state value. Fortunately, we can obtain a very similar form of TD learning with function approximation to estimate the action value:

$$
w_{t+1} = w_{t} + \alpha_{t}\left( r_{t+1}+\gamma \hat{q}(s_{t+1},a_{t+1},w_t) - \hat{q}(s_t,a_t,w_t) \right) \nabla_w \hat{q}(s_t,a_t,w_t)
$$

which is, in fact, the Sarsa algorithm with value function approximation.

Similarly, tabular Q-learning can also be extended to the case of value function approximation:

$$
w_{t+1} = w_{t} + \alpha_{t}\left( r_{t+1}+\gamma \max_{a\in A(s_{t+1})} \hat{q}(s_{t+1},a,w_t) - \hat{q}(s_t,a_t,w_t) \right) \nabla_w \hat{q}(s_t,a_t,w_t)
$$

which can also be applied in either on-policy or off-policy settings.

## Deep Q-Learning

So far, the algorithms discussed use linear functions, such as polynomial basis or Fourier basis, as the value approximation function. We can extend this to nonlinear function approximators such as neural networks. One of the earliest and most successful algorithms that introduces deep neural networks into RL is Deep Q-learning or Deep Q-Network (DQN).

The algorithms discussed so far have the objective function:

$$
J(w)=\mathbb{E}_{S \sim d_\pi}\left[ \left( v_\pi(S) - \hat{v}(S,w) \right)^2 \right]
$$

However, Deep Q-learning has a different objective function:

$$
J(w)=\mathbb{E}_{}\left[ \left( R + \gamma \max_{a\in A(S')} \hat{q}(S',a,w) - \hat{q}(S,A,w) \right)^2 \right]
$$

where $$(S,A,R,S’)$$ are random variables. Unlike the former $$J(w)$$ where $$S \sim d_\pi$$, in the latter $$J(w)$$ we do not know any prior regarding $$(S,A,R,S’)$$. Therefore, we assume a uniform distribution and use an **experience replay buffer** to approximate this condition.

However, the objective function of Deep Q-learning is somewhat tricky, since the parameter $$w$$ appears not only in $$\hat{q}(S,A,w)$$ but also in

$$
y = R + \gamma \max_{a\in A(S')} \hat{q}(S',a,w)
$$

To address this issue, two neural networks are introduced:

- One is a main network representing $$\hat{q}(s,a,w)$$
- The other is a target network $$\hat{q}(s,a,w_T)$$

The objective function in this case becomes:

$$
J=\mathbb{E}_{}\left[ \left( R + \gamma \max_{a\in A(S')} \hat{q}(S',a,w_T) - \hat{q}(S,A,w) \right)^2 \right]
$$

where $$w_T$$ is the target network parameter and $$w$$ is the main network parameter.

When $$w_T$$ is fixed, we can compute the gradient of $$J$$ with respect to parameter $$w$$ as:

$$
\nabla_w J=\mathbb{E}_{}\left[ \left( R + \gamma \max_{a\in A(S')} \hat{q}(S',a,w_T) - \hat{q}(S,A,w) \right) \nabla_w \hat{q}(S,A,w) \right]
$$

Then, we can apply stochastic gradient descent to update the parameter.

## Transition to Policy-Based Methods

If we look more closely, we emphasize value-based methods and value function methods, where we are required to estimate the state or action value to update the policy. In the next chapter, we will move:

1. from value-based methods to policy-based methods  
2. from value function methods to policy function methods (also called policy gradient methods)
