# Chapter 7 Temporal Difference

## Table of Contents

* [Bellman Expectation Equation](#bellman-expectation-equation)
* [TD via Robbins-Monro](#td-via-robbins-monro)
* [The TD Algorithm](#the-td-algorithm)
* [Action-Value TD Methods](#action-value-td-methods)
* [Remarks](#remarks)

## Bellman Expectation Equation

Recall from Chapter 2, the Bellman equation is given by

$$
v_\pi(s)=\mathbb{E}[R_{}+\gamma G_{}|S=s]
$$

Since we know that 

$$
\mathbb{E}[G|S=s] = \sum_{a} \pi(a|s) \sum_{s'}p(s'|s,a)v_{\pi}(s') = \mathbb{E}[v_{\pi}(S')|S=s]
$$

Then we can have the previous Bellman equation:

$$
v_\pi(s)=\mathbb{E}[R_{}+\gamma v_{\pi}(S')|s]
$$

It is another expression of the Bellman equation and is sometimes known as the Bellman expectation equation.

## TD via Robbins-Monro

Recall from Chapter 6, we can solve the Bellman expectation equation using the RM algorithm by defining:

$$
g(v(s)) = v(s) - \mathbb{E}[R_{}+\gamma v_{\pi}(S')|s]
$$

Then we can rewrite the Bellman expectation equation:

$$
g(v(s)) = 0
$$

Since we do not know the true value of the expectation because we can only obtain samples of $r$ and $s’$ from $R$ and $S’$, therefore, we can only have the noisy observation:

$$
\tilde{g}(v(s)) = v(s) - [r + \gamma v_{\pi}(s')]
$$

and we can show that it is the noisy observation of $g$ by rewriting the equation:

$$
\tilde{g}(v(s)) = g(v(s)) + \eta
$$

where 

$$
g(v(s)) = \left( v(s) - \mathbb{E}[R_{}+\gamma v_{\pi}(S')|s] \right)
$$

and

$$
\eta = \left( \mathbb{E}[R_{}+\gamma v_{\pi}(S')|s] - [r + \gamma v_{\pi}(s')] \right)
$$

Therefore, the RM algorithm for solving $g(v(s))=0$ is:

$$
v_{k+1}(s) = v_{k}(s) - \alpha \tilde{g}(v_k(s))
$$

$$
v_{k+1}(s) = v_{k}(s) - \alpha \left( v_k(s) - [r_k + \gamma v_{\pi}(s_k')] \right)
$$

where $v_k(s)$ is the estimate of $v_\pi(s)$ at the $k$th step; $r_k, s’_k$ are samples of $R,S’$ obtained at the $k$th step.

## The TD Algorithm

The TD algorithm looks very similar to the RM algorithm for solving $g(v(s))$, except:

1. $\{( s, r_k, s'_k )\}$ is changed to $\{( s_t, r_{t+1}, s_{t+1} )\}$ so that it can utilize sequential samples in an episode
2. $v_\pi(s’_k)$ is replaced by an estimate $v_t(s_{t+1})$

With that, we have the TD algorithm to estimate the state value:

$$
v_{t+1}(s_t) = v_t(s_t) - \alpha_t(s_t)[v_t(s_t) - [r_{t+1} + \gamma v_t(s_{t+1})]]
$$

where $v_{t}(s_t)$ is the estimated state value of $v_\pi(s_t)$ at time $t$; $\alpha_t(s_t)$ is the learning rate of $s_t$ at time $t$; $v_{t}(s_{t+1})$ is the estimated state value of $v_\pi(s_t)$ at time $t+1$.

Also, $[r_{t+1} + \gamma v_t(s_{t+1})]$ is known as the TD target ($\bar{v}_t$), and its TD error ($\delta_t$) is $v_t(s_t) - \bar{v}_t$.

Hence, we can rewrite the TD algorithm to estimate the state value:

$$
v_{t+1}(s_t) = v_t(s_t) - \alpha_t(s_t)[v_t(s_t) - \bar{v}_t]
$$

## Action-Value TD Methods

However, remember that to find the optimal policy in model-free RL, we need to estimate the action value rather than the state value. Fortunately, we can have a very similar form of the TD algorithm to estimate the action value:

$$
q_{t+1}(s_t, a_t) = q_t(s_t, a_t) - \alpha_t(s_t, a_t)[q_t(s_t, a_t) - \bar{q}_t]
$$

In fact, we can use the last equation as a unified point of view for the following model-free RL methods:

1. Sarsa with its TD target $\bar{q}_t = r_{t+1} + \gamma q_t(s_{t+1}, a_{t+1})$  
   It is known as Sarsa because it requires $(s_t, a_t, r_{t+1}, s_{t+1}, a_{t+1})$

2. n-step Sarsa with its TD target  
   $\bar{q}_t = r_{t+1} + \gamma r_{t+2} + ... + \gamma^{n-1} r_{t+n} + \gamma^{n} q_t(s_{t+n}, a_{t+n})$  
   It is known as n-step Sarsa because it requires sampling $r_{t+1}, r_{t+2}, …, r_{t+n}$

3. Q-learning with its TD target $\bar{q}_t = r_{t+1} + \gamma \max_a q_t(s_{t+1}, a)$  
   It is quite different from the other TD methods, where other methods estimate the action values of a given policy and then combine them with a policy improvement step to find the optimal policy. Q-learning directly estimates optimal action values and hence optimal policies.

4. Monte Carlo (non-incremental) with its TD target $\bar{q}_t = r_{t+1} + \gamma r_{t+2} + ...$  
   In n-step Sarsa, if $n \rightarrow \infty$, or when requiring sampling for the whole trajectory of an episode, then n-step Sarsa becomes Monte Carlo.

## Remarks

Some remarks on TD algorithms, specifically Q-learning, concern on-policy and off-policy methods. On-policy means that the behavior policy used to generate samples is the same as the target policy being optimized, whereas off-policy means that the behavior policy and the target policy are different.

We can use Q-learning with either on-policy or off-policy; however, the other approaches are on-policy.

Another remark is that while Monte Carlo optimizes the policy for every state, Sarsa and Q-learning only optimize the policy for the initial state to reach the target state.

That is all.
