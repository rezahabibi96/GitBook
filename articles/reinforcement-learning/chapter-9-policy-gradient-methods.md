# Chapter 9 Policy Gradient Methods

So far, policies have always been represented by tables. However, similar to state value and action value, policies can also be represented by parameterized function $\pi(a|s,\theta)$, where $\theta \in \mathbb{R}^m$ is a parameter vector.

For value-based methods, our goal is to find a parameterized function $\hat{v}(s,w)$ that best approximates the true state value $v(s)$, where $w\in \mathbb{R}^m$ is the parameter vector. In contrast, for policy-based methods, the objective of the parameterized policy $\pi(a|s,\theta)$ is to best approximate the optimal policy $\pi^*$. 

Note that in the tabular case, a policy $\pi$ is optimal if it can maximize every state value. However, in the function case, a policy $\pi$ is optimal if it can maximize certain scalar metrics. These scalar metrics can be:

1. Average value $\bar{v}_\pi$. 

There are several ways to express this:

- Expression 1: $\sum_{s\in S}{d(s)v_\pi (s)}$
- Expression 2: $\mathbb{E}_{S\sim d}[v_\pi (S)]$
- Expression 3: $\lim _{n \rightarrow \infty}{\mathbb{E}\left[\sum_{t=0}^{n} \gamma ^t R_{t+1}\right]}$

where $\bar{v}_\pi$ is a weighted average of the state value or simply called average value. The $d(s)$ is a probability distribution of state $s$ that can either be uniform (independent of $\pi$) or long-run stationary distribution under $\pi$.

2. Average reward $\bar{r}_\pi$.

- Expression 1: $\sum_{s\in S}{d_\pi(s)r_\pi (s)}$
- Expression 2: $\mathbb{E}_{S\sim d_\pi}[r_\pi (S)]$
- Expression 3: $\lim _{n \rightarrow \infty}{\frac{1}{n} \mathbb{E}\left[\sum_{t=0}^{n-1} \gamma ^t R_{t+1}\right]}$

where $\bar{r}_\pi$ is a weighted average of the immediate reward or simply called average reward, $r_\pi(s)$ is the average immediate reward that can be obtained from state $s$, and $d_\pi$ is the long-run stationary distribution under $\pi$.

The relationship between $\bar{v}_\pi$ and $\bar{r}_\pi$ is given by $\bar{r}_\pi=(1-\gamma) \bar{v}_\pi$, so they can be maximized simultaneously. After knowing the possible metrics for the optimal policy (in the case of function approximation), we can search the optimal values of  $\theta$ such that $\pi(a|s,\theta)$ can maximize these metrics.

Knowing this, we can apply gradient-based optimization methods to maximize $J(\theta)$. The corresponding gradient expression is given below (see the textbook for the detailed proof):

$$
\nabla _\theta J(\theta) = \sum_{s\in S} \eta(s) \sum_{a\in A} \nabla _\theta \pi(a|s,\theta)q_\pi(s,a)
$$

where $J(\theta)$ can either be to maximize $\bar{v}_{\pi _\theta}$ or to maximize $\bar{r}_{\pi _\theta}$ and $\eta$ is a distribution or weight of the states.  We can also express it in the form of expectation (again please see the textbook for the detailed proof) so we can apply stochastic gradient:

$$
\nabla _\theta J(\theta) = \mathbb{E}_{S\sim \eta, A\sim \pi} \left[\nabla _\theta \ln \pi(A|S,\theta)q_\pi(S,A)\right]
$$

By using stochastic gradient, we can approximate the gradient:

$$
\nabla _\theta J(\theta) = \nabla _\theta \ln \pi(a|s,\theta)q_\pi(s,a)
$$

where $s, a$ are samples. It is required by $\ln \pi (a|s,\theta)$ that $\pi(a,|s,\theta)>0$ for any $s,a,\theta$.  We can use softmax function at the last layer to satisfy it.

Therefore, we can update the values of  $\theta$ using the stochastic gradient ascent as follow:

$$
\theta_{t+1} = \theta _t + \alpha \nabla _\theta \ln \pi(a_t|s_t,\theta _t)q_\pi(s_t,a_t)
$$

However, since $q_\pi$ is unknown, we can replace it by an estimate:

$$
\theta_{t+1} = \theta _t + \alpha \nabla _\theta \ln \pi(a_t|s_t,\theta _t)q_t(s_t,a_t)
$$

If $q_\pi(s_t,a_t)$ is estimated by Monte Carlo, it is known as REINFORCE, one of the earliest and simplest policy gradient algorithms. If $q_\pi(s_t,a_t)$ is estimated by TD learning, it is known as Actor-Critic, which will be introduced next.

REINFORECE is an on-policy gradient algorithm because $A \sim \pi(A|S,\theta)$ hence it requires $a_t$ to be sampled following $\pi (s_t, \theta_t)$.

Since $\nabla _\theta \ln \pi(a_t|s_t,\theta _t) = \frac{\nabla _\theta \pi(a_t|s_t,\theta _t)}{\pi(a_t|s_t,\theta _t)}$, we can also rewrite:

$$
\theta_{t+1} = \theta _t + \alpha \nabla _\theta \ln \pi(a_t|s_t,\theta _t)q_t(s_t,a_t)
$$

as follow:

$$
\theta_{t+1} = \theta _t + \alpha \beta_t \nabla _\theta \pi(a_t|s_t,\theta _t)
$$

where $\beta _t = \frac{q_t(s_t,a_t)}{\pi(a_t|s_t,\theta _t)}$ and it can be interpreted to balance between exploration and exploitation.

Since $\pi(a,|s,\theta)>0$  and  $a_t$ is sampled following $\pi (s_t, \theta_t)$, therefore, the parameterized policy is stochastic. There also exist deterministic policy gradient methods that will be discussed in the next lecture.

Unlike REINFORCE (stochastic policy gradient method), deterministic policy gradient method is off-policy.
