# Chapter 5 Monte Carlo Methods

This chapter introduces the concept of the earliest and most basic model-free reinforcement learning, Monte Carlo or MC-based RL. The key idea is to convert the policy iteration (chapter 3) to be model-free.

Recall that the policy improvement step in policy iteration (element-wise form), based on chapter 2 and 4, given by:

$$
 \pi_{k+1}=\arg \max_\pi \sum_a{\pi (a|s) q_{\pi_k} (s,a)}
$$

As we can see, to find the better policy, the key is to calculate $q_{\pi_k} (s,a)$.

There are two ways to express action value:

1. Expression 1 requires the model $p(r|s,a)$ and $p(s'|s,a)$
    
    $$
    q_{\pi_k} (s,a) = \sum_r{p(r|s,a)r + \gamma \sum_{s'}{p(s'|s,a)v_{\pi_k} (s')}}
    $$
    
    We can’t use expression 1.
    
2. Expression 2 does not require the model
    
    $$
    q_{\pi_k} (s,a)=\mathbb{E}[G_t|S_t=s, A_t=a]
    $$
    
    We can use expression 2 to obtain $q$ based on the data (samples/experiences)!
    

Since we don’t have the model and don’t know the true value of the expectation, and we only have the data, how do we obtain $q_{\pi_k} (s,a)$?

- Starting from $(s,a)$, following policy $\pi_k$, generate an episode
- The return of this episode is $g(s,a)$, which is a sample of $G_t$ in $q_{\pi_k} (s,a)=\mathbb{E}[G_t|S_t=s, A_t=a]$
- Suppose we have a set of episodes $\{g^{(j)}(s,a)\}$ then
    
    $$
    q_{\pi_k} (s,a)=\mathbb{E}[G_t|S_t=s, A_t=a]\approx \frac{1}{N}\sum_{i=1}^{N}{g^{(i)}(s,a)}
    $$
    

There are several variants of MC-based RL, such as MC basic, MC exploring start, MC $\epsilon$-greedy, but the underlying idea is similar among all of them. So, understanding MC basic is enough based on my perspective.

However, it is important to know about the greedy policies vs $\epsilon$-greedy policies: 

- In policy improvement or policy update step, the greedy policy always chose the policy with the highest action value for every state, leaving other actions underrepresented (less exploration, more exploitation)
- On the other hand, the $\epsilon$-greedy policy seeks to balance the tradeoff between exploitation vs exploration through the lens of $\epsilon$

If $\epsilon \rightarrow 0$, it becomes greedy (exploitation). However, when $\epsilon \rightarrow 1$, it becomes uniform (exploration)

$$
\pi(a|s) = 
\begin{cases}
1-\frac{\epsilon}{|A(s)|}(|A(s)|-1) &\quad \text{for the greedy action} \\
\frac{\epsilon}{|A(s)|} &\quad \text{for the other} |A(s)|-1 \text{ actions}
\end{cases}
$$

In MC-based RL, we need state, action, and reward set with an initial policy $\pi_0$.
