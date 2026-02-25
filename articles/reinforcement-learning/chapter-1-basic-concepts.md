# Chapter 1 Basic Concepts

This chapter introduces the concept of Markov Decision Process (MDP) with its key elements:

Sets:

* State set: the set of states $$S$$
* Action set: the set of actions $$A(s)$$ is associated for every state
* Reward set: the set of rewards $$R(s, a)$$

Probability distribution:

* State transition probability: at state $$s$$, taking action $$a$$, the probability to transit to state $$s’$$ is $$p(s’|s,a)$$. The state transition depends on the state and action
* Reward probability: at state $$s$$, taking action $$a$$, the probability to get reward $$r$$ is $$p(r|s,a)$$
* Policy probability: at state $$s$$, the probability to choose action $$a$$ is $$\pi(a|s)$$

The state transition and reward depend on the state and action, and the action to choose from the state depends on the policy.

The ultimate goal of reinforcement learning is to find the optimal policy.
