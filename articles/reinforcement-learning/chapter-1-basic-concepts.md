# Chapter 1 Basic Concepts

This chapter introduces the concept of a Markov Decision Process (MDP) and its key elements:

## Table of Contents

* [Sets](#sets)
* [Probability Distributions](#probability-distributions)

## Sets

* State set: the set of states $$S$$
* Action set: the set of actions $$A(s)$$ associated with each state
* Reward set: the set of rewards $$R(s, a)$$

## Probability Distributions

* State transition probability: at state $$s$$, taking action $$a$$, the probability of transitioning to state $$s’$$ is $$p(s’|s,a)$$. The state transition depends on the state and the action.
* Reward probability: at state $$s$$, taking action $$a$$, the probability of receiving reward $$r$$ is $$p(r|s,a)$$
* Policy probability: at state $$s$$, the probability of choosing action $$a$$ is $$\pi(a|s)$$

The state transition and reward depend on the state and the action, and the action chosen from a state depends on the policy.

The ultimate goal of reinforcement learning is to find the optimal policy.
