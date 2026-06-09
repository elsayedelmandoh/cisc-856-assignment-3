"""
Assignment 3 — Q-learning Algorithm Implementation (10 points)

Specification:
  - Implement off-policy TD control (Q-learning).
  - States: grid positions (S = start, G = goal).
  - Actions: up (0), right (1), down (2), left (3).
  - Rewards: -5 for walls, +10 for goal, 0 otherwise.
  - Discount factor: γ = 0.9.
  - Update: Q(s,a) += α [r + γ max_a' Q(s',a') − Q(s,a)]
    where max is over ALL actions (greedy target) — NOT the action actually taken.
  - Behaviour policy is ε-greedy; target policy is greedy (off-policy).
"""

import numpy as np


class QLearningAgent():
    """
    off-policy TD control.
    update: Q(s,a) += alpha * [r + gamma * max_a' Q(s', a') - Q(s,a)] max over ALL actions — not what was taken
    key: behavior policy is epsilon-greedy (for exploration),
         target policy is greedy (max Q). they're DIFFERENT.
    result: Q converges to Q* regardless of exploration policy.
    """
 
    def __init__(self, number_of_states, actions, initial_state, step_size=0.1):
        self._state     = initial_state
        self._n_states  = number_of_states
        self._alpha     = step_size
        self._actions   = actions
        self._action    = self.initialAction()
        self._n_actions = len(actions)
 
        self._q = np.zeros((number_of_states, self._n_actions))
 
    @property
    def q_values(self):
        return self._q
 
    def initialAction(self):
        return self._actions(0)
 
    def greedy_policy_step(self, state, epsilon):
        """same epsilon-greedy — used for BEHAVIOR only in q-learning."""
        if np.random.random() < epsilon:
            action_idx = np.random.randint(self._n_actions)
        else:
            q_vals     = self._q[state]
            action_idx = np.random.choice(np.where(q_vals == q_vals.max())[0])
        return self._actions(action_idx)
 
    def step(self, reward, discount, next_state, epsilon):
        """
        q-learning update — off-policy:
        1. Q(s,a) += alpha * [r + gamma * MAX_a' Q(s',a') - Q(s,a)]
            key difference from sarsa
        2. choose next action from epsilon-greedy (for behavior/exploration)
        3. s <- s', a <- a'  (a' is epsilon-greedy, NOT max)
 
        the update uses max Q(s'), but the action TAKEN next can be exploratory.
        this decoupling is what makes it off-policy.
        """
        s = self._state
        a = self._action.value
 
        max_q_next = np.max(self._q[next_state])  
        td_target  = reward + discount * max_q_next
        td_error   = td_target - self._q[s, a]
        self._q[s, a] += self._alpha * td_error

        next_action  = self.greedy_policy_step(next_state, epsilon)
        self._state  = next_state
        self._action = next_action
 
        return self._action
 