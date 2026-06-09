"""
Assignment 3 — SARSA Algorithm Implementation (10 points)

Specification:
  - Implement on-policy TD control (SARSA).
  - States: grid positions (S = start, G = goal).
  - Actions: up (0), right (1), down (2), left (3).
  - Rewards: -5 for walls, +10 for goal, 0 otherwise.
  - Discount factor: γ = 0.9.
  - Update: Q(s,a) += α [r + γ Q(s',a') − Q(s,a)]
    where a' is chosen by the SAME ε-greedy policy (on-policy).
"""

import numpy as np


class SarsaAgent():
    """
    on-policy TD control.
    update: Q(s,a) += alpha * [r + gamma * Q(s', a') - Q(s,a)]
                                              ^^^^
                                    a' is chosen by SAME epsilon-greedy policy
    key: both the behavior policy (what generates experience) and the
         target policy (what gets updated) are the same epsilon-greedy policy.
    """
 
    def __init__(self, number_of_states, actions, initial_state, step_size=0.1):
        self._state      = initial_state
        self._n_states   = number_of_states
        self._alpha      = step_size
        self._actions    = actions
        self._action     = self.initialAction()
        self._n_actions  = len(actions)
        self._q = np.zeros((number_of_states, self._n_actions))
 
    @property
    def q_values(self):
        return self._q
 
    def initialAction(self):
        return self._actions(0)  
 
    def greedy_policy_step(self, state, epsilon):
        """
        epsilon-greedy action selection.
        with prob epsilon: random (explore)
        with prob 1-epsilon: argmax Q(state, .) (exploit)
        """
        if np.random.random() < epsilon:
            action_idx = np.random.randint(self._n_actions)
        else:
            q_vals     = self._q[state]
            action_idx = np.random.choice(np.where(q_vals == q_vals.max())[0])
 
        return self._actions(action_idx)
 
    def step(self, reward, discount, next_state, epsilon):
        """
        sarsa update — on-policy:
        1. choose a' from s' using epsilon-greedy (same policy!)
        2. Q(s,a) += alpha * [r + gamma * Q(s',a') - Q(s,a)]
        3. s <- s', a <- a'
 
        note: discount=0 when terminal (goal reached) — this zeros out
        the future term automatically, no special terminal handling needed.
        """
        s = self._state
        a = self._action.value 
        next_action = self.greedy_policy_step(next_state, epsilon)
        a_prime     = next_action.value

        td_target = reward + discount * self._q[next_state, a_prime]
        td_error  = td_target - self._q[s, a]
        self._q[s, a] += self._alpha * td_error
        self._state  = next_state
        self._action = next_action
 
        return self._action