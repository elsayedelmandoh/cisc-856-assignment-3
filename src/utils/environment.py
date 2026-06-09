"""
Gridworld environment: Action enum, Grid class, and experiment runner.
"""

from enum import Enum

import numpy as np
import matplotlib.pyplot as plt


class Action(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class Grid():
    def __init__(self, discount=0.9, penalty_for_walls=-5):
        self._layout = np.array([
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1,  0,  0,  0,  0,  0, -1,  0,  0, -1],
            [-1,  0,  0,  0, -1,  0,  0,  0, 10, -1],
            [-1,  0,  0,  0, -1, -1,  0,  0,  0, -1],
            [-1,  0,  0,  0, -1, -1,  0,  0,  0, -1],
            [-1,  0,  0,  0,  0,  0,  0,  0,  0, -1],
            [-1,  0,  0,  0,  0,  0,  0,  0,  0, -1],
            [-1,  0,  0,  0,  0,  0,  0,  0,  0, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        ])
        self._start_state     = (2, 2)
        self._goal_state      = (2, 8)
        self._state           = self._start_state
        self._number_of_states = np.prod(np.shape(self._layout))
        self._discount        = discount
        self._penalty_for_walls = penalty_for_walls
        self._layout_dims     = self._layout.shape
        self._name            = 'Grid'

    @property
    def number_of_states(self):
        return self._number_of_states

    def plot_grid(self):
        fig, ax = plt.subplots(figsize=(6, 5))
        h, w = self._layout.shape
        ax.imshow(self._layout <= -1, interpolation="nearest", cmap='Blues', extent=[-0.5, w-0.5, h-0.5, -0.5])
        
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)
        ax.set_xticks(range(w))
        ax.set_yticks(range(h))
        ax.grid(True, alpha=0.3, color='black', linewidth=0.5)
        ax.set_title(self._name, fontsize=12, fontweight='bold')
        
        # Draw grid lines
        for y in range(h + 1):
            ax.axhline(y - 0.5, color='black', linewidth=1)
        for x in range(w + 1):
            ax.axvline(x - 0.5, color='black', linewidth=1)
        
        ax.text(self._start_state[1], self._start_state[0],
                 r"$\mathbf{S}$", ha='center', va='center', fontsize=16, color='green', fontweight='bold',
                 bbox=dict(boxstyle='circle', facecolor='lightgreen', alpha=0.7))
        ax.text(self._goal_state[1], self._goal_state[0],
                 r"$\mathbf{G}$", ha='center', va='center', fontsize=16, color='red', fontweight='bold',
                 bbox=dict(boxstyle='circle', facecolor='lightcoral', alpha=0.7))
        
        return fig

    def get_obs(self):
        y, x = self._state
        return y * self._layout.shape[1] + x

    def int_to_state(self, int_obs):
        x = int_obs % self._layout.shape[1]
        y = int_obs // self._layout.shape[1]
        return y, x

    def step(self, action: Action):
        y, x = self._state
        if action == Action.UP:
            new_state = (y - 1, x)
        elif action == Action.RIGHT:
            new_state = (y, x + 1)
        elif action == Action.DOWN:
            new_state = (y + 1, x)
        elif action == Action.LEFT:
            new_state = (y, x - 1)
        else:
            raise ValueError(f"invalid action: {action}")

        new_y, new_x = new_state
        if self._layout[new_y, new_x] == -1:      
            reward    = self._penalty_for_walls
            discount  = self._discount
            new_state = (y, x)                     
        elif self._layout[new_y, new_x] == 0:      
            reward   = 0.
            discount = self._discount
        else:                                      
            reward    = self._layout[new_y, new_x] 
            discount  = 0.                         
            new_state = self._start_state          

        self._state = new_state
        return reward, discount, self.get_obs()


def run_experiment(env, agent, number_of_steps, epsilon):
    """
    runs agent-env loop for `number_of_steps` steps.
    returns running mean reward (incremental update to avoid summing all rewards).
    also returns per-step reward history for plotting.
    """
    mean_reward  = 0.
    reward_hist  = np.zeros(number_of_steps)
    action       = agent.initialAction()

    for i in range(number_of_steps):
        reward, discount, next_state = env.step(action)
        action = agent.step(reward, discount, next_state, epsilon)
        mean_reward += (reward - mean_reward) / (i + 1.)
        reward_hist[i] = mean_reward

    return mean_reward, reward_hist
