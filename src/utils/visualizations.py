"""
Visualization utilities for gridworld value functions and policies.
"""

import numpy as np
import matplotlib.pyplot as plt

from src.utils.environment import Action


def map_from_action_to_subplot(a):
    """Map an Action enum value to a subplot position (2, 6, 8, or 4)."""
    return (2, 6, 8, 4)[a.value]


def plot_values(values, colormap='pink', vmin=-1, vmax=10):
    plt.imshow(values, interpolation="nearest", cmap=colormap, vmin=vmin, vmax=vmax)
    plt.yticks([])
    plt.xticks([])
    plt.colorbar(ticks=[vmin, vmax])


def plot_state_value(action_values, epsilon=0.1):
    q    = action_values
    fig  = plt.figure(figsize=(4, 4))
    vmin = np.min(action_values)
    vmax = np.max(action_values)
    v = (1 - epsilon) * np.max(q, axis=-1) + epsilon * np.mean(q, axis=-1)
    plot_values(v, colormap='summer', vmin=vmin, vmax=vmax)
    plt.title('state value')
    plt.colorbar()


def plot_action_values(action_values, epsilon=0.1):
    q    = action_values
    vmin = np.min(action_values)
    vmax = np.max(action_values)
    fig  = plt.figure(figsize=(8, 8))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    for action in Action:
        plt.subplot(3, 3, map_from_action_to_subplot(action))
        plot_values(q[..., action.value], vmin=vmin, vmax=vmax)
        action_name = action.name
        plt.title(f'Q({action_name})')
    plt.subplot(3, 3, 5)
    v = (1 - epsilon) * np.max(q, axis=-1) + epsilon * np.mean(q, axis=-1)
    plot_values(v, colormap='summer', vmin=vmin, vmax=vmax)
    plt.title('V(s)')


def visualise_policy(grid, agent):
    """draws arrows showing the greedy policy at each non-wall cell."""
    arrows = {0: '↑', 1: '→', 2: '↓', 3: '←'}
    q = agent.q_values
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(grid._layout <= -1, cmap='Blues', interpolation='nearest')
    h, w = grid._layout.shape
    for y in range(h):
        for x in range(w):
            state = y * w + x
            if grid._layout[y, x] != -1:
                best_a = np.argmax(q[state])
                ax.text(x, y, arrows[best_a], ha='center', va='center',
                        fontsize=12, color='darkblue')
    ax.text(grid._start_state[1], grid._start_state[0], 'S',
            ha='center', va='center', fontsize=14, color='green', fontweight='bold')
    ax.text(grid._goal_state[1], grid._goal_state[0], 'G',
            ha='center', va='center', fontsize=14, color='red', fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('greedy policy')
    for yy in range(h - 1):
        ax.plot([-0.5, w - 0.5], [yy + 0.5, yy + 0.5], '-k', lw=1)
    for xx in range(w - 1):
        ax.plot([xx + 0.5, xx + 0.5], [-0.5, h - 0.5], '-k', lw=1)
