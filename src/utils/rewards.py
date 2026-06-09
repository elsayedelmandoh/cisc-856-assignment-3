"""
Reward analysis: epsilon comparison experiments and plotting.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from src.utils.environment import Grid, Action, run_experiment
from src.config.config import settings, FIGURES_DIR


def run_epsilon_comparison(AgentClass, num_steps=100_000, n_runs=5,
                           epsilons=(0.1, 0.5, 1.0)):
    """
    runs AgentClass for each epsilon, averages over n_runs seeds.
    returns dict: epsilon -> smoothed reward curve
    """
    results = {}

    for eps in epsilons:
        all_hists = []
        for run in range(n_runs):
            np.random.seed(run * 42)
            env   = Grid()
            agent = AgentClass(
                number_of_states=env._layout.size,
                actions=Action,
                initial_state=env.get_obs(),
                step_size=0.1
            )
            _, hist = run_experiment(env, agent, num_steps, eps)
            all_hists.append(hist)

        mean_hist = np.mean(all_hists, axis=0)
        results[eps] = mean_hist

    return results


def smooth(x, window=500):
    return np.convolve(x, np.ones(window) / window, mode='valid')


def plot_epsilon_comparison(sarsa_results, qlearn_results,
                            epsilons=(0.1, 0.5, 1.0), num_steps=100_000):
    """side-by-side plot: sarsa vs q-learning, one curve per epsilon."""
    colors = {0.1: 'steelblue', 0.5: 'coral', 1.0: 'seagreen'}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.suptitle('average reward vs steps: sarsa vs q-learning', fontsize=13)

    for ax, results, title in zip(
        axes,
        [sarsa_results, qlearn_results],
        ['sarsa (on-policy)', 'q-learning (off-policy)']
    ):
        for eps in epsilons:
            curve = smooth(results[eps])
            steps = np.arange(len(curve))
            ax.plot(steps, curve, color=colors[eps],
                    label=f'ε={eps}', lw=1.8)

        ax.axhline(0, color='gray', lw=0.5, ls='--')
        ax.set_xlabel('steps')
        ax.set_ylabel('running mean reward')
        ax.set_title(title)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURES_DIR, 'td_epsilon_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("saved: td_epsilon_comparison.png")


def plot_sarsa_vs_qlearning(sarsa_results, qlearn_results,
                            eps=0.1, num_steps=100_000):
    """direct comparison of sarsa vs q-learning at a fixed epsilon."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(smooth(sarsa_results[eps]),  color='steelblue', lw=2, label='sarsa')
    ax.plot(smooth(qlearn_results[eps]), color='coral',     lw=2, label='q-learning')
    ax.axhline(0, color='gray', lw=0.5, ls='--')
    ax.set_xlabel('steps')
    ax.set_ylabel('running mean reward')
    ax.set_title(f'sarsa vs q-learning  |  ε={eps}')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURES_DIR, 'td_sarsa_vs_qlearning.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("saved: td_sarsa_vs_qlearning.png")
