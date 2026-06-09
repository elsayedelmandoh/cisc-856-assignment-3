"""
cisc 856 assignment 3: applying TD learning algorithms (SARSA & Q-learning).
"""
import os
import matplotlib.pyplot as plt
from src.config.config import settings, FIGURES_DIR
from src.utils.environment import Grid, Action, run_experiment
from src.utils.sarsa_agent import SarsaAgent
from src.utils.q_learning_agent import QLearningAgent
from src.utils.rewards import run_epsilon_comparison, plot_epsilon_comparison, plot_sarsa_vs_qlearning
from src.utils.visualizations import plot_action_values, visualise_policy

def main(): 
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("running sarsa (eps=0.1, 100k steps)...")
    grid         = Grid()
    sarsa_agent  = SarsaAgent(
        number_of_states=grid._layout.size,
        actions=Action,
        initial_state=grid.get_obs(),
        step_size=0.1
    )
    mean_r, _ = run_experiment(grid, sarsa_agent, settings.num_steps, settings.epsilon)
    print(f"  sarsa mean reward: {mean_r:.4f}")
  
    q = sarsa_agent.q_values.reshape(grid._layout.shape + (4,))
    fig = plt.figure(figsize=(10, 9))
    plot_action_values(q, epsilon=settings.epsilon)
    plt.suptitle('sarsa: action value functions', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'sarsa_q_values.png'), dpi=150, bbox_inches='tight')
    plt.close()
  
    visualise_policy(grid, sarsa_agent)
    plt.title('sarsa: greedy policy')
    plt.savefig(os.path.join(FIGURES_DIR, 'sarsa_policy.png'), dpi=150, bbox_inches='tight')
    plt.close()
  
    print("running q-learning (eps=0.1, 100k steps)...")
    grid        = Grid()
    ql_agent    = QLearningAgent(
        number_of_states=grid._layout.size,
        actions=Action,
        initial_state=grid.get_obs(),
        step_size=0.1
    )
    mean_r, _ = run_experiment(grid, ql_agent, settings.num_steps, settings.epsilon)
    print(f"  q-learning mean reward: {mean_r:.4f}")
  
    q = ql_agent.q_values.reshape(grid._layout.shape + (4,))
    fig = plt.figure(figsize=(10, 9))
    plot_action_values(q, epsilon=settings.epsilon)
    plt.suptitle('q-learning: action value functions', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'qlearning_q_values.png'), dpi=150, bbox_inches='tight')
    plt.close()
  
    visualise_policy(grid, ql_agent)
    plt.title('q-learning: greedy policy')
    plt.savefig(os.path.join(FIGURES_DIR, 'qlearning_policy.png'), dpi=150, bbox_inches='tight')
    plt.close()
  
    print(f"running epsilon sweep {settings.epsilons} x {settings.n_runs} seeds each...")
    sarsa_results  = run_epsilon_comparison(SarsaAgent,     settings.num_steps, settings.n_runs, settings.epsilons)
    qlearn_results = run_epsilon_comparison(QLearningAgent, settings.num_steps, settings.n_runs, settings.epsilons)

    plot_epsilon_comparison(sarsa_results, qlearn_results, settings.epsilons, settings.num_steps)
    plot_sarsa_vs_qlearning(sarsa_results, qlearn_results, eps=0.1)
    print("done.")

if __name__ == "__main__":
    main()
