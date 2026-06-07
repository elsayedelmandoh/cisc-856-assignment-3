# Assignment 3: Implementing and Analyzing TD Algorithms    
Due: Tuesday June 2, before midnight
 
## Objective: 
To gain hands-on experience in implementing SARSA and Q-learning algorithms in gridworld.

## Tasks: 
- Implement SARSA algorithm: (10 points)
- Implement Q-learning algorithm: (10 points)
- Analyze SARSA and Q-learning algorithms. (40 points)
    1. Plot the average reward as a function of the number of steps (both algorithms)
    2. Generate the above plot for different values of epsilon (epsilon = 0.1, 0.5, 1) (both algorithms).
    3. How different values of epsilon affect the training in both algorithms? (both algorithms)
    4. What is the optimal value for epsilon? (both algorithms)

## Submission:
- Python Code: Fully documented code implementing the algorithms. (The points divided in the tasks total of 60 points)
- Report: A comprehensive report detailing the algorithm's results, and an analysis describing task III. (40 points)

## The Problem Description:
- The code snippet is provided, and you can check the details of the grid world implementation and the environment dynamics there.
- States: the different position on the grid. S indicates the start state and G indicates the final state.
- Actions: up (0), right (1), down (2), and left (3).
- Rewards: -5 for bumping into a wall, +10 for reaching the goal, and 0 otherwise.
- Discount factor: 𝛾=0.9
- Note: You are given a code snippet. Feel free to use it or use your own implementation.
 