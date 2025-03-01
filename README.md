# CartPole Reinforcement Learning with DQN & Genetic Algorithm

## Overview

This project implements **Deep Q-Learning (DQN) and Genetic Algorithms** to train an AI agent to play **CartPole-v1** in OpenAI Gym. The project explores both **neural network-based reinforcement learning (DQN)** and **evolutionary strategies (GA)** for optimizing policy learning.

## Features

- **Deep Q-Learning (DQN) Agent**

  - Uses a neural network to approximate Q-values.
  - Implements experience replay for better sample efficiency.
  - Uses an epsilon-greedy policy for exploration.

- **Genetic Algorithm for Optimization**

  - Selects top-performing agents based on fitness scores.
  - Uses crossover to create offspring from parent networks.
  - Applies mutation to introduce diversity in learning.

- **Evaluation and Rendering**

  - `evaluate_agent()`: Runs a trained agent and computes the average reward.
  - `render_agent_performance()`: Visualizes the trained agent playing CartPole in real-time.

## Installation

To set up and run this project, follow these steps:

### **1. Clone the Repository**

```bash
git clone https://github.com/yourusername/cartpole-rl.git
cd cartpole-rl
```

### **2. Install Dependencies**

```bash
pip install numpy tensorflow gym matplotlib
```

### **3. Run the Training Script**

```bash
python cartpole_dqn_genetic.py
```

## Usage

- The script automatically trains an agent using DQN and/or Genetic Algorithm.
- After training, the best agent is evaluated and can be rendered for visualization.
- Modify hyperparameters in the script to experiment with different configurations.

## Example Output

```
Episode 100: Reward: 180
Episode 500: Reward: 200 (Solved!)
Trained agent evaluation: Avg Reward: 195.6
```

## File Renaming Guide

| **Old Name**                          | **New Name (Suggested)** | **Description**                    |
| ------------------------------------- | ------------------------ | ---------------------------------- |
| `Cart_Pole_Reinforcement_Learning.py` | ``                       | Main script for training the agent |

## License

This project is licensed under the **MIT License**.

## Contributions

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch (`feature-new-feature`).
3. Commit and push your changes.
4. Open a pull request.

## Contact

For any questions or support, please open an issue on GitHub.

