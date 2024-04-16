# Double-DQN-implementation-in-Frozen-Lake-environment

## Double Deep Q-Network (DDQN)

## Overview
Double Deep Q-Networks (DDQN) are an extension of the basic Deep Q-Network (DQN) that aim to reduce overestimations of action values under certain conditions. This overestimation is mitigated by decoupling the selection and evaluation of the action in the Q-learning update. While the original DQN uses the same values both to select and to evaluate an action, DDQN uses the current network to select actions and an independent target network to evaluate the action.

## Architecture
DDQN involves two key components:
1. **Main Network**: This network learns from each interaction and is updated at every step or after a few steps depending on the training configuration.
2. **Target Network**: This network's weights are updated less frequently and are used to generate the Q-value targets for the training of the Main Network.

Both networks are copies of each other at the beginning of training but diverge temporarily as the Main Network is updated more frequently.

## Advantages of DDQN
- **Reduces Overestimations**: By using a separate network to evaluate the action's value, DDQNs tend to give a more stable and reliable estimate which avoids the overoptimistic value estimates.
- **Improves Learning Stability**: The use of a target network helps stabilize the learning process by keeping the target Q-values fixed for a number of training steps.

## Applications
DDQNs are particularly useful in environments where the state-action space is large or when the agent has to deal with high-dimensional sensory inputs like images. Some typical applications include:
- Video game AI, such as learning optimal strategies in complex games.
- Robotics, for tasks that involve decision making in real-time.
- Any task that involves sequential decision making where the environment's dynamics are complex and multi-dimensional.

## Requirements
To run the provided implementation, you will need:
- Python 3.6 or newer
- PyTorch
- OpenAI Gym

## Installation
Install the required libraries using pip:

```bash
pip install torch gym numpy

