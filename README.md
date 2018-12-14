# fasterRL

Library for deep reinforcement learning based on pytorch. Under development.

Future plan includes:

- support to tensorflow 
- support for platforms:
    - PyBullet
    - VizDoom
    - DeepMind Lab
    - Industrial Benchmark
- parallelization in multiagents
- unit and functional tests
- benchmarks for main algorithms and environments

## Installation

Open `install.sh`, and change the `LOG_DIR` path to where you want to save the experiments data. Run install.sh to create the environment variable `log dir` and the folders structure for logging. 
(to add: requirements)

## Use

Requires three steps only:

1. define a dictionary with the hyperparameters of the model.
2. initialize the experiment class
3. run.

Example:

```
params = {
    "LOG_LEVEL": 2,
    "PLATFORM": "openai",
    "ENV_NAME": "FrozenLake-v0",
    "METHOD": "QLearning",
    "NUM_TRIALS": 3,
    "MAX_EPISODES": 1000,
    "LEARNING_RATE": 0.3,
    "GAMMA": 0.99
}

exp = BaseExperiment(params)
exp.run()
```

Sample code for each algorithm can be found in the folder `examples`.

## Functionalities

Currently available algorithms:

- Q-Learning
- Sarsa
- MonteCarlo
- Policy Gradients (PG)
- Cross Entropy (CE)
- Reinforce
- Deep Q-Networks (DQN)
- Double Deep Q-Networks (DDQN)
- Deep Deterministic Policy Gradient (DDPG)
- Actor-Critic
- Advantage Actor-Critic (A2C)

Customization options are available for state-of-the-art methods (not full list):

- Discretization with aggregation (for state and/or action space)
- Discretization with tile coding (for state and/or action space)
- N-steps for off-policy methods
- Importance Sampling
- Gradient Clipping
- Priority Replay
- Multiagents (currently only sequential implementation)
- Experience Sharing
- Planned: eligibility traces, boltzman exploration, optimistic starts

Allows different levels of logging:

- Step details or episode details as events (for tensorboard)
- Experiments results as json 
- Command line outputs 

Supports platforms:

- OpenAI
- Malmo/Marlo, based on minecraft

# Repository Map

## Agents

Each file contains a different class of related algorithms. Whenever possible use of hierarchy is encouraged to avoid code reuse. Modularization, simplicity and self-explainability are preferred over performance.
 
## Common

Common classes shared amongst different agents.

**Environment**: abstraction that handles differences between RL platforms.

**Wrappers**: act as a decorator class to the original environment, modifying its attributes or functions (for example changing the position of the color channel to be used in pytorch). Inspired on OpenAI wrappers.

**Buffers**: experience replay buffers. Used in a model to store experiences, which an agent can retrieve later for training.

**Loggers**: responsible to collect and report data from the experiments. Loggers can save details to events file (tensorboard), save results to json or output progress to command line, depending on the log level defined in the experiment parameters.

<!--**Exploration**: exploration strategies. currently only $\epsilon$-greedy exploration available.
-->
**Networks**: neural network models used for function approximation.

## Others

**Examples**: code examples to kick start your project.

**Experiments**: scripts with detailed experiments for past research projets.

**Notebooks**: jupyter notebook with analysis of past experiments conducted.






