# fasterRL

Library for deep reinforcement learning based on pytorch. Under development.

Future plan includes adding tensorflow models as well for completion and comparison. 

# Installation

Run install.sh to create the environment variable log dir and the folders structure for logging. Feel free to change the path to the folder you desire.

To add: requirements

# Repository Map

## Agents

Each file contains a different class of related agents. Whenever possible use of hierarchy is preferred to avoid code reuse. Current files:

* base_agent: Main interface shared amongst all models. Models can inherit from it and overwrite  

Currently available agents:

*Discrete state space and discrete action space*
- Q-Learning
- Sarsa
- MonteCarlo

*Continuous state space and discrete action space*
- Deep Q-Networks (DQN)

*Continuous state space and continuous action space*
- Deep Deterministic Policy Gradient (DDPG)

## Common

Common classes share amongst different agents. Instantiated by the agent instance. 

* Wrappers: environment wrappers. Act as a decorator class to the original environment, adding functionality, such as slight changes to the state to adjust to a model (for example changing the position of the color channel to be used in pytorch)

* Buffers: experience replay buffers. Used in a model to store experiences, that the agent can retrieve later for training.

* Loggers: responsible to collect and report data from the experiments. Loggers can save to file, output to screen or both depending on the logging level defined.

* Exploration: exploration strategies

* Sharing: tools that allow agents in a multiagent setting to share information

* Networks: neural network models used for function approximation.

## Support

Functions and classes not directly related to the model, but which support experimentation. 

* Utils: general smaller functions reused in the code, not belonging to a specific category

* Experiments: tool to generate and run experiments based on parameters given

## Examples

Contains code examples that can be kick start your project 

## Notebooks

Examples of experiments evaluation. To be removed later





