from numpy import random

class BaseAgent():

    def __init__(self, params, alias="agent"):

        # which functions and variables need to be initialized in the base agent?
        self.alias = alias
        self.params = params

        # configuration parameters
        if "PLATFORM" in params:
            self.platform = params["PLATFORM"]

        self.render = False
        if "RENDER" in params:
            self.render = params["RENDER"]

        self.random_seed = 42
        if "RANDOM_SEED" in params:
            self.random_seed = params["RANDOM_SEED"]

        self.device = "cpu"
        if "DEVICE" in params:
            self.device = params["device"]

        # common variables shared accross most agents
        self.gamma = 0.99
        if "GAMMA" in params:
            self.gamma = params["GAMMA"]

        self.learning_rate = 0.1
        if "LEARNING_RATE" in params:
            self.learning_rate = params["LEARNING_RATE"]

        # vars to be used in logger
        self.step_reward = 0

    def set_environment(self, env):
        self.env = env
        self.reset()

    def reset(self):
        self.state = self.env.reset()

    def select_action(self):

        return self.env.action_space.sample()

    def play_step(self):

        # take a step
        action = self.select_action()
        next_state, reward, done, _ = self.env.step(action)
        self.render_environment()

        # learn
        self.learn(action, next_state, reward, done)
        
        # prepare for next
        self.state = next_state

        # bookkeeping
        self.step_reward = reward

        self.update_params()

        return done

    def update_params(self):
        pass

    def render_environment(self):

        if self.platform == 'malmo':
            if self.render:
                self.env.render('human') # specific for minecraft

    def learn(self, action, next_state, reward, done):
        pass # no learning in random action agents

class ValueBasedAgent(BaseAgent):

    def __init__(self, params, alias="agent"):
        super(ValueBasedAgent, self).__init__(params, alias)

        # unpack specific variables for algorithm
        self.epsilon = 1.0
        self.epsilon_final = 0.02
        self.epsilon_decay_last_frame = 1000
        if "EPSILON_START" in params:
            self.epsilon_start = params["EPSILON_START"]
        if "EPSILON_FINAL" in params:
            self.epsilon_final = params["EPSILON_FINAL"]            
        if "EPSILON_DECAY_LAST_FRAME" in params:
            self.epsilon_decay_last_frame = params["EPSILON_DECAY_LAST_FRAME"]

        self.epsilon_decay = (self.epsilon - self.epsilon_final) / self.epsilon_decay_last_frame

    def set_environment(self, env):
        super(ValueBasedAgent, self).set_environment(env)

        self.num_actions = self.env.action_space.n

    def select_action(self):

        if random.rand() < self.epsilon:
            return random.randint(self.num_actions)        
        else:
            return self.select_best_action(self.state)

    def select_best_action(self, state):

        # select all possible actions
        possible_actions = list(self.qtable[state].items())

        # shuffle before sorting, to ensure randomness in case of tie
        random.shuffle(possible_actions)
        # sort and get first - can also use argmax
        action = sorted(possible_actions, key=lambda x:-x[1])[0][0]

        return action

    def update_params(self):

        self.epsilon = max(self.epsilon_final, self.epsilon - self.epsilon_decay)



