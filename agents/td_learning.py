from .base_agent import ValueBasedAgent
from fasterRL.common.buffer import TransitionBuffer, Experience

class TDLearning(ValueBasedAgent):

    def __init__(self, params, alias="agent"):
        super(TDLearning, self).__init__(params, alias)

        self.td_type = "QLearning"
        if "TD_TYPE" in params:
            self.td_type = params["TD_TYPE"]

    def set_environment(self, env):
        super(TDLearning, self).set_environment(env)

        self.num_actions = self.env.action_space.n

        # include qtable initialization
        self.qtable = {}
        for state in range(self.env.observation_space.n):
            self.qtable[state] = {}
            for action in range(self.env.action_space.n):
                self.qtable[state][action] = 0    

    def select_next_action(self, next_state):

        if self.td_type == "QLearning":
            return self.select_best_action(next_state)
        elif self.td_type == "SARSA":
            return self.select_action()

    def learn(self, action, next_state, reward, done):

        self.update_qtable(self.state, action, reward, done, next_state)

    def update_qtable(self, state, action, reward, done, next_state):

        # calculate td_target
        if not done:        
            next_action = self.select_next_action(next_state)
            td_target = reward + self.gamma * self.qtable[next_state][next_action]
        else:
            td_target = reward            
            
        # update q-table
        td_error = td_target - self.qtable[state][action]
        self.qtable[state][action] += self.learning_rate * td_error


class NStepsTDLearning(TDLearning):

    def __init__(self, params, alias="agent"):
        super(NStepsTDLearning, self).__init__(params, alias)

        self.n_steps = 5
        if "N_STEPS" in params:
            self.n_steps = params["N_STEPS"]

    def set_environment(self, env):
        super(NStepsTDLearning, self).set_environment(env)

        self.buffer = TransitionBuffer(self.n_steps, self.gamma)

    def learn(self, action, next_state, reward, done):

        experience = Experience(self.state, action, reward, done, next_state)
        self.buffer.append(experience)        
 
        # regular step, if not done and buffer full
        if not done and self.buffer.full():
            buffer_experience = self.buffer.calculate_value()
            self.update_qtable(*buffer_experience)
        # if complete (no matter if buffer full or not)
        elif done:
            # flush all remaining experiences
            for buffer_experience in self.buffer.flush():
                self.update_qtable(*buffer_experience)                

class QLearning(TDLearning):
    pass

class NStepsQLearning(NStepsTDLearning):
    pass

class Sarsa(TDLearning):
    def __init__(self, params, alias="agent"):
        super(Sarsa, self).__init__(params, alias)
        self.td_type = "SARSA"

class NStepsSarsa(NStepsTDLearning):
    def __init__(self, params, alias="agent"):
        super(NStepsSarsa, self).__init__(params, alias)
        self.td_type = "SARSA"


