from .base_agent import ValueBasedAgent

class DiscreteTDLearning(ValueBasedAgent):

    def set_environment(self, env):
        super(DiscreteTDLearning, self).set_environment(env)

        self.num_actions = self.env.action_space.n

        # include qtable initialization
        self.qtable = {}
        for state in range(self.env.observation_space.n):
            self.qtable[state] = {}
            for action in range(self.env.action_space.n):
                self.qtable[state][action] = 0    

class QLearning(DiscreteTDLearning):

    def learn(self, action, next_state, reward, done):

        # calculate td_target
        if not done:        
            next_action = self.select_best_action(next_state)
            td_target = reward + self.gamma * self.qtable[next_state][next_action]
        else:
            td_target = reward            
            
        # update q-table
        td_error = td_target - self.qtable[self.state][action]
        self.qtable[self.state][action] += self.learning_rate * td_error

class Sarsa(DiscreteTDLearning):

    def learn(self, action, next_state, reward, done):

        # calculate td_target
        if not done:        
            next_action = self.select_action() # only change to SARSA is here
            td_target = reward + self.gamma * self.qtable[next_state][next_action]
        else:
            td_target = reward            
            
        # update q-table
        td_error = td_target - self.qtable[self.state][action]
        self.qtable[self.state][action] += self.learning_rate * td_error




