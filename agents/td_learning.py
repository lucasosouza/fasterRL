from .base_agent import ValueBasedAgent
from fasterRL.common.buffer import TransitionBuffer, Experience
import numpy as np
from time import sleep

class TDLearning(ValueBasedAgent):

    def __init__(self, params):
        super(TDLearning, self).__init__(params)

        self.td_type = "QLearning"
        if "TD_TYPE" in params:
            self.td_type = params["TD_TYPE"]

        self.importance_sampling = False
        if "IMPORTANCE_SAMPLING" in params:
            self.importance_sampling = params["IMPORTANCE_SAMPLING"]
            self.sampling_per_decision = False
            if "SAMPLING_PER_DECISION" in params:
                self.sampling_per_decision = params["SAMPLING_PER_DECISION"]

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

    def __init__(self, params):
        super(NStepsTDLearning, self).__init__(params)

        self.n_steps = 5
        if "N_STEPS" in params:
            self.n_steps = params["N_STEPS"]

        # pre-calculate discount vector
        discount_v = []
        for i in range(1,self.n_steps):
            discount_v.append(self.gamma**i)
        self.discount_v = np.array(discount_v)

    def set_environment(self, env):
        super(NStepsTDLearning, self).set_environment(env)

        self.buffer = TransitionBuffer(self.n_steps, self.gamma)

    def learn(self, action, next_state, reward, done):

        experience = Experience(self.state, action, reward, done, next_state)
        self.buffer.append(experience)        
 
        # regular step, if not done and buffer full
        if not done and self.buffer.full():
            transitions = self.buffer.all()
            nsteps_experience = self.calculate_value(transitions)
            self.update_qtable(*nsteps_experience)
        # if complete (no matter if buffer full or not)
        elif done:
            # flush all remaining experiences
            for transitions in self.buffer.flush():
                nsteps_experience = self.calculate_value(transitions)
                self.update_qtable(*nsteps_experience)

    def calculate_value(self, transitions):


        # will always look for beggining and end of buffer
        state = transitions[0].state
        action = transitions[0].action
        reward = transitions[0].reward
        next_state = transitions[-1].next_state
        done = transitions[-1].done

        if len(transitions) > 1:

            # calculate return
            rewards_v = np.array(list(map(lambda e:e.reward, transitions[1:])))
            discounted_reward_v = self.discount_v[:len(rewards_v)] * rewards_v
            ret = np.sum(discounted_reward_v)

            # special case when importance sampling is used
            if self.importance_sampling:
                importance_sampling = self.calculate_importance_sampling(transitions)
                if self.sampling_per_decision:
                    """ 
                        Note: implementation of per decision is based on my understanding and derivation of the theory covered in Sutton book 7.4. Need to verify math. Results are not good
                    """
                    p1 = ret * importance_sampling
                    t = transitions[1] # get next transition
                    p2 = self.qtable[t.state][t.action] * (1-importance_sampling)
                    ret = p1 + p2
                else:
                    ret = ret * importance_sampling

            reward += ret

        return state, action, reward, done, next_state  

    def calculate_importance_sampling(self, transitions):

        importance_sampling_v = [] # not for first action
        for t in transitions[1:]:
            imp_samp = self.step_importance_sampling(t.state, t.action)
            importance_sampling_v.append(imp_samp)

        importance_sampling = np.product(importance_sampling_v)

        return importance_sampling  

    def step_importance_sampling(self, state, action):
        """ Calculate importance sampling considering an e-greedy behavior policy
            Accounts for possible randomness in the greedy policy of breaking ties randomly when more than one (state,action) has the same value
        """

        # calculate probability in target policy
        sorted_actions = sorted(self.qtable[state].items(), key=lambda x:-x[1])
        max_value = sorted_actions[0][1] # first of the list, get_value
        best_actions_values = filter(lambda x:x[1]==max_value, sorted_actions)
        best_actions = list(map(lambda x:x[0], best_actions_values))
        if action in best_actions:
            # need to account for cases where ties are randomly broken
            prob_greedy = 1/(len(best_actions))
        else:
            prob_greedy = 0

        # calculate probability in behavior policy
        prob_exploration = self.epsilon/self.num_actions + (1-self.epsilon) * prob_greedy

        return prob_greedy / prob_exploration


class QLearning(TDLearning):
    pass

class NStepsQLearning(NStepsTDLearning):
    pass

class Sarsa(TDLearning):
    def __init__(self, params):
        super(Sarsa, self).__init__(params)
        self.td_type = "SARSA"

class NStepsSarsa(NStepsTDLearning):
    def __init__(self, params):
        super(NStepsSarsa, self).__init__(params)
        self.td_type = "SARSA"


