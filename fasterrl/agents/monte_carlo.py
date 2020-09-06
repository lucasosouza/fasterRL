from fasterrl.agents.td_learning import TDLearning
from fasterrl.common.buffer import MCTransitionBuffer

class MonteCarlo(TDLearning):

    def reset(self):
        super(MonteCarlo, self).reset()

        # recreate transition buffer every reset
        self.buffer = MCTransitionBuffer()

    def set_environment(self, env):
        super(MonteCarlo, self).set_environment(env)

        # include a count for each state action pair
        self.qcount = {}
        for state in range(self.env.observation_space.n):
            self.qcount[state] = {}
            for action in range(self.env.action_space.n):
                self.qcount[state][action] = 0

    def learn(self, action, next_state, reward, done):

        self.buffer.append((self.state, action, reward))
        if done:
            if self.importance_sampling:
                self.learn_with_importance_sampling(action, next_state, reward, done)
            else:
                for state, action, value in self.buffer.calculate_value(self.gamma):
                    error = value - self.qtable[state][action]
                    # will need to change how is this update in monte carlo
                    # to account for discretization
                    self.qcount[state][action] += 1
                    self.qtable[state][action] += error/self.qcount[state][action]

    def learn_with_importance_sampling(self, action, next_state, reward, done):
        """ An incremental implementation of Monte-Carlo importance sampling
            Based on pseudocode in Sutton and Barto Book 2nd edition, page 109
        """

        weight = 1
        for state, action, value in self.buffer.calculate_value(self.gamma):
            # exits when importance sampling equals zero
            if weight == 0:
                break
            # otherwise learns
            error = value - self.qtable[state][action]
            self.qcount[state][action] += weight
            self.qtable[state][action] += error * weight/self.qcount[state][action]
            weight *= self.calculate_importance_sampling(state, action)

    def calculate_importance_sampling(self, state, action):
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


class EveryVisitMonteCarlo(MonteCarlo):
    "Just another way to call Monte Carlo"

    pass

class FirstVisitMonteCarlo(MonteCarlo):

    def reset(self):
        super(FirstVisitMonteCarlo, self).reset()
        self.buffer.configure(first_visit=True)
