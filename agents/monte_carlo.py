from .td_learning import TDLearning
from fasterRL.common.buffer import MCTransitionBuffer

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

        # only do learning here, once it is done
        if done:     
            for state, action, value in self.buffer.calculate_value(self.gamma):
                error = value - self.qtable[state][action] 
                self.qcount[state][action] += 1
                self.qtable[state][action] += error/self.qcount[state][action]
                # algorithm from: https://www.jeremyjordan.me/rl-learning-implementations/
                # different than book version, which always gets the average

class EveryVisitMonteCarlo(MonteCarlo):
    "Just another way to call Monte Carlo"

    pass

class FirstVisitMonteCarlo(MonteCarlo):

    def reset(self):
        super(FirstVisitMonteCarlo, self).reset()
        self.buffer.configure(first_visit=True)
