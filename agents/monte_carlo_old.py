from .base_agent import ValueBasedAgent

class MonteCarlo(ValueBasedAgent):

    def set_environment(self, env):
        super(MonteCarlo, self).set_environment(env)

        # include a count for each state action pair
        self.qcount = {}
        for state in range(self.env.observation_space.n):
            self.qcount[state] = {}
            for action in range(self.env.action_space.n):
                self.qcount[state][action] = 0

    def reset(self):
        super(MonteCarlo, self).reset()

        self.transitions = []

    def learn(self, action, next_state, reward, done):
  
        self.transitions.append((self.state, action, reward))

class EveryVisitMonteCarlo(MonteCarlo):

    def learn(self, action, next_state, reward, done):
        super(EveryVisitMonteCarlo, self).learn(action, next_state, reward, done)

        # only do learning here, once it is done
        if done:        
            value = 0
            for state, action, reward in self.transitions[::-1]:
                value = reward + self.gamma * value
                error = value - self.qtable[state][action] 
                self.qcount[state][action] += 1
                self.qtable[state][action] += error/self.qcount[state][action]
                # algorithm from: https://www.jeremyjordan.me/rl-learning-implementations/
                # different than book version, which always gets the average

class FirstVisitMonteCarlo(MonteCarlo):

    def learn(self, action, next_state, reward, done):
        super(FirstVisitMonteCarlo, self).learn(action, next_state, reward, done)

        # only do learning here, once it is done
        if done:
            # verify which transitions are first visits
            # optimized for one pass instead of several passes to check in set
            visited_states = set()
            first_visits = []
            for state, action, reward in  self.transitions[::-1]:
                if (state, action) not in visited_states:
                    visited_states.add((state, action))
                    first_visits.append(True)
                else:
                    first_visits.append(False)

            # update
            value = 0
            for (state, action, reward), fv in reversed(list(zip(self.transitions, first_visits))):
                value = reward + self.gamma * value
                if fv:
                    error = value - self.qtable[state][action] 
                    self.qcount[state][action] += 1
                    self.qtable[state][action] += error/self.qcount[state][action]

