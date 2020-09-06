import numpy as np
from collections import namedtuple, deque
from functools import reduce

__all__ = [
    "Experience",
    "Transition",
    "ShortExperience",
    "Episode",
    "TransitionBuffer",
    "MCTransitionBuffer",
    "ExperienceBuffer",
    "EpisodeBuffer",
    "PrioReplayBuffer"
]

# need to better organize this log of experience, transition, etc

Experience = namedtuple('Experience',
    field_names=['state', 'action', 'reward', 'done', 'next_state'])

Transition = namedtuple('Transition',
    field_names=['state', 'action', 'reward'])

ShortExperience = namedtuple('ShortExperience',
    field_names=['state', 'action'])

Episode = namedtuple('Episode',
    field_names=['reward', 'experiences'])


# will do separate classes then merge if I see opportunity to merge

class TransitionBuffer:
    """ A transition buffer used for MonteCarlo or ND-steps """

    def __init__(self, n_steps=5, gamma=0.99):

        self.buffer = deque(maxlen=n_steps)

        discount_v = []
        for i in range(n_steps):
            discount_v.append(gamma**i)
        self.discount_v = np.array(discount_v)

        self.n_steps = n_steps

    def all(self):
        return list(self.buffer)

    def full(self):
        if len(self.buffer) == self.n_steps:
            return True
        return False

    def append(self, experience):
        self.buffer.append(experience)

    def flush(self):
        for _ in range(len(self.buffer)):
            yield self.all()
            self.buffer.popleft()

class MCTransitionBuffer:
    """ A transition buffer used for MonteCarlo or ND-steps """

    # challenge: to insert logic on importance sampling here on buffer

    def __init__(self, first_visit = False):
        self.buffer = []
        self.first_visit = first_visit

    def configure(self, first_visit):
        self.first_visit = first_visit

    def append(self, transition):
        self.buffer.append(transition)

    def calculate_value(self, gamma):
        """ Calculate value according to some pre-specified n-step """

        if not self.first_visit:
            value = 0
            for state, action, reward in self.buffer[::-1]:
                value = reward + gamma * value
                yield state, action, value

        else:
            # identify which states are first visit
            visited_states = set()
            first_visits = []
            for state, action, reward in  self.buffer[::-1]:
                if (state, action) not in visited_states:
                    visited_states.add((state, action))
                    first_visits.append(True)
                else:
                    first_visits.append(False)

            # yield only if first visit
            value = 0
            for (state, action, reward), fv in reversed(list(zip(self.buffer, first_visits))):
                value = reward + gamma * value
                if fv:
                    yield state, action, value

class ExperienceBuffer:

    def __init__(self, capacity):
        # initializes a deque
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def receive(self, experiences):
        """ Receive and append a batch of experience. """

        for experience in experiences:
            self.append(experience)

    def append(self, experience):
        self.buffer.append(experience)

    def select_batch(self, batch_size):

        # restrict to number of experiences available
        batch_size = min(batch_size, len(self.buffer))

        # randomly select experiences
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)

        return [self.buffer[idx] for idx in indices]

    def sample(self, batch_size):
        """ Sample from experience batch based on predetermined rules.
        Main 'meat' from the class is in this method """

        batch = self.select_batch(batch_size)

        # break down into one tuple per variable of the experience
        states, actions, rewards, dones, next_states = zip(*batch)

        # convert tuples into np arrays
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
            np.array(dones, dtype=np.uint8), np.array(next_states)

class EpisodeBuffer:

    def __init__(self, capacity, cutoff_percentile):

        self.cutoff_percentile = cutoff_percentile
        self.capacity = capacity
        self.buffer = []
        self.current_experiences = []

        # only tracking for logger
        self.reward_mean = 0
        self.reward_bound = 0

    def __len__(self):
        return len(self.buffer)

    def append_experience(self, experience):
        self.current_experiences.append(experience)

    def append_episode(self, reward):
        self.buffer.append(Episode(reward, self.current_experiences))
        self.current_experiences = []

        # warns agent when the buffer is full
        if len(self.buffer) == self.capacity:
            return True
        return False

    def sample(self):
        """ Select top episodes to run """

        # get mean of rewards
        rewards = list(map(lambda ep:ep.reward, self.buffer))
        self.reward_bound = np.percentile(rewards, self.cutoff_percentile)
        self.reward_mean = np.mean(rewards)

        filtered_experiences = []
        for reward, experiences in filter(lambda ep: ep[0] >= self.reward_bound, self.buffer):
            filtered_experiences.extend(experiences)

        # get list of actions and observations and return
        states, actions = zip(*filtered_experiences)

        # zero buffer to restart
        self.buffer = []

        # still an issue here - why actions are a long tensor, while action scores are a list?
        # need to look into this
        # return as tensors for learning
        return np.array(states), np.array(actions)


class PrioReplayBuffer(ExperienceBuffer):
    """ implementation from From Deep Reinforcement Learning Handson book """

    def __init__(self, capacity, prob_alpha=0.6):

        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.pos = 0
        self.buffer = []
        self.priorities = np.zeros((capacity, ), dtype=np.float32)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        """ Pull the given amount of transitions form the experience source object and store them in buffer """

        # set experience to maximum priority when it enter the buffer
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            # if buffer not full, append new transition
            self.buffer.append(experience)
        else:
            # otherwise overwrite last position
            self.buffer[self.pos] = experience

        # set priorities
        self.priorities[self.pos] = max_prio

        # adjust position - when ends, goes back to zero
        self.pos = (self.pos + 1) % self.capacity

    def select_batch(self, batch_size):
        """ Need to similar calculation as in samples, but do not do any updates or calculate weights are needed.

            Don't need beta since no importance sampling is done in the method
        """

        # restrict to number of experiences available
        batch_size = min(batch_size, len(self.buffer))

        # calculate probabilities
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        probs = prios ** self.prob_alpha
        probs /= probs.sum()

        # with probabilities, sample buffer
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)

        return [self.buffer[idx] for idx in indices]

    def sample(self, batch_size, beta=0.4):
        """ Convert priorities to probabilities using alpha parameters """

        # calculate probabilities
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        probs = prios ** self.prob_alpha
        probs /= probs.sum()

        # with probabilities, sample buffer
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = zip(*[self.buffer[idx] for idx in indices])

        # calculate importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        # also return indices, since they are required to update priorities for sampled items
        return samples, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        """ Update new priorities for the processed batch """

        for idx, prio in zip(batch_indices, batch_priorities):
           self.priorities[idx] = prio



"""
too generic
    # def configure(**kwargs):
    #     for k,v in kwargs.items():
    #         self[k] = v


old version of nd-steps transition did calculation in the buffer
reconsider when doing DQN where it should stay

class TransitionBuffer:

    def __init__(self, n_steps=5, gamma=0.99):

        self.buffer = deque(maxlen=n_steps)

        discount_v = []
        for i in range(n_steps):
            discount_v.append(gamma**i)
        self.discount_v = np.array(discount_v)

        self.n_steps = n_steps

    def __call__(self):
        return self.buffer

    def full(self):
        if len(self.buffer) == self.n_steps:
            return True
        return False

    def append(self, experience):
        self.buffer.append(experience)

    def flush(self):
        for _ in range(len(self.buffer)):
            yield self.calculate_value()
            self.buffer.popleft()

    def calculate_value(self):
        # will always look for beggining and end of buffer
        state = self.buffer[0].state
        action = self.buffer[0].action
        next_state = self.buffer[-1].next_state
        done = self.buffer[-1].done

        # rewards. starts with zero and update according to experiences
        rewards_v = np.zeros(self.n_steps)
        rewards = map(lambda e:e.reward, self.buffer)
        for idx, r in enumerate(rewards):
            rewards_v[idx] = r

        # multiply rewards with discount vector to get value
        value = np.dot(self.discount_v, rewards_v)

        return state, action, value, done, next_state



"""
