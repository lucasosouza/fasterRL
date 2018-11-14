import numpy as np
from collections import namedtuple, deque

Experience = namedtuple('Experience', 
    field_names=['state', 'action', 'reward', 'done', 'new_state'])

ShortExperience = namedtuple('ShortExperience', 
    field_names=['state', 'action'])

Episode = namedtuple('Episode',
    field_names=['reward', 'experiences'])

class ExperienceBuffer:
    
    def __init__(self, capacity):
        # initializes a deque
        self.buffer = deque(maxlen=capacity)
        self.experiences_received = 0
        
    def __len__(self):
        return len(self.buffer)
    
    def receive(self, experiences):
        """ Included for regular experience sharing """

        self.experiences_received += len(experiences)
        self.extend(experiences)

    def extend(self, experiences):
        """ Included for regular experience sharing """

        for experience in experiences:
            self.append(experience)

    def append(self, experience):
        self.buffer.append(experience)
        
    def sample(self, batch_size):
        """ Sample from experience batch based on predetermined rules.
        Main 'meat' from the class is in this method """
        
        # pick random experiences in buffer, with no replacement
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        # break down into one tuple per variable of the experience
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])

        # convert tuples into np arrays
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
            np.array(dones, dtype=np.uint8), np.array(next_states)

    def sample_no_mask(self, batch_size):
        """ Method for simple experience sharing """

        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[idx] for idx in indices]

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



