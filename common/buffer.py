import numpy as np
from collections import namedtuple, deque
from functools import reduce
from skimage.measure import block_reduce

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

        # calculate weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        # also return indices, since they are required to update priorities for sampled items
        return samples, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        """ Update new priorities for the processed batch """

        for idx, prio in zip(batch_indices, batch_priorities):
           self.priorities[idx] = prio            



# also need to implement these with prioreplay
# seems to be a good time to start experimenting with decorators

class ExperienceBufferGrid(ExperienceBuffer):
    
    def __init__(self, capacity, n_bins=10):
    
        # uses a list instead
        self.buffer = []
        self.capacity = capacity
        self.n_bins = n_bins

    def set_grid(self, observation_space, action_space):

        # need access to the env to do this
        # only action space and observation space are necessary however

        n_states = observation_space.shape[0]
        n_actions = action_space.n

        # grid shape: discretize all state variables, then add action variable
        grid_shape = [self.n_bins for _ in range(n_states)]
        grid_shape.append(n_actions)

        # store experiences by grid position, to facilitate recovery
        # best way I could find to init an array with empty lists
        m = np.zeros(np.product(grid_shape), dtype=list)
        for v in range(len(m)):
            m[v] = []
        self.grid_experiences = m.reshape(grid_shape)
        # grid occupancy is just a matrix with integers to count experiences
        self.grid_occupancy = np.zeros(grid_shape, dtype=np.int32)

        # calculate state bins
        self.state_bins = []
        for i in range(n_states):
            low, high = observation_space.low[i], observation_space.high[i]

            # hacky fix - cap low an high if infinite
            # limit to +1000 and -1000
            low = max(low, -1e3)
            high = min(high, 1e3)

            bins = np.histogram([low, high], bins=self.n_bins)[1]
            self.state_bins.append(bins[1:])

    def identify_unexplored(self, threshold):

        mask = self.grid_occupancy <= threshold
        return mask

    def remove_from_grid(self, removed_experience):

        position_old = self.get_position(removed_experience)
        # remove from grid - will always be the first to be added
        self.grid_experiences[position_old].pop(0)
        # remove from count
        self.grid_occupancy[position_old] -= 1

    def add_to_grid(self, experience):

        position_new = self.get_position(experience)
        # store in grid
        self.grid_experiences[position_new].append(experience)
        # add to counter
        self.grid_occupancy[position_new] += 1


    def append(self, experience):
        """ Adds to grid as well as appending to buffer. Remove if buffer full """ 

        # append to buffer
        self.buffer.append(experience)
        self.add_to_grid(experience)

        # check if a state needs to be removed
        if len(self.buffer) > self.capacity:
            # remove from buffer
            experience_to_delete = self.buffer.pop(0)
            self.remove_from_grid(experience_to_delete)

    def get_position(self, experience):
        """ Calculate position in grid for a given experience """

        position = []
        state = experience.state
        action = experience.action
        for idx in range(len(state)):
            place = min(self.n_bins-1, int(np.digitize(state[idx], self.state_bins[idx], right=True)))
            position.append(place)
        position.append(action)

        return tuple(position)            

    def select_batch_with_mask(self, batch_size, mask):
        """ Sample from experience batch based on predetermined rules.
        Main 'meat' from the class is in this method """
        
        # filter only relevant experiences
        selected_experiences = reduce(lambda x,y: x+y, list(self.grid_experiences[mask]))

        # pick random experiences in buffer, with no replacement
        selected_batch_size = min(batch_size, len(selected_experiences))

        # only proceed if batch size is greater than 0
        if selected_batch_size > 0:
            # select indices
            indices = np.random.choice(len(selected_experiences), selected_batch_size, replace=False)
            return [selected_experiences[idx] for idx in indices]

        # else return an empty array to standardize output
        else:
            return []


class ExperienceBufferGridImage(ExperienceBufferGrid):
    """ 
        Adaptations to ExperienceBufferGrid to handle images
        Changes how grid is set and how to get position of the experience
    """
    
    def set_grid(self, observation_space, action_space):
        """ Customized to handle 4 actions only. Need to modify to handle n actions """

        # rewrite n bins
        self.n_bins = 4

        n_states = observation_space.shape[0]
        n_actions = action_space.n

        # define bins for digitize (discretization operation)
        self.bins = [0.25, 0.5, 0.75]
        # self.reduce_block = (4,28,28) # reduces to 3x3x1
        self.reduce_block = (1,28,28) # reduces to 3x3x1 # modified to single channel
        self.reduced_state_size = 9
        self.reduced_state_action_size = self.reduced_state_size + 1
        self.exponentials = []
        for exponential in list(range(self.reduced_state_action_size))[::-1]:
            self.exponentials.append(self.n_bins**exponential) 
        self.grid_size = self.n_bins ** self.reduced_state_action_size

        # initialize new structures
        self.grid_occupancy = np.zeros(self.grid_size)
        self.grid_experiences = np.zeros(self.grid_size, dtype=list)
        for ii in range(len(self.grid_experiences)):
            self.grid_experiences[ii] = []
            
    def get_position(self, experience):

        state = experience.state
        action = experience.action
        
        reduced_state = block_reduce(state, self.reduce_block, func=np.mean).ravel()
        bin_placements = list(np.digitize(reduced_state , self.bins))

        # considering action as just part of the state, since num actions = num bins it works
        position = 0
        bin_placements.append(action)
        for bin_placement, exponential in zip(bin_placements, self.exponentials):
            position += bin_placement * exponential

        return position



class PrioExperienceBufferGrid(PrioReplayBuffer, ExperienceBufferGrid):


    def __init__(self, capacity, prob_alpha=0.6, n_bins=10):

        PrioReplayBuffer.__init__(self, capacity, prob_alpha)

        # adds only extra variable for experience grid
        self.n_bins = n_bins

    def append(self, experience):

        # set experience to maximum priority when it enter the buffer
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            # if buffer not full, append new transition
            self.buffer.append(experience)
        else:
            # otherwise overwrite last position
            experience_to_delete = self.buffer[self.pos]
            # rewrites the new
            self.buffer[self.pos] = experience     
            # calculate position
            self.remove_from_grid(experience_to_delete)

        self.add_to_grid(experience)

        # set priorities
        self.priorities[self.pos] = max_prio
        # adjust position - when ends, goes back to zero
        self.pos = (self.pos + 1) % self.capacity


class PrioExperienceBufferGridImage(PrioExperienceBufferGrid, ExperienceBufferGridImage):

    """ Same as PrioExperienceBufferGrid, but with added images """ 

    pass







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
