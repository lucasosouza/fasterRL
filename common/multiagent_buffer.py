from .buffer import *

class ExperienceBufferGrid(ExperienceBuffer):
    
    def __init__(self, capacity):
    
        # uses a list instead
        self.buffer = []
        self.capacity = capacity

    def set_grid(self, discretizer, action_size):

        # save reference to discretizer
        self.discretizer = discretizer

        # grid will need to be for actions and states, not just states
        # need to ensure these are from the discretization, not the original environment
        grid_size = tuple(discretizer.bin_sizes)+(action_size,)

        # initialize experiences grid
        self.grid_experiences = np.full(grid_size, list)
        for pos in self.discretizer.calculate_grid_positions(action_size):
            self.grid_experiences[pos] = []

        # initialize occupancy grid, for faster access
        self.grid_occupancy = np.zeros(grid_size, dtype=np.int32)

    def get_position(self, experience):
        """ Calculate position in grid for a given experience """

        state = self.discretizer.convert(experience.state)
        action = experience.action
        position = state + (action, )
        return position        

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

        self.buffer.append(experience)
        self.add_to_grid(experience)

        # check if a state needs to be removed
        if len(self.buffer) > self.capacity:
            # remove from buffer
            experience_to_delete = self.buffer.pop(0)
            self.remove_from_grid(experience_to_delete)    

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

