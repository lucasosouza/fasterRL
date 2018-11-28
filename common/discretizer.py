
# creates a discretizer object to be used to discretize classes

# how to discretize? see code used in the focused sharing sample

# several approaches possible

# the ones that keeps the vector structure used for qtable is using the method of calculating the position described in the summation formula I've drawn

# the other option is having a qtable be a matrix and use the alternate approach

# for now, I will implement the one that plays along with the implemented methods for qlearning, and leave a backup implementation with the second method

# alternative initialization of qtable as matrix
# self.q_table = np.zeros(shape=(self.state_size + (self.action_size,)))
# the advantage of moving over to this state is that I can implement discretizer as a state that can be passed to a matrix

# I don't have a way to test it
# not in openai gym at least
# unless I build it myself
# well, I already have the function define from samples, I think I will use that

import numpy as np

class Discretizer():
    # don't access the environment - leave the agent or buffer do the sampling

    def __init__(self, space, bin_size=None, bin_sizes=None):

        self.space = space
        self.n_vars = space.shape[0]

        # set up bin sizes with default size if not given
        self.bin_size = bin_size or 10
        self.bin_sizes = bin_sizes or [bin_size for var in range(self.n_vars)]

        # calculate state bins
        self.bins = []
        for i in range(self.n_vars):
            low, high = space.low[i], space.high[i]
            bins = np.histogram_bin_edges([low, high], bins=self.bin_sizes[i])
            self.bins.append(bins[1:-1])

    def convert(self, sample):

        discrete_sample = [int(np.digitize(s, b)) for s,b in zip(sample, self.bins)]
        return tuple(discrete_sample)

    def define_bins_from_samples(self, samples):

        # convert to array if given in samples
        if type(samples) is not np.ndarray:
            samples = np.array(samples)

        # recalculate bins
        self.bins = []
        for i in range(self.n_vars):
            bins = np.histogram_bin_edges(samples[:, i], bins=self.bin_sizes[i])
            self.bins.append(bins[1:-1])

class ActionDiscretizer(Discretizer):

    def __init__(self, space, bin_size=None, bin_sizes=None):
        super(ActionDiscretizer, self).__init__(space, bin_size, bin_sizes)

        self.lower_bound = space.low
        self.upper_bound = space.high

        self.dim_sizes = [int(np.prod(self.bin_sizes[i+1:])) for i in range(len(self.bin_sizes))]

    def vector_to_matrix(self, position):
        """ Convert from flattened vector representation to a multi-dimensional matrix representation """

        matrix_pos = []
        for ds in self.dim_sizes:
            matrix_pos.append(position // ds)
            position = position % ds

        return matrix_pos

    def revert(self, sample):
        """ Draw an uniform value in the interval between bins """

        # converto matrix representation
        sample = self.vector_to_matrix(sample)

        # iterate through all vars in action 
        continuous_sample = []
        for idx, s in enumerate(sample):
            # initialize boundaries to be the environment boundaries 
            lower_bound = self.lower_bound[idx]
            upper_bound = self.upper_bound[idx]

            # zoom in the correct boundaries of the bin, if not on the edge
            if s>0:
                lower_bound = self.bins[idx][s-1]
            if s<9:
                upper_bound = self.bins[idx][s]

            # what now? now I have to take these two boundaries and get 
            continuous_sample.append(np.random.uniform(low=lower_bound, high=upper_bound))

        return continuous_sample





