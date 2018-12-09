import numpy as np
from itertools import product

class Discretizer():
    # discretizer should not access the environment - leave the agent or buffer do the sampling

    def __init__(self, space, bin_size=None, bin_sizes=None):

        self.space = space
        self.n_vars = space.shape[0]

        # get low, high and total interval for each variable
        self.lower_bounds = space.low
        self.upper_bounds = space.high

        # manually handle infinite environments temporarily
        # only for CartPole, but need a better way to define as boundaries (as done in sampling)
        self.lower_bounds[self.lower_bounds<-10] = -10.
        self.upper_bounds[self.upper_bounds>10] = 10.

        # set up bin sizes with default size if not given
        self.bin_size = bin_size or 10
        self.bin_sizes = bin_sizes or [self.bin_size for var in range(self.n_vars)]

        # calculate state bins
        self.bins = []
        for v in range(self.n_vars):
            low, high = self.lower_bounds[v], self.upper_bounds[v]
            bins = np.histogram_bin_edges([low, high], bins=self.bin_sizes[v])
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
        for v in range(self.n_vars):
            bins = np.histogram_bin_edges(samples[:, v], bins=self.bin_sizes[v])
            self.bins.append(bins[1:-1])

    def calculate_grid_positions(self, action_size):
        """ Calculate all possible positions in the grid """

        ranges = []
        for bs in self.bin_sizes:
            ranges.append(range(bs))
        ranges.append(range(action_size))

        return product(*ranges)

class TileDiscretizer(Discretizer):

    # has an issue with infinity.
    # Can appy directly to values, but will break on 0
    # what other options are there?
    # how do I handle infinity in state discretization?

    def __init__(self, space, bin_size=None, bin_sizes=None, offsets=None):
        super(TileDiscretizer, self).__init__(space, bin_size, bin_sizes)

        self.intervals = self.upper_bounds - self.lower_bounds

        # alternatively, can define a number of tiles hyperparameter for optimization
        # and randomly initialize the offsets
        self.offsets = offsets or [0, -.05, -0.1, +0.05, + 0.1]

        # calculate state bins
        self.bins = []
        for idx_offset, offset in enumerate(self.offsets):
            self.bins.append([])
            for v in range(self.n_vars):
                offset_v = offset * self.intervals[v]
                low, high = self.lower_bounds[v], self.upper_bounds[v]
                bins = np.histogram_bin_edges([low, high], bins=self.bin_sizes[v])
                self.bins[idx_offset].append(bins[1:-1] + offset_v)

    def tiles_count(self):
        return len(self.offsets)

    def convert(self, sample):
        """ Output is no longer in the form (x,y,z) as in regular state aggregation, but in the form of ((x1, y1, z1), (x2, y2, z2), (x3, y3, z3)).

            Every var is represented by the position of several tiles and similar displacements for different dimension are grouped 
        """

        # make it more explicit
        discrete_sample = []
        for offset_bins in self.bins:
            discrete_var = []
            for s, bin in zip(sample, offset_bins):
                discrete_var.append(int(np.digitize(s, bin)))
            discrete_sample.append(tuple(discrete_var))

        return tuple(discrete_sample)

    def define_bins_from_samples(self, samples):
        """ Similar to state aggregation method, but adding the offsets """

        # convert to array if given in samples
        if type(samples) is not np.ndarray:
            samples = np.array(samples)

        # recalculate bins
        self.bins = []
        for idx_offset, offset in enumerate(self.offsets):
            self.bins.append([])
            for v in range(self.n_vars):
                offset_v = offset * self.intervals[v]
                bins = np.histogram_bin_edges(samples[:, v], bins=self.bin_sizes[v])
                self.bins[idx_offset].append(bins[1:-1] + offset_v)


class ActionDiscretizer(Discretizer):

    def __init__(self, space, bin_size=None, bin_sizes=None):
        super(ActionDiscretizer, self).__init__(space, bin_size, bin_sizes)

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
            lower_bounds = self.lower_bounds[idx]
            upper_bounds = self.upper_bounds[idx]

            # zoom in the correct boundaries of the bin, if not on the edge
            if s>0:
                lower_bounds = self.bins[idx][s-1]
            if s<9:
                upper_bounds = self.bins[idx][s]

            # what now? now I have to take these two boundaries and get 
            continuous_sample.append(np.random.uniform(low=lower_bounds, high=upper_bounds))

        return continuous_sample





