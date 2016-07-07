#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

def error_to_accuracy(error):
    return 0 if error is None else 1 - error

class GroundTruth(object):
    def __init__(self, settings):
        self.settings = settings
        self.generate_grid()

    def generate_grid(self):
        self.grid = np.empty(self.settings.DIM)
        for loc in self.settings.LOCATIONS:
            self.grid[loc] = self.classify(loc)

        print self.grid

    def classify(self, loc):
        raise NotImplementedError

    def at(self, loc):
        return self.grid[loc]

    def plot(self, filename):
        plt.figure()

        plt.axis('off')
        plt.title("Ground truth with boundary\n%s" % str(self))
        plt.imshow(self.grid, cmap=self.settings.GRID_CMAP, interpolation='none', origin='upper')

        fig = plt.gcf()
        fig.set_size_inches(6, 4)
        fig.savefig('%s.png' % filename, dpi=100)

        plt.close()

    # Compute 0-1 loss between prediction and ground truth, averaged over all grid points.
    # This is the risk if all grid points are weighted equally.
    # Also equals 1 - accuracy, where accuracy = (TP + TN) / n_grid_pts
    def prediction_error(self, prediction):
        if prediction is None:
            return 1.0

        d = self.settings.DIM
        n_grid_pts = d[0] * d[1]
        return np.sum(abs(prediction - self.grid)) / n_grid_pts


# Boundary is given by <w, x> = b
class LinearGroundTruth(GroundTruth):
    def __init__(self, settings):
        self.settings = settings
        self.set_boundary()
        self.generate_grid()

    def classify(self, loc):
        return np.dot(self.w, np.array(loc)) - self.b < 0

    def set_boundary(self):
        raise NotImplementedError

    def __str__(self):
        return "w=%s, b=%d" % (str(self.w), self.b)


class SimpleLinearGroundTruth(LinearGroundTruth):
    def set_boundary(self):
        # Linear boundary depends on single feature (widest dimension)
        active_dim = max(enumerate(self.settings.DIM), key=lambda x: x[1])[0]
        self.w = np.zeros(len(self.settings.DIM))
        self.w[active_dim] = 1
        self.b = self.settings.DIM[active_dim] / 2


class GeneralLinearGroundTruth(LinearGroundTruth):
    def set_boundary(self):
        # All features weights are sampled uniformly from [-1, 1]
        self.w = np.random.uniform(-1, 1, len(self.settings.DIM))
        # line should pass through center of grid
        origin = np.array([b/2 for b in self.settings.DIM])
        self.b = np.dot(self.w, origin)
