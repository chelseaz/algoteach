#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

class GroundTruth(object):
    def __init__(self, settings):
        self.settings = settings

        self.grid = np.empty(settings.DIM)
        for loc in settings.LOCATIONS:
            self.grid[loc] = self.classify(loc)

    def classify(self, loc):
        # left half is 0, right half is 1
        w = np.array([0, 1, 0])
        b = 2
        return np.dot(w, np.array(loc)) - b < 0

    def at(self, loc):
        return self.grid[loc]

    def plot(self):
        plt.figure()

        plt.axis('off')
        plt.title("Ground truth")
        plt.imshow(self.grid, cmap=self.settings.GRID_CMAP, interpolation='none', origin='upper')

        fig = plt.gcf()
        fig.set_size_inches(6, 4)
        fig.savefig('ground-truth.png', dpi=100)

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

def error_to_accuracy(error):
    return 0 if error is None else 1 - error
