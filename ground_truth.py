#!/usr/bin/env python

import numpy as np
from sklearn import metrics


PREDICTION_METRICS=['accuracy', 'precision', 'recall']

class GroundTruth(object):
    def generate_grid(self):
        self.grid = np.empty(self.settings.DIM)
        for loc in self.settings.LOCATIONS:
            self.grid[loc] = self.classify(loc)

        print self.grid.T

    # return True or False
    def classify(self, loc):
        raise NotImplementedError

    # return True (1) or False (0)
    def at(self, loc):
        return self.grid[loc]

    # Compute 0-1 loss between prediction and ground truth, averaged over all grid points.
    # This is the risk if all grid points are weighted equally.
    # Also equals 1 - accuracy, where accuracy = (TP + TN) / n_grid_pts
    def prediction_error(self, prediction):
        if prediction is None:
            return 1.0

        cm = self.confusion_matrix(prediction)
        n_grid_pts = np.product(self.settings.DIM)
        return float(cm['fp']+cm['fn']) / n_grid_pts

    # Compute accuracy, precision and recall
    def prediction_metrics(self, prediction):
        if prediction is None:
            return dict(accuracy=0., precision=0., recall=0.)

        cm = self.confusion_matrix(prediction)
        return dict(
            accuracy=float(cm['tp']+cm['tn']) / (cm['tp']+cm['tn']+cm['fp']+cm['fn']),
            precision=0 if cm['tp']+cm['fp'] == 0 else float(cm['tp']) / (cm['tp']+cm['fp']),
            recall=0 if cm['tp']+cm['fn'] == 0 else float(cm['tp']) / (cm['tp']+cm['fn'])
        )

    def confusion_matrix(self, prediction):
        mat = metrics.confusion_matrix(self.grid.flatten(), prediction.flatten(),
            labels=[1, 0])
        return dict(tp=mat[0,0], fn=mat[0,1], fp=mat[1,0], tn=mat[1,1])


# Boundary is given by <w, x> = b
class LinearGroundTruth(GroundTruth):
    def __init__(self, settings):
        self.settings = settings
        self.set_boundary()
        self.generate_grid()
        self.name = settings.dim_string() + "-linear"

    def classify(self, loc):
        return np.dot(self.w, np.array(loc)) - self.b < 0

    def set_boundary(self):
        raise NotImplementedError

    def __str__(self):
        return "linear boundary (w=%s, b=%.2f)" % (str(self.w), self.b)


class SimpleLinearGroundTruth(LinearGroundTruth):
    def set_boundary(self):
        # Linear boundary depends on single feature (widest dimension)
        active_dim = max(enumerate(self.settings.DIM), key=lambda x: x[1])[0]
        self.w = np.zeros(len(self.settings.DIM))
        self.w[active_dim] = 1
        self.b = self.settings.DIM[active_dim] / 2


class GeneralLinearGroundTruth(LinearGroundTruth):
    def set_boundary(self):
        # All feature weights are sampled uniformly from [-1, 1]
        self.w = np.round(np.random.uniform(-1, 1, len(self.settings.DIM)), 2)
        # line should pass through center of grid
        origin = np.array([b/2 for b in self.settings.DIM])
        self.b = round(np.dot(self.w, origin), 2)


# Boundary is given by <w, \phi(x)> = 0 where
# \phi(x)^T = (1   x_1 x_1^2 ... x_1^m x_2)
#       w^T = (w_0 w_1 w_2   ... w_m   1)
class SimplePolynomialGroundTruth(GroundTruth):
    def __init__(self, degree, settings):
        self.degree = degree
        self.settings = settings
        self.set_boundary()
        self.generate_grid()
        self.name = "%s-poly-%d" % (settings.dim_string(), degree)

    def classify(self, loc):
        return np.dot(self.w, self.phi(loc)) < 0

    def phi(self, loc):
        x_1, x_2 = loc[0:2]
        return np.concatenate([[1.0], [x_1 ** i for i in range(1, self.degree+1)], [x_2]])

    def set_boundary(self):
        # A polynomial of degree m is determined by m+1 points.
        # Sample m+1 coordinates in the x_1 dimension without replacement,
        # then sample a coordinate in the x_2 dimension for each independently.
        # Compute the w vector that passes through these points.
        x_1_coords = np.random.choice(range(self.settings.DIM[0]), size=self.degree+1, replace=False)
        x_2_coords = np.random.choice(range(self.settings.DIM[1]), size=self.degree+1, replace=True)
        self.points = zip(x_1_coords, x_2_coords)  # really first 2 dimensions of points
        self.points = sorted(self.points, key=lambda pt: pt[0])

        # solve \Phi w = 0 where
        # \Phi = [\phi(point 1)^T \\ \vdots \\ \phi(point m+1)^T]
        # Also enforce last coordinate of w is 1
        phi_matrix = np.vstack([self.phi(point) for point in self.points])    # (m+1) x (m+2)
        last_row = np.append(np.zeros(self.degree+1), [1.0])
        A = np.vstack([phi_matrix, last_row])    # (m+2) x (m+2)
        b = np.append(np.zeros(self.degree+1), [1.0])
        self.w = np.linalg.solve(A, b)

        assert self.w[-1] == 1.0

    def __str__(self):
        return "polynomial boundary of degree %d\npassing through %s\n(w=%s)" % \
            (self.degree, ', '.join(["(%d,%d)" % (x, y) for x, y in self.points]), str(np.round(self.w, 2)))


# Boundary is given by x_2 >= f(x_1). In 2D, y >= f(x)
# Translate coordinates so that center of grid is origin
class SimpleFunctionGroundTruth(GroundTruth):
    def __init__(self, settings, fn):
        self.settings = settings
        self.fn = fn
        self.origin = np.array([b/2 for b in self.settings.DIM])
        self.generate_grid()
        self.name = "%s-%s" % (settings.dim_string(), fn.name)

    def classify(self, loc):
        x_1, x_2 = np.array(loc[0:2]) - self.origin
        return x_2 >= self.fn.f(x_1)

    def __str__(self):
        return "boundary %s" % self.fn.formula

