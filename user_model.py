#!/usr/bin/env python
import math
import numpy as np
from sklearn import svm


class PredictionResult(object):
    def __init__(self, prediction, evaluation=None):
        self.prediction = prediction
        self.evaluation = evaluation


class UserModel(object):
    def __init__(self, settings):
        self.name = "base"
        self.settings = settings
        self.prior = None

    # predict label of all locations
    def predict_grid(self, examples):
        threshold = lambda p: 1 if p >= 0.5 else 0
        vthreshold = np.vectorize(threshold)
        evaluation = self.evaluate_grid(examples)
        if evaluation is None:
            # we don't have examples for both labels yet
            return PredictionResult(prediction=None)
        else:
            return PredictionResult(prediction=vthreshold(evaluation), evaluation=evaluation)

    # get probability of label 1 for all locations
    def evaluate_grid(self, examples):
        raise NotImplementedError


class SVMUserModel(UserModel):
    def predict_grid(self, examples):
        # fit SVM to all examples
        model = self.get_model()
        X = [loc for (loc, _) in examples]
        y = [label for (_, label) in examples]
        if len(set(y)) < 2:
            # we don't have examples for both labels yet
            return PredictionResult(prediction=None)

        model.fit(X, y)
        prediction_list = model.predict(self.settings.LOCATIONS)

        prediction_array = np.empty(self.settings.DIM)
        for loc, pred in zip(self.settings.LOCATIONS, prediction_list):
            prediction_array[loc] = pred

        return PredictionResult(prediction=prediction_array)

    def get_model(self):
        raise NotImplementedError


class LinearSVMUserModel(SVMUserModel):
    def __init__(self, settings):
        super(self.__class__, self).__init__(settings)
        self.name = "linear SVM"

    def get_model(self):
        return svm.SVC(kernel='linear')


class RBFSVMUserModel(SVMUserModel):
    def __init__(self, settings, C, gamma):
        super(self.__class__, self).__init__(settings)
        self.C = C
        self.gamma = gamma
        self.name = "RBF SVM"

    def get_model(self):
        return svm.SVC(kernel='rbf', C=self.C, gamma=self.gamma)


# Performs function approximation using an online kernel machine
# See Dragan and Srinivasa (RoMan 2012)
# TODO: make stateful, so that implementation is truly online and previous computations are saved.
# Currently the user model interface is immutable.
class RBFOKMUserModel(UserModel):
    def __init__(self, settings, prior, eta, lambda_param, w):
        super(self.__class__, self).__init__(settings)
        assert prior.shape == settings.DIM
        self.prior = prior
        self.eta = eta  # new example weight
        self.lambda_param = lambda_param  # additional forgetting rate
        self.w = w  # determines kernel width
        self.kernel_cache = dict()
        self.name = "RBF OKM"

    def evaluate_grid(self, examples):
        offset_coefs = np.empty(len(examples))  # offset coefficients corresponding to examples
        for i, example in enumerate(examples):
            new_loc, new_label = example
            offset_coefs *= 1 - self.eta * self.lambda_param
            new_alpha = new_label - self.evaluate(new_loc, offset_coefs[:i], examples[:i])
            offset_coefs[i] = self.eta * new_alpha

        eval_array = np.empty(self.settings.DIM)
        for loc in self.settings.LOCATIONS:
            eval_array[loc] = self.evaluate(loc, offset_coefs, examples)

        return eval_array

    # Return function approximation at loc X, P(Y=1|X)
    def evaluate(self, loc, offset_coefs, examples):
        offsets = offset_coefs * np.array([self.kernel(old_loc, loc) for old_loc, _ in examples])
        return self.prior[loc] + offsets.sum()

    def kernel(self, loc1, loc2):
        key = (loc1, loc2) if loc1 <= loc2 else (loc2, loc1)
        if key in self.kernel_cache:
            return self.kernel_cache[key]
        else:
            val = np.exp(-0.5 * self.w * ((np.array(loc1) - np.array(loc2)) ** 2).sum())
            self.kernel_cache[key] = val
            return val


class GenerativeUserModel(UserModel):
    # return P(Y=1|X) across grid locations
    def evaluate_grid(self, examples):
        pr_true = self.class_density(examples, True)
        pr_false = self.class_density(examples, False)
        if pr_true is None or pr_false is None:
            return None

        # assume noninformative prior, so P(Y=0) = P(Y=1)
        return pr_true / (pr_true + pr_false)

    # compute P(X|Y) across grid locations X for given class label Y
    def class_density(self, examples, class_label):
        class_examples = np.array([loc for loc, label in examples if label == class_label])
        if len(class_examples) == 0:
            return None

        result = np.empty(self.settings.DIM)
        for loc in self.settings.LOCATIONS:
            result[loc] = self.class_density_at(loc, class_examples)
        return result


# TODO: to improve performance, if bw is some multiple of the identity, just call
# sklearn.neighbors.KernelDensity
class KDEUserModel(GenerativeUserModel):
    def __init__(self, settings, bw):
        super(self.__class__, self).__init__(settings)
        self.bw = bw
        self.name = "KDE"

    def class_density_at(self, loc, class_examples):
        d = len(self.settings.DIM)
        n = len(class_examples)

        xdiff = np.array(loc) - class_examples  # n x d
        bwx = np.linalg.solve(self.bw, xdiff.T)  # d x n
        # like np.diag(np.dot(x, bwx)), but without computing off-diagonal elements
        xbwx = (xdiff.T * bwx).sum(0)  # n
        factor = 1.0 / (n * (2*math.pi)**(d/2.0) * np.linalg.det(self.bw)**0.5)
        return factor * np.exp(-0.5 * xbwx).sum()


# Generalized context model
# Uses notation in BayesGCM (Vanpaemel, 2009) with beta = 1/2, gamma = 1 and
# deterministic feedback (a_j = 1)
class GCMUserModel(GenerativeUserModel):
    def __init__(self, settings, c, alpha=2, r=2):
        super(self.__class__, self).__init__(settings)
        self.c = c    # determines kernel width
        self.alpha = alpha    # distance is raised to the alpha power
        self.r = r    # distance is calculated according to l_r metric
        d = len(settings.DIM)
        self.w = np.ones(d) / d    # uniform attention weights
        self.name = "GCM"

    def class_density_at(self, loc, class_examples):
        xdiff = self.w * abs(np.array(loc) - class_examples)  # n x d
        xdist = (xdiff ** self.r).sum(axis=1) ** (1.0/self.r)
        return np.exp(-self.c * xdist ** self.alpha).sum()
