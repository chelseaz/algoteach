#!/usr/bin/env python
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
