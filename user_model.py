#!/usr/bin/env python
import numpy as np
from sklearn import svm


class OfflineUserModel(object):
    def __init__(self, settings):
        self.name = "offline-base"
        self.settings = settings

    # predict label of all locations
    def predict_grid(self, examples):
        predict = lambda p: 1 if p >= 0.5 else 0
        vpredict = np.vectorize(predict)
        return vpredict(self.evaluate_grid(examples))

    # get probability of label 1 for all locations
    def evaluate_grid(self, examples):
        raise NotImplementedError


class SVMUserModel(OfflineUserModel):
    def predict_grid(self, examples):
        # fit SVM to all examples
        model = self.get_model()
        X = [loc for (loc, _) in examples]
        y = [label for (_, label) in examples]
        if len(set(y)) < 2:
            # we don't have examples for both labels yet
            return None

        model.fit(X, y)
        prediction_list = model.predict(self.settings.LOCATIONS)

        prediction_array = np.empty(self.settings.DIM)
        for loc, pred in zip(self.settings.LOCATIONS, prediction_list):
            prediction_array[loc] = pred

        return prediction_array

    def get_model(self):
        raise NotImplementedError


class LinearSVMUserModel(SVMUserModel):
    def __init__(self, settings):
        super(self.__class__, self).__init__(settings)
        self.name = "linear SVM"

    def get_model(self):
        return svm.SVC(kernel='linear')


class RBFSVMUserModel(SVMUserModel):
    def __init__(self, settings):
        super(self.__class__, self).__init__(settings)
        self.name = "RBF SVM"

    def get_model(self):
        return svm.SVC(kernel='rbf', C=528.0, gamma=0.001)


# Mutable by definition
class OnlineUserModel(object):
    def __init__(self, settings):
        self.name = "online-base"
        self.settings = settings

    def add_example(self, example):
        raise NotImplementedError

    # predict label of all locations
    def predict_grid(self):
        predict = lambda p: 1 if p >= 0.5 else 0
        vpredict = np.vectorize(predict)
        return vpredict(self.evaluate_grid())

    # get probability of label 1 for all locations
    def evaluate_grid(self):
        raise NotImplementedError


# Performs function approximation using an online kernel machine
# See Dragan and Srinivasa (RoMan 2012)
class RBFOKMUserModel(OnlineUserModel):
    def __init__(self, settings, prior, eta, lambda_param, w):
        super(self.__class__, self).__init__(settings)
        assert prior.shape == settings.DIM
        self.prior = prior
        self.eta = eta  # new example weight
        self.lambda_param = lambda_param  # additional forgetting rate
        self.w = w  # determines kernel width
        self.examples = []
        self.offset_coefs = np.empty(0)  # offset coefficients corresponding to examples
        self.name = "RBF OKM"

    def add_example(self, example):
        new_loc, new_label = example
        self.offset_coefs *= 1 - self.eta * self.lambda_param
        new_alpha = new_label - self.evaluate(new_loc)
        new_offset_coef = self.eta * new_alpha
        self.offset_coefs = np.append(self.offset_coefs, [new_offset_coef])
        self.examples.append(new_loc)

    # Return function approximation at loc X, P(Y=1|X)
    def evaluate(self, loc):
        offset = self.offset_coefs * np.array([self.kernel(old_loc, loc) for old_loc in self.examples])
        return self.prior[loc] + offset

    def kernel(self, loc1, loc2):
        return np.exp(-0.5 * self.w * ((np.array(loc1) - np.array(loc2)) ** 2).sum())

    def evaluate_grid(self):
        eval_array = np.empty(self.settings.DIM)
        for loc in self.settings.LOCATIONS:
            eval_array[loc] = self.evaluate(loc)

        return eval_array
