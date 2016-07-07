#!/usr/bin/env python
import numpy as np
from sklearn import svm

class UserModel(object):
    def __init__(self, settings):
        self.name = "base"
        self.settings = settings
        self.prior = None

    # predict label of all locations
    def predict_grid(self, examples):
        predict = lambda p: 1 if p >= 0.5 else 0
        vpredict = np.vectorize(predict)
        return vpredict(self.evaluate_grid(examples))

    # get probability of label 1 for all locations
    def evaluate_grid(self, examples):
        raise NotImplementedError

class SVMUserModel(UserModel):
    def __init__(self, settings):
        super(self.__class__, self).__init__(settings)
        self.name = "SVM"

    def predict_grid(self, examples):
        # fit SVM to all examples
        model = svm.SVC(kernel='linear')  # default params include RBF kernel
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

