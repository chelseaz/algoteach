#!/usr/bin/env python
import numpy as np
from sklearn import svm

class UserModel(object):
    def __init__(self, settings):
        self.name = "base"
        self.settings = settings
        self.prior = None

    # predict label of all locations
    def predict_grid(self, history):
        predict = lambda p: 1 if p >= 0.5 else 0
        vpredict = np.vectorize(predict)
        return vpredict(self.evaluate_grid(history))

    # get probability of label 1 for all locations
    def evaluate_grid(self, history):
        pass

class SVMUserModel(UserModel):
    def __init__(self, settings):
        super(self.__class__, self).__init__(settings)
        self.name = "SVM"

    def predict_grid(self, history):
        # fit SVM to all examples in history
        model = svm.SVC(kernel='linear')  # default params include RBF kernel
        X = [loc for (loc, _) in history.examples]
        y = [label for (_, label) in history.examples]
        if len(set(y)) < 2:
            # we don't have examples for both labels yet
            return None

        model.fit(X, y)
        prediction_list = model.predict(self.settings.LOCATIONS)

        d = self.settings.DIM
        return np.matrix(prediction_list).reshape((d[0], d[1]))

