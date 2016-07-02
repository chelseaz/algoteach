#!/usr/bin/env python

class History(object):
    def __init__(self, user_prior):
        self.examples = []    # teaching examples ((x, y), 0/1)
        self.predictions = [user_prior]    # user predictions of grid labels

    def add_example(self, example):
        self.examples.append(example)

    def add_prediction(self, prediction):
        self.predictions.append(prediction)
