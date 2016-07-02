#!/usr/bin/env python

import random

class Teacher(object):
    def __init__(self, settings, ground_truth):
        self.settings = settings
        self.ground_truth = ground_truth
        self.locations = set(settings.LOCATIONS)  # immutable

    def next_example(self, history):
        pass

class RandomTeacher(Teacher):
    def next_example(self, history):
        shown_locs = set([loc for (loc, _) in history.examples])
        next_loc = random.sample(self.locations - shown_locs, 1)[0]
        return (next_loc, self.ground_truth[next_loc])

class OptimalTeacher(Teacher):
    def __init__(self, settings, ground_truth, user_model):
        super(self.__class__, self).__init__(settings, ground_truth)
        self.user_model = user_model

    def next_example(self, history):
        pass
