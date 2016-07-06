#!/usr/bin/env python

import itertools
import random

class Teacher(object):
    def __init__(self, settings, ground_truth):
        self.name = "base"
        self.settings = settings
        self.ground_truth = ground_truth
        self.locations = set(settings.LOCATIONS)  # immutable

    # set of locations that have not been shown yet
    def remaining_locs(self, history):
        shown_locs = set([loc for (loc, _) in history.examples])
        return self.locations - shown_locs

    def next_example(self, history):
        pass

class RandomTeacher(Teacher):
    def __init__(self, settings, ground_truth):
        super(self.__class__, self).__init__(settings, ground_truth)
        self.name = "random"

    def next_example(self, history):
        next_loc = random.sample(self.remaining_locs(history), 1)[0]
        return (next_loc, self.ground_truth.at(next_loc))

class OptimalTeacher(Teacher):
    def __init__(self, settings, ground_truth, user_model):
        super(self.__class__, self).__init__(settings, ground_truth)
        self.name = "optimal"
        self.user_model = user_model

    def next_example(self, history):
        return next_example_rhc(self, history)
        #return next_example_beam_search(self, history)

    def next_example_rhc(self, history, horizon=3):
        example_seqs = itertools.combinations(self.remaining_locs(history), horizon)
        for example_seq in example_seqs:
            extra_examples = [(loc, self.ground_truth.at(loc)) for loc in example_seq]
            all_examples = history.examples + extra_examples
            prediction = self.user_model.predict_grid(all_examples)


    def next_example_beam_search(self, history):
        pass

