#!/usr/bin/env python

import math
import matplotlib.pyplot as plt

class History(object):
    def __init__(self, user_prior):
        self.prior = user_prior
        self.examples = []    # teaching examples ((x, y), 0/1)
        self.predictions = []    # user predictions of grid labels

    def add_example(self, example):
        self.examples.append(example)

    def add_prediction(self, prediction):
        self.predictions.append(prediction)

    def plot_iteration(self, plt, i, settings):
        prediction = self.predictions[i]

        plt.axis('off')
        plt.title("Prediction after %d iterations" % (i+1))
        # label=0 is dark gray, label=1 is silver
        plt.imshow(prediction.T, cmap=settings.GRID_CMAP, interpolation='none', origin='lower')
        for j in range(i+1):
            loc, label = self.examples[j]
            x, y = loc
            c = 'maroon' if j == i else 'black'
            plt.annotate(s=str(j+1), xy=(x, y), color=c)

    def plot(self, filename, title, settings):
        plt.figure()
        plt.suptitle(title)

        valid_iter = [i for i in range(len(self.examples)) if self.predictions[i] is not None]
        N = len(valid_iter)
        nrow = int(math.ceil(N/2.0))
        for fignum, i in enumerate(valid_iter):
            plt.subplot(nrow, 2, fignum+1)
            self.plot_iteration(plt, i, settings)

        fig = plt.gcf()
        fig.set_size_inches(8, 12)
        fig.savefig('%s.png' % filename, dpi=100)

        plt.close()
