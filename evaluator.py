#!/usr/bin/env python
import argparse
import datetime
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import shutil
import time

from user_model import *
from teacher import RandomTeacher, GridTeacher, OptimalTeacher
from ground_truth import *
from viz import *


class Settings(object):
    def __init__(self, DIM, N_EXAMPLES, RUN_DIR, TEACHER_REPS):
        self.DIM = DIM
        self.N_EXAMPLES = N_EXAMPLES
        self.RUN_DIR = RUN_DIR
        self.TEACHER_REPS = TEACHER_REPS
        self.LOCATIONS = self.compute_locations()
        self.GRID_CMAP = matplotlib.colors.ListedColormap(['dimgray', 'silver'])

    def compute_locations(self):
        D = len(self.DIM)
        coords_by_dim = [range(b) for b in self.DIM]
        locations = np.vstack(np.meshgrid(*coords_by_dim)).reshape(D, -1).T
        return [tuple(loc) for loc in locations]

    def dim_string(self):
        return 'x'.join([str(b) for b in self.DIM])

    def uniform_prior(self, p=0.5):
        prior = np.empty(self.DIM)
        prior.fill(p)
        return prior


class History(object):
    def __init__(self, user_prior):
        self.prior = user_prior
        self.examples = []    # teaching examples ((x, y), 0/1)
        self.predictions = []    # user predictions of grid labels (0/1 array)
        self.evaluations = []    # user evaluations of grid labels (array of values in [0,1])

    def add_example(self, example):
        self.examples.append(example)

    def add_prediction_result(self, prediction_result):
        self.predictions.append(prediction_result.prediction)
        self.evaluations.append(prediction_result.evaluation)


class TeacherConfig(object):
    def __init__(self, teacher, reps):
        self.teacher = teacher
        self.reps = reps


class Function(object):
    def __init__(self, f, name, formula):
        self.f = f
        self.name = name
        self.formula = formula


# Run active learning with given teacher. User behaves according to user model.
def run(settings, user_model, teacher, ground_truth):
    start_time = time.time()
    print "Running active learning with %s grid, %s user model, %s teacher" % \
        (ground_truth.name, user_model.name, teacher.name)

    history = History(user_model.prior)
    for i in range(settings.N_EXAMPLES):
        example = teacher.next_example(history)
        history.add_example(example)
        prediction_result = user_model.predict_grid(history.examples)
        history.add_prediction_result(prediction_result)
        # print "examples: " + str(history.examples)
        # print prediction

    end_time = time.time()
    print "Took %d seconds" % (end_time - start_time)

    plot_history(
        history=history,
        filename="%s/%s-%s-%s" % (settings.RUN_DIR, ground_truth.name, user_model.name, teacher.name),
        title="Active learning with %s user model, %s teacher\n%s grid with %s" % \
            (user_model.name, teacher.name, settings.dim_string(), str(ground_truth)),
        settings=settings)
    return history


# Return dataframe of metrics for a single run with this teacher. 
# Row i contains metrics for the learner prediction at time i.
def compute_teacher_metrics(settings, user_model, teacher, ground_truth):
    history = run(settings, user_model, teacher, ground_truth)
    metrics = [ground_truth.prediction_metrics(prediction) for prediction in history.predictions]
    return pd.DataFrame(metrics)


def compute_all_teachers(settings, user_model, teacher_configs, ground_truth):
    all_teacher_metrics = []
    for config in teacher_configs:
        teacher_name = config.teacher.name
        teacher_metrics_all_reps = [compute_teacher_metrics(settings, user_model, config.teacher, ground_truth) for _ in range(config.reps)]
        all_teacher_metrics.append((teacher_name, teacher_metrics_all_reps))

    return all_teacher_metrics


def aggregate_teacher_metrics(all_teacher_metrics, metric_name):
    agg_metrics = []
    for (teacher_name, teacher_metrics_all_reps) in all_teacher_metrics:
        if len(teacher_metrics_all_reps) == 1:
            teacher_metrics_df = teacher_metrics_all_reps[0]
            agg_metrics.append((teacher_name, teacher_metrics_df[metric_name].values))
        else:
            # compute median, 5th and 95th percentiles
            all_reps = np.vstack([teacher_metrics_df[metric_name].values for teacher_metrics_df in teacher_metrics_all_reps])
            agg_metrics += [
                ('%s-p95' % teacher_name, np.percentile(all_reps, 95, axis=0)),
                ('%s-median' % teacher_name, np.percentile(all_reps, 50, axis=0)),
                ('%s-p05' % teacher_name, np.percentile(all_reps, 5, axis=0))
            ]

    return agg_metrics


# Simulate user behaving exactly according to user model. Compare teachers.
def eval_omniscient_teachers(ground_truth, user_model_fns, settings):
    plot_ground_truth(ground_truth)

    if settings.TEACHER_REPS <= 0:
        return

    for user_model_fn in user_model_fns:
        user_model = user_model_fn(settings)

        random_teacher = RandomTeacher(settings, ground_truth, with_replacement=True)
        grid_teacher = GridTeacher(settings, ground_truth, with_replacement=True)
        optimal_teacher = OptimalTeacher(settings, ground_truth, user_model, with_replacement=True)

        teacher_configs = [
            TeacherConfig(random_teacher, settings.TEACHER_REPS),
            TeacherConfig(grid_teacher, settings.TEACHER_REPS),
            TeacherConfig(optimal_teacher, 1)
        ]
        all_teacher_metrics = compute_all_teachers(settings, user_model, teacher_configs, ground_truth)
        teacher_accuracies = aggregate_teacher_metrics(all_teacher_metrics, 'accuracy')
        plot_teacher_accuracy(teacher_accuracies, 
            filename='%s/%s-%s-teacher-accuracy' % (settings.RUN_DIR, ground_truth.name, user_model.name),
            title="Comparison of teacher accuracy with %s user model\n%s grid with %s" % \
                (user_model.name, settings.dim_string(), str(ground_truth))
        )


def all_simulations(args):
    # set random seed globally
    random.seed(1234)
    np.random.seed(1234)

    # prepare directory for saving files
    run_dir = datetime.datetime.now().strftime("%Y%m%d %H%M")
    if args.desc:
        run_dir = "%s-%s" % (run_dir, args.desc)
    shutil.rmtree(run_dir, ignore_errors=True)
    os.mkdir(run_dir)

    # define literal ground truth functions
    exp = Function(f=lambda x: math.exp(x)-2, name="exp", formula="e^x - 2")
    sin = Function(f=lambda x: 2*math.sin(x), name="sin", formula="2 * sin(x)")
    xsinx = Function(f=lambda x: x * math.sin(x), name="x sin x", formula="x * sin(x)")

    # other settings
    if args.dry_run:
        teacher_reps = 0    # just generate ground truth
    else:
        teacher_reps = 20

    # run experiments
    RBF1SVMUserModelFn = lambda settings: RBF1SVMUserModel(settings, nu=0.05, gamma=0.1)
    Linear2SVMUserModelFn = lambda settings: Linear2SVMUserModel(settings)
    RBF2SVMUserModelFn = lambda settings: RBF2SVMUserModel(settings, C=1.0, gamma=0.1)
    RBFOKMUserModelFn = lambda settings: RBFOKMUserModel(settings,
        prior=settings.uniform_prior(), eta=0.85, lambda_param=0.05, w=1)
    KDEUserModelFn = lambda settings: KDEUserModel(settings,
        bw=np.eye(len(settings.DIM)))
    GCMUserModelFn = lambda settings: GCMUserModel(settings, c=1.0, r=1)

    settings = Settings(DIM=(13, 6), N_EXAMPLES=16, RUN_DIR=run_dir, TEACHER_REPS=teacher_reps)
    eval_omniscient_teachers(
        ground_truth=GeneralLinearGroundTruth(settings),
        user_model_fns=[RBF1SVMUserModelFn, Linear2SVMUserModelFn],#, RBFOKMUserModelFn, KDEUserModelFn, GCMUserModelFn],
        settings=settings
    )

    # for degree in range(2, 5):
    #     eval_omniscient_teachers(
    #         ground_truth=SimplePolynomialGroundTruth(degree, settings),
    #         user_model_fns=[RBF1SVMUserModelFn, RBF2SVMUserModelFn],#, RBFOKMUserModelFn, KDEUserModelFn, GCMUserModelFn],
    #         settings=settings
    #     )
    # for fn in [exp, sin, xsinx]:
    #     eval_omniscient_teachers(
    #         ground_truth=SimpleFunctionGroundTruth(settings, fn),
    #         user_model_fns=[RBF1SVMUserModelFn, RBF2SVMUserModelFn],#, RBFOKMUserModelFn, KDEUserModelFn, GCMUserModelFn],
    #         settings=settings
    #     )

    # settings = Settings(DIM=(5, 5, 5), N_EXAMPLES=27, RUN_DIR=run_dir, TEACHER_REPS=teacher_reps)
    # eval_omniscient_teachers(
    #     ground_truth=GeneralLinearGroundTruth(settings),
    #     user_model_fns=[RBF1SVMUserModelFn, Linear2SVMUserModelFn],#, RBFOKMUserModelFn, KDEUserModelFn, GCMUserModelFn],
    #     settings=settings
    # )
    # eval_omniscient_teachers(
    #     ground_truth=SimplePolynomialGroundTruth(2, settings),
    #     user_model_fns=[RBF2SVMUserModelFn],#, RBFOKMUserModelFn, KDEUserModelFn, GCMUserModelFn],
    #     settings=settings
    # )

    # settings = Settings(DIM=(3, 3, 3, 3), N_EXAMPLES=32, RUN_DIR=run_dir, TEACHER_REPS=teacher_reps)
    # eval_omniscient_teachers(
    #     ground_truth=GeneralLinearGroundTruth(settings),
    #     user_model_fns=[RBF1SVMUserModelFn, Linear2SVMUserModelFn],#, RBFOKMUserModelFn, KDEUserModelFn, GCMUserModelFn],
    #     settings=settings
    # )
    # eval_omniscient_teachers(
    #     ground_truth=SimplePolynomialGroundTruth(2, settings),
    #     user_model_fns=[RBF2SVMUserModelFn],#, RBFOKMUserModelFn, KDEUserModelFn, GCMUserModelFn],
    #     settings=settings
    # )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run algorithmic teaching simulations.")
    parser.add_argument('--dry-run', action='store_true', help="only generate ground truth")
    parser.add_argument('--desc', type=str, help="description appended to directory name")
    args = parser.parse_args()

    all_simulations(args)
