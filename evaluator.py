#!/usr/bin/env python
import datetime
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

from user_model import LinearSVMUserModel, RBFSVMUserModel
from teacher import RandomTeacher, GridTeacher, OptimalTeacher
from history import History
from ground_truth import error_to_accuracy, SimpleLinearGroundTruth, GeneralLinearGroundTruth, \
    SimplePolynomialGroundTruth

class Settings(object):
    def __init__(self, DIM, N_EXAMPLES, RUN_DIR):
        self.DIM = DIM
        self.N_EXAMPLES = N_EXAMPLES
        self.RUN_DIR = RUN_DIR
        self.LOCATIONS = self.compute_locations()
        self.GRID_CMAP = matplotlib.colors.ListedColormap(['dimgray', 'silver'])

    def compute_locations(self):
        D = len(self.DIM)
        coords_by_dim = [range(b) for b in self.DIM]
        locations = np.vstack(np.meshgrid(*coords_by_dim)).reshape(D, -1).T
        return [tuple(loc) for loc in locations]

    def dim_string(self):
        return 'x'.join([str(b) for b in self.DIM])


class TeacherConfig(object):
    def __init__(self, teacher, reps):
        self.teacher = teacher
        self.reps = reps


# Run active learning with given teacher. User behaves according to user model.
def run(settings, user_model, teacher, ground_truth):
    print "Running active learning with %s grid, %s user model, %s teacher" % \
        (ground_truth.name, user_model.name, teacher.name)

    history = History(user_model.prior)
    for i in range(settings.N_EXAMPLES):
        example = teacher.next_example(history)
        history.add_example(example)
        prediction = user_model.predict_grid(history.examples)
        history.add_prediction(prediction)
        # print "examples: " + str(history.examples)
        # print prediction

    if len(settings.DIM) == 2:
        # plotting history only supported for two dimensions
        history.plot(
            filename="%s/%s-%s-%s" % (settings.RUN_DIR, ground_truth.name, user_model.name, teacher.name),
            title="Active learning with %s user model, %s teacher\n%s grid with %s" % \
                (user_model.name, teacher.name, settings.dim_string(), str(ground_truth)),
            settings=settings)
    return history

def compute_teacher_accuracies(settings, user_model, teacher, ground_truth):
    history = run(settings, user_model, teacher, ground_truth)
    errors = [ground_truth.prediction_error(prediction) for prediction in history.predictions]
    return [error_to_accuracy(error) for error in errors]

def aggregate_teacher_accuracies(settings, user_model, teacher_configs, ground_truth):
    teacher_accuracies = []
    for config in teacher_configs:
        teacher_name = config.teacher.name
        if config.reps == 1:
            teacher_accuracies.append(
                (teacher_name, compute_teacher_accuracies(settings, user_model, config.teacher, ground_truth))
            )
        else:
            # compute median, 5th and 95th percentiles
            all_reps = np.vstack([compute_teacher_accuracies(
                settings, user_model, config.teacher, ground_truth) for _ in range(config.reps)])
            teacher_accuracies += [
                ('%s-p95' % teacher_name, np.percentile(all_reps, 95, axis=0)),
                ('%s-median' % teacher_name, np.percentile(all_reps, 50, axis=0)),
                ('%s-p05' % teacher_name, np.percentile(all_reps, 5, axis=0))
            ]

    return teacher_accuracies

def plot_teacher_accuracy(teacher_accuracies, filename, title):
    plt.figure()

    for name, accuracies in teacher_accuracies:
        plt.plot(range(1, len(accuracies)+1), accuracies, label=name, linestyle='-', linewidth=2)

    axes = plt.gca()
    axes.set_ylim([0, 1.1])

    plt.xlabel("Teaching examples")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend(loc='lower right')

    fig = plt.gcf()
    fig.set_size_inches(8, 8)
    fig.savefig('%s.png' % filename, dpi=100)

    plt.close()

# Simulate user behaving exactly according to user model. Compare teachers.
def eval_omniscient_teachers(ground_truth, user_model, settings):
    if len(settings.DIM) == 2:
        # plotting ground truth only supported for two dimensions
        ground_truth.plot()

    random_teacher = RandomTeacher(settings, ground_truth, with_replacement=True)
    grid_teacher = GridTeacher(settings, ground_truth, with_replacement=True)
    optimal_teacher = OptimalTeacher(settings, ground_truth, user_model, with_replacement=True)

    teacher_configs = [
        TeacherConfig(random_teacher, 20),
        TeacherConfig(grid_teacher, 20),
        TeacherConfig(optimal_teacher, 1)
    ]
    teacher_accuracies = aggregate_teacher_accuracies(settings, user_model, teacher_configs, ground_truth)
    plot_teacher_accuracy(teacher_accuracies, 
        filename='%s/%s-%s-teacher-accuracy' % (settings.RUN_DIR, ground_truth.name, user_model.name),
        title="Comparison of teacher accuracy with %s user model\n%s grid with %s" % \
            (user_model.name, settings.dim_string(), str(ground_truth))
    )


def all_simulations():
    run_dir = datetime.datetime.now().strftime("%Y%m%d %H%m")
    os.mkdir(run_dir)

    settings2d = Settings(DIM=(6, 13), N_EXAMPLES=16, RUN_DIR=run_dir)
    eval_omniscient_teachers(
        ground_truth=GeneralLinearGroundTruth(settings2d),
        user_model=LinearSVMUserModel(settings2d),
        settings=settings2d
    )
    eval_omniscient_teachers(
        ground_truth=SimplePolynomialGroundTruth(2, settings2d),
        user_model=RBFSVMUserModel(settings2d),
        settings=settings2d
    )
    eval_omniscient_teachers(
        ground_truth=SimplePolynomialGroundTruth(3, settings2d),
        user_model=RBFSVMUserModel(settings2d),
        settings=settings2d
    )

    settings3d = Settings(DIM=(5, 5, 5), N_EXAMPLES=27, RUN_DIR=run_dir)
    eval_omniscient_teachers(
        ground_truth=GeneralLinearGroundTruth(settings3d),
        user_model=LinearSVMUserModel(settings3d),
        settings=settings3d
    )
    eval_omniscient_teachers(
        ground_truth=SimplePolynomialGroundTruth(2, settings3d),
        user_model=RBFSVMUserModel(settings3d),
        settings=settings3d
    )

if __name__ == "__main__":
    all_simulations()
