#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from user_model import SVMUserModel
from teacher import RandomTeacher, OptimalTeacher
from history import History

class Settings(object):
    DIM = (6, 13)
    N_EXAMPLES = 10
    LOCATIONS = [(x,y) for x in range(DIM[0]) for y in range(DIM[1])]

settings = Settings()

def generate_ground_truth():
    # left half is 0, right half is 1
    d = settings.DIM
    col = np.matrix(np.ones(d[0])).transpose()
    row = np.matrix(np.ones(d[1]))
    row[0, 0:(d[1]/2)] = 0
    return col * row

# Run active learning with given teacher. User behaves according to user model.
def run(user_model, teacher):
    print "Running active learning with %s and %s" % \
        (user_model.__class__.__name__, teacher.__class__.__name__)

    history = History(user_model.prior)
    for i in range(settings.N_EXAMPLES):
        example = teacher.next_example(history)
        history.add_example(example)
        prediction = user_model.predict_grid(history)
        history.add_prediction(prediction)
        print "examples: " + str(history.examples)
        print prediction

    history.plot(filename="%s-%s" % (user_model.name, teacher.name),
        title="Active learning with %s user model, %s teacher" % (user_model.name, teacher.name))
    return history

# Compute 0-1 loss between prediction and ground truth, averaged over all grid points.
# This is the risk if all grid points are weighted equally.
# Also equals 1 - accuracy, where accuracy = (TP + TN) / n_grid_pts
def compute_error(prediction, ground_truth):
    if prediction is None:
        return None

    d = settings.DIM
    n_grid_pts = d[0] * d[1]
    return np.sum(abs(prediction - ground_truth)) / n_grid_pts

def error_to_accuracy(error):
    return 0 if error is None else 1 - error

def plot_teacher_accuracy(teacher_errors, filename, title):
    plt.figure()

    for name, errors in teacher_errors.items():
        accuracies = [error_to_accuracy(error) for error in errors]
        plt.plot(range(1, len(accuracies)+1), accuracies, label=name, linestyle='-', linewidth=2)

    plt.axis([1, 10, 0, 1.1])
    plt.xlabel("Teaching examples")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend(loc='lower right')

    fig = plt.gcf()
    fig.set_size_inches(6, 6)
    fig.savefig('%s.png' % filename, dpi=100)

# Compute prediction error over course of training for each teacher, and plot together.
def compare(teacher_histories, ground_truth, user_model):
    teacher_errors = dict()
    for name, history in teacher_histories.items():
        teacher_errors[name] = [compute_error(prediction, ground_truth) for prediction in history.predictions]

    plot_teacher_accuracy(teacher_errors, filename='%s-teacher-accuracy' % user_model.name,
        title="Comparison of teacher accuracy with %s user model" % user_model.name)
    return teacher_errors

# Simulate user behaving exactly according to user model. Compare teachers.
def eval_teachers_assuming_user_model():
    ground_truth = generate_ground_truth()
    user_model = SVMUserModel(settings)
    random_teacher = RandomTeacher(settings, ground_truth)
    optimal_teacher = OptimalTeacher(settings, ground_truth, user_model)

    teachers = [random_teacher]#, optimal_teacher]
    teacher_histories = dict(
        [(teacher.name, run(user_model, teacher)) for teacher in teachers])
    compare(teacher_histories, ground_truth, user_model)

if __name__ == "__main__":
    eval_teachers_assuming_user_model()
