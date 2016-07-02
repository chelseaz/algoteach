#!/usr/bin/env python
import numpy as np
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

def compare(teacher_preds):
    pass

# Simulate user behaving exactly according to user model. Compare teachers.
def eval_teachers_assuming_user_model():
    ground_truth = generate_ground_truth()
    user_model = SVMUserModel(settings)
    random_teacher = RandomTeacher(settings, ground_truth)
    optimal_teacher = OptimalTeacher(settings, ground_truth, user_model)

    compare(dict(random=run(user_model, random_teacher),
                 optimal=run(user_model, optimal_teacher)))

if __name__ == "__main__":
    eval_teachers_assuming_user_model()
