import numpy as np
from keras.models import load_model
from .layers import MinMaxDenormalization

__ALL__ = ['create', 'load']

def create(task, policy):
    return PolicyAgent(task, policy)

def load(task, path):
    return create(task, load_model(path, custom_objects={'MinMaxDenormalization': MinMaxDenormalization}))

class PolicyAgent():
    def __init__(self, task, policy):
        self.task = task
        self.policy = policy

    def reset(self):
        return self.task.reset()

    def act(self, state, **kwargs):
        """Returns actions for given state(s) as per policy."""
        state = np.reshape(state, [-1, self.task.state_size])
        return list(self.policy.predict(state)[0])
