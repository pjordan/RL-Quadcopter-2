import numpy as np
from task import Task
from agents import agent as ddpg
import runner

try:
    import gp
except ImportError:
    print("Please download https://raw.githubusercontent.com/thuijskens/bayesian-optimization/master/python/gp.py")
    exit()

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
# config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))

N_EPOCHS = 1000
N_ITERS = 20
PRE_SAMPLES = 5

hyperparameter_meta_data = [
    {
        'name':'exploration_sigma',
        'min': 0.0,
        'max': 300.0,
    },
    {
        'name':'tau',
        'min': 1e-4,
        'max': 1e-2,
        'transform': np.log,
        'itransform': np.exp,
    },
    {
        'name':'actor_learning_rate',
        'min': 1e-5,
        'max': 1e-3,
        'transform': np.log,
        'itransform': np.exp,
    },
    {
        'name':'critic_learning_rate',
        'min': 1e-5,
        'max': 1e-3,
        'transform': np.log,
        'itransform': np.exp,
    }
]

names = [p['name'] for p in hyperparameter_meta_data]
transforms = [p['transform'] if p and 'transform' in p else lambda x: x for p in hyperparameter_meta_data]
itransforms = [p['itransform'] if p and 'itransform' in p else lambda x: x for p in hyperparameter_meta_data]
mins = [p['min'] for p in hyperparameter_meta_data]
maxes = [p['max'] for p in hyperparameter_meta_data]
transformed_mins = [t(v) for t, v in zip(transforms, mins)]
transformed_maxes = [t(v) for t, v in zip(transforms, maxes)]
bounds = np.array([transformed_mins, transformed_maxes]).T

def itransform(params):
    return [it(v) for it, v in zip(itransforms,params)]

def params_to_hyperparams(params):
    return {name:value for name, value in zip(names, itransform(params))}

def sample_loss(params):
    hyperparameters = params_to_hyperparams(params)
    print("Parameters:", hyperparameters)
    task = Task(target_pos=np.array([0., 0., 10.]))
    agent = ddpg.create(task, hyperparameters=hyperparameters) 
    results = runner.train(task, agent, nb_epochs=N_EPOCHS, nb_train_steps_per_epoch=1)
    best_reward = results.max()
    print("Reward:",best_reward)
    return best_reward

xp, yp = gp.bayesian_optimisation(
    n_iters=N_ITERS, 
    sample_loss=sample_loss, 
    bounds=bounds,
    n_pre_samples=PRE_SAMPLES)
