import numpy as np
import pandas as pd
from task import Task
from tqdm import tqdm, tqdm_notebook
import math

def exec_episode(
    task,
    agent,
    store_transition=False,
    add_noise=False,
    output_detailed_trace=False):
    """Execute an episode.

    Arguments:
        task: The task that wraps the simulator.
        agent: The agent.

    Optional arguments:
        store_transition: Whether to store the transition data in the agent.
        add_noise: Whether to add noise to the agent's actions
        output_detailed_trace: Whether to return a detailed policy trace.
    """
    def _trace(sim, rotor_speeds, reward):
        return {
            'time': sim.time,
            'x': sim.pose[0],
            'y': sim.pose[1],
            'z': sim.pose[2],
            'phi': sim.pose[3],
            'theta': sim.pose[4],
            'psi': sim.pose[5],
            'x_velocity': sim.v[0],
            'y_velocity': sim.v[1],
            'z_velocity': sim.v[2],
            'phi_velocity': sim.angular_v[0],
            'theta_velocity': sim.angular_v[1],
            'psi_velocity': sim.angular_v[2],
            'rotor_speed1': rotor_speeds[0],
            'rotor_speed2': rotor_speeds[1],
            'rotor_speed3': rotor_speeds[2],
            'rotor_speed4': rotor_speeds[3],
            'rewards': reward
        }
    # Reset the agent, this also resets the task.
    obs = agent.reset()
    # Initialize the episode rewards and steps
    episode_reward, episode_step = 0.0, 0
    # If we are returning the trace, intialize it.
    if output_detailed_trace:
        detailed_traces = []
    # Loop until the episode ends.
    while True:
        # Request the agent from the agent, optionally adding noise.
        action = agent.act(obs, add_noise=add_noise)
        # Simulate the result of the action
        new_obs, r, done = task.step(action)
        # Update the reward
        episode_reward += r
        # Update the step count in the episode
        episode_step += 1
        # Optionally append to the trace
        if output_detailed_trace:
            detailed_traces.append(_trace(task.sim, action, r))
        # Optionally store the transition data
        if store_transition:
            agent.store_transition(obs, action, r, new_obs, done)
        # Transition to the new state.
        obs = new_obs
        # Break if the episode is done
        if done:
            break
    # Output the reward, steps and, optionally, the trace
    if output_detailed_trace:
        return episode_reward, episode_step, pd.DataFrame(detailed_traces)
    else:
        return episode_reward, episode_step

def train(
    task,
    agent,
    nb_epochs=100,
    nb_train_steps_per_epoch=20,
    best_policy_path=None,
    notebook=False,
    adaptive_learning=True):

    """Train the DDPG agent on the task.

    Arguments:
        task: The task that wraps the simulator.
        agent: The DDPG agent to train.

    Optional arguments:
        nb_epochs: The number of epochs (episodes).
        nb_train_steps_per_epoch: The number of training steps for each epoch.
        best_policy_path: The path to write the best model.
    """
    # Maintain the best reward, so that we can save the corresponding model.
    best_reward = -np.infty
    
    # Record the evaluated rewards for each epoch.
    rewards = []

    t = tqdm_notebook if notebook else tqdm

    # Record intermediate output in the progress bar
    with t(range(nb_epochs)) as pbar:
        # Iterate over the epochs
        for _ in pbar:
            # Simulate an episode, allowing noise in the actions
            # and storing the resulting transitions.
            exec_episode(task, agent, store_transition=True, add_noise=True)
            # Check if the agent is ready to train, this occurs 
            # when there is sufficient data in the replay buffer.

            if agent.is_ready_to_train():
                for _ in range(nb_train_steps_per_epoch):
                    agent.train()
            # Simulate an episode without noise.
            reward, steps = exec_episode(task, agent, store_transition=True)

            # Maintain the best reward results
            if reward > best_reward:
                best_reward = reward
                # If a path exists, write the model to the file.
                if best_policy_path:
                    agent.save_policy(best_policy_path)
            # Update the progress bar
            pbar.set_postfix(
                best_reward=best_reward,
                reward=reward,
                steps=steps)
            # Add the lastest rewards
            rewards.append(reward)
    # Return the rewards as a Pandas data frame.
    return pd.DataFrame({'rewards': rewards})