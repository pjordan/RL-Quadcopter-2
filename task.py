import numpy as np
from physics_sim import PhysicsSim
import sklearn.gaussian_process.kernels

def similarity(length_scale=1.0):
    kernel = sklearn.gaussian_process.kernels.RBF(length_scale=length_scale)
    return lambda x,y: kernel(x.reshape(1,-1), y.reshape(1,-1))[0][0]

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(
        self,
        init_pose=None,
        init_velocities=None, 
        init_angle_velocities=None,
        runtime=5.,
        target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.individual_state_size = 6
        self.state_size = self.action_repeat * self.individual_state_size
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4
        self.actions_low = np.array([self.action_low] * self.action_size)
        self.actions_high = np.array([self.action_high] * self.action_size)

        self.kernel_position  = similarity(length_scale=1)
        self.kernel_velocity  = similarity(length_scale=1)
        self.kernel_acceleration  = similarity(length_scale=2)
        self.kernel_height  = similarity(length_scale=1)
        self.kernel_angle  = similarity(length_scale=0.2)

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
        self.target_vel = target_pos if target_pos is not None else np.array([0., 0., 0.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        target_height = self.target_pos[2]
        height = self.sim.pose[2]

        # Reward for not being done.
        reward = 1.0

        # Reward for keeping close to the target position
        reward += self.kernel_position(self.sim.pose[:3],self.target_pos) * (1+self.sim.time)

        # Reward for keeping close to the target height
        reward += self.kernel_height(height,target_height)
        
        # Reward for keeping close to level
        reward += self.kernel_angle(self.sim.pose[5],np.array([0]))

        # Reward for minimal accelaration
        #reward += self.kernel_acceleration(self.sim.linear_accel,np.array([0,0,0]))

        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        return np.concatenate([self.sim.pose] * self.action_repeat)
