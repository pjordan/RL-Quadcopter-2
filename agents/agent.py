import numpy as np
from .noise import OUNoise
from .replay_buffer import ReplayBuffer
from . import layers as norm_layers
from keras import layers, models, optimizers
from keras import backend as K
from keras import regularizers, initializers

def create(task, hyperparameters={}):
    state_size = task.state_size
    action_size = task.action_size
    action_low = np.array([task.action_low] * action_size)
    action_high = np.array([task.action_high] * action_size)
    return DDPG(task, state_size, action_size, action_low, action_high, **hyperparameters)

class DDPG():
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(
            self,
            task,
            state_size,
            action_size,
            action_low,
            action_high, 
            buffer_size=100000,
            batch_size=64,
            gamma=0.99,
            tau=0.1,
            exploration_mu=0.0,
            exploration_theta=0.15,
            exploration_sigma=0.2,
            actor_learning_rate=1e-3,
            critic_learning_rate=1e-3,
            critic_l2_reg=1e-2):
        self.task = task
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high

        # Actor (Policy) Model
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high, learning_rate=actor_learning_rate)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high,  learning_rate=actor_learning_rate)

        # Critic (Value) Model
        self.critic_local = Critic(self.state_size, self.action_size, learning_rate=critic_learning_rate, critic_l2_reg=critic_l2_reg)
        self.critic_target = Critic(self.state_size, self.action_size,  learning_rate=critic_learning_rate, critic_l2_reg=critic_l2_reg)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Noise process
        self.exploration_mu = exploration_mu
        self.exploration_theta = exploration_theta
        self.exploration_sigma = exploration_sigma
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        # Replay memory
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Algorithm parameters
        self.gamma = gamma  # discount factor
        self.tau = tau  # for soft update of target parameters

    def reset(self):
        self.noise.reset()
        state = self.task.reset()
        return state

    def is_ready_to_train(self):
        return len(self.memory) > self.batch_size

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def act(self, state, add_noise=False):
        """Returns actions for given state(s) as per current policy."""
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]
        if add_noise:
            return list(action + self.noise.sample())
        else:
            return list(action)

    def save_policy(self, path):
        self.actor_local.model.save(path, include_optimizer=False)

    def train(self):
        experiences = self.memory.sample()
        # Learn by random policy search, using a reward-based score
        
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models
        #     Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])  # custom training function

        # Soft-update target models
        self._soft_update(self.critic_local.model, self.critic_target.model)
        self._soft_update(self.actor_local.model, self.actor_target.model)   

    def adjust_clipping(self, factor):
        self.actor_target.optimizer.clipnorm = factor
        self.actor_local.optimizer.clipnorm = factor
        self.critic_target.optimizer.clipnorm = factor
        self.critic_local.optimizer.clipnorm = factor

    def _soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)

class Actor:
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, action_low, action_high, learning_rate=1e-3):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.learning_rate = learning_rate

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        # Define input layer (states)
        states = layers.Input(shape=(self.state_size,), name='states')
        
        # Add hidden layers
        net = layers.BatchNormalization()(states)
        net = layers.Dense(units=32, activation=None)(net)
        net = layers.BatchNormalization()(net)
        net = layers.Activation('relu')(net)
        net = layers.Dense(units=32, activation=None)(net)
        net = layers.BatchNormalization()(net)
        net = layers.Activation('relu')(net)

        # Add final output layer with sigmoid activation
        raw_actions = layers.Dense(
            units=self.action_size,
            activation='sigmoid',
            name='raw_actions')(net)

        # Scale [0, 1] output for each action dimension to proper range
        actions = norm_layers.MinMaxDenormalization(
            self.action_low,
            self.action_high,
            name='actions')(raw_actions) 
        
        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)

        # Define loss function using action value (Q value) gradients
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        # Incorporate any additional losses here (e.g. from regularizers)

        # Define optimizer and training function
        self.optimizer = optimizers.Adam(lr=self.learning_rate)
        updates_op = self.optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)

class Critic:
    """Critic (Value) Model."""

    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=1e-3,
        critic_l2_reg=1e-2):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.critic_l2_reg = critic_l2_reg
        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Add hidden layer(s) for state pathway
        net_states = layers.BatchNormalization()(states)
        net_states = layers.Dense(
            units=32,
            activation=None,
            kernel_regularizer=regularizers.l2(self.critic_l2_reg))(net_states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Activation('relu')(net_states)

        # Add hidden layer(s) for action pathway
        net_actions = layers.BatchNormalization()(actions)
        net_actions = layers.Dense(
            units=32,
            activation=None,
            kernel_regularizer=regularizers.l2(self.critic_l2_reg))(net_actions)
        net_actions = layers.BatchNormalization()(net_actions)
        net_actions = layers.Activation('relu')(net_actions)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Combine state and action pathways
        net = layers.Concatenate()([net_states, net_actions])
        net = layers.Dense(
            units=32,
            activation=None,
            kernel_regularizer=regularizers.l2(self.critic_l2_reg))(net)
        net = layers.BatchNormalization()(net)
        net = layers.Activation('relu')(net)

        # Add more layers to the combined network if needed

        # Add final output layer to prduce action values (Q values)
        Q_values = layers.Dense(
            units=1,
            kernel_initializer=initializers.RandomUniform(minval=-1, maxval=1),
            name='q_values')(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        self.optimizer = optimizers.Adam(lr=self.learning_rate)
        self.model.compile(optimizer=self.optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)