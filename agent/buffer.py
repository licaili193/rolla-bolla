import numpy as np

class ReplayBuffer:
    """
    A replay buffer class that stores and samples transitions for reinforcement learning.

    Parameters:
    - max_size (int): The maximum size of the replay buffer.
    - input_shape (tuple): The shape of the input state.
    - n_actions (int): The number of possible actions.

    Attributes:
    - mem_size (int): The current size of the replay buffer.
    - mem_cntr (int): The counter for the number of stored transitions.
    - state_memory (ndarray): The memory for storing the current states.
    - new_state_memory (ndarray): The memory for storing the next states.
    - action_memory (ndarray): The memory for storing the actions.
    - reward_memory (ndarray): The memory for storing the rewards.
    - terminal_memory (ndarray): The memory for storing the terminal flags.
    """

    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def store_transition(self, state, action, reward, state_, done):
        """
        Store a transition in the replay buffer.

        Parameters:
        - state (ndarray): The current state.
        - action (ndarray): The action taken.
        - reward (float): The reward received.
        - state_ (ndarray): The next state.
        - done (bool): Whether the episode is done.
        """
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        """
        Sample a batch of transitions from the replay buffer.

        Parameters:
        - batch_size (int): The size of the batch to sample.

        Returns:
        - states (ndarray): The sampled states.
        - actions (ndarray): The sampled actions.
        - rewards (ndarray): The sampled rewards.
        - states_ (ndarray): The sampled next states.
        - dones (ndarray): The sampled terminal flags.
        """
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones