import numpy as np
import gym
from gym import spaces


def _gen_adjacency_lists(n_rooms, n_portals, rng):
    """ Generate an n_rooms x n_portals array where each row is a list of rooms that can be reached from that room. """

    assert n_portals <= n_rooms

    for attempt in range(1000):
        adjacency_lists = np.zeros((n_rooms, n_portals), dtype=np.int32)
        for i in range(n_rooms):
            adjacency_lists[i, :] = rng.choice(n_rooms, n_portals, replace=False)

        # Generate connectivity matrix from adjacency list and check if it is connected
        C = np.zeros((n_rooms, n_rooms), dtype=bool)
        for i in range(n_rooms):
            C[i, adjacency_lists[i, :]] = True
        C = np.linalg.matrix_power(C, n_rooms)
        disconnected = not np.all(C)
        if not disconnected:
            return adjacency_lists
    # Return error if we could not generate a connected graph
    raise ValueError("Did not generate a connected graph")

class GraphWorld(gym.Env):
    """Generate a GraphWorld environment"""
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 n_rooms,
                 seed,
                 episode_length = 100,
                 n_portals = 4,
                 p_correct_transition = 0.7,
                 p_correct_observation = 0.8,
                 discount = 0.99,
                 rewards = [10.0, 3.0, -5.0, -10.0]):

        super().__init__()

        # Create a non-defaut numpy rng
        rng = np.random.RandomState(seed)

        # Generate random connected set of ajacency lists
        self.adjacency_lists = _gen_adjacency_lists(n_rooms, n_portals, rng)

        # Assign rewards to rooms
        n_rewards = len(rewards)
        rooms = rng.choice(n_rooms, n_rewards, replace=False)
        room_rewards = dict(zip(rooms, rewards))

        # Store parameters
        self.n_rooms = n_rooms
        self.episode_length = episode_length
        self.n_portals = n_portals
        self.p_correct_transition = p_correct_transition
        self.p_correct_observation = p_correct_observation
        self.discount = discount
        self.room_rewards = room_rewards

        # Set up gym spaces
        self.action_space = spaces.Discrete(n_portals)
        self.observation_space = spaces.Box(low=0, high=1, shape=(n_rooms + n_portals,), dtype=np.float32)

    def observe (self):
        # Observe the state
        if np.random.rand() < self.p_correct_observation:
            return self.state
        else:
            p = np.arange(self.n_rooms) != self.state # Keep all but the current state
            return np.random.choice(np.arange(self.n_rooms)[p])




    def _combine(self, observation, action):
        return np.concatenate([np.arange(self.n_rooms) == observation,
               np.arange(self.n_portals) == action], dtype=np.float32)

    def step(self, action):
        reward = 0

        previous_state = self.state

        # Take the adjacency list corresponding to the previous state
        adjacency_list = self.adjacency_lists[previous_state, :]

        # Transition to a new state
        if np.random.rand() < self.p_correct_transition:
            self.state = adjacency_list[action]
        else:
            p = np.arange(self.n_portals) != action # Keep all but the selected action
            self.state = np.random.choice(adjacency_list[p])

        # Observe the new state
        observation = self.observe()

        if observation in self.room_rewards:
            reward = self.room_rewards[observation]

        done = reward > 0

        self.time += 1
        truncated = self.time >= self.episode_length

        info = {}

        return self._combine(observation, action), reward, done or truncated, info

    def reset(self, options=None, seed=None):
        # Reset the sate of the environment to an initial state
        initial_states = list(set(range(self.n_rooms)) - self.room_rewards.keys())
        self.state = np.random.choice(initial_states)

        self.time = 0
       
        return self._combine(self.observe(), -1)

    def render(self, mode='human'):
        pass

    def close(self):
        pass
