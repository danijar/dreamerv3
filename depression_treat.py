import numpy as np
import gym
from gym import spaces

class DepressionTreatmentEnv(gym.Env):
    # Define the action space (index corresponds to a treatment)
    ACTIONS = ["SSRI", "SNRI", "TCA", "Atypical Antipsychotic", "Psychotherapy", "ECT", "TMS", "Lifestyle", "Ketamine"]

    # Define the hidden disease states
    HIDDEN_STATES = ["MDD", "Bipolar", "Serotonin Syndrome", "No Change", "Remission"]

    # Define possible symptoms and their scores (for simplicity, using 5 symptoms)
    SYMPTOMS = ["Sadness", "Fatigue", "Mania", "Physical Agitation", "Well-being"]

    def __init__(self):
        super(DepressionTreatmentEnv, self).__init__()

        # Action space is discrete and corresponds to each treatment option
        self.action_space = spaces.Discrete(len(self.ACTIONS))
        
        # The observation space will now be a vector of symptom scores (here between 0 to 10 for each symptom)
        self.observation_space = spaces.Box(low=0, high=10, shape=(len(self.SYMPTOMS),), dtype=np.float32)
        
        # Initialize the current state as MDD
        self.current_state = 0  # MDD is the starting state

        # Define a reward matrix or a function to give rewards based on transitions (for simplicity, using a matrix)
        self.reward_matrix = np.array([
            [10, -10, -20, -5, 50],  # Rewards for state transitions after SSRI
            [10, -10, -20, -5, 50],  # Rewards for state transitions after SNRI
            [8, -10, -20, -5, 50],   # Rewards for state transitions after TCA
            [10, -10, -15, -5, 50],  # ... and so on for each treatment
            [8, 0, 0, -5, 40],
            [15, -10, 0, -5, 60],
            [8, 0, -10, -5, 40],
            [5, 0, 0, -3, 30],
            [12, -10, -15, -5, 55]
        ])

    def step(self, action):
        # Transition the state based on the action (in the real-world, this would be more complex and probabilistic)
        # For simplicity, using random transitions
        self.current_state = np.random.choice(self.observation_space.n, p=self.transition_probabilities(action))
        
        # Get reward from the matrix based on the new state
        reward = self.reward_matrix[action][self.current_state]
        
        # Check if we've reached a terminal state (Remission or severe side effect)
        done = (self.current_state == self.STATES.index("Remission")) or (self.current_state == self.STATES.index("Serotonin Syndrome"))

        return self.current_state, reward, done, {}

    def transition_probabilities(self, action):
        # Define some mock transition probabilities for each treatment (must sum to 1 for each action)
        # Here, just for illustration. In reality, this needs to be based on empirical data or expert knowledge
        probabilities = {
            0: [0.3, 0.1, 0.05, 0.4, 0.15],  # For SSRI
            1: [0.3, 0.1, 0.05, 0.4, 0.15],  # For SNRI
            2: [0.25, 0.1, 0.1, 0.4, 0.15],  # ... and so on for each treatment
            3: [0.25, 0.15, 0.05, 0.4, 0.15],
            4: [0.35, 0.05, 0, 0.45, 0.15],
            5: [0.2, 0.1, 0, 0.5, 0.2],
            6: [0.3, 0.05, 0.05, 0.45, 0.15],
            7: [0.4, 0.05, 0, 0.45, 0.1],
            8: [0.25, 0.1, 0.1, 0.4, 0.15]
        }
        return probabilities[action]
    
    def generate_symptom_scores(self, hidden_state):
        # Generate symptom scores based on the hidden state using probabilistic distributions
        # Each state has a mean and standard deviation for each symptom

        symptom_distributions = {
            0: {'mean': np.array([8, 8, 2, 2, 1]), 'std': np.array([1, 1, 1, 1, 1])},  # For MDD
            1: {'mean': np.array([6, 5, 9, 8, 2]), 'std': np.array([1, 1, 1, 1, 1])},  # For Bipolar
            2: {'mean': np.array([4, 5, 6, 9, 1]), 'std': np.array([1, 1, 1, 1, 1])},  # For Serotonin Syndrome
            3: {'mean': np.array([5, 5, 5, 5, 5]), 'std': np.array([1, 1, 1, 1, 1])},  # For No Change
            4: {'mean': np.array([1, 1, 1, 1, 9]), 'std': np.array([1, 1, 1, 1, 1])}   # For Remission
        }
        
        mean_scores = symptom_distributions[hidden_state]['mean']
        std_scores = symptom_distributions[hidden_state]['std']
        
        observation = np.random.normal(mean_scores, std_scores)
        
        # Clip scores to be between 0 and 10
        observation = np.clip(observation, 0, 10)
        
        return observation

    def reset(self):
        # Reset the environment to the initial hidden state (MDD)
        self.current_hidden_state = 0
        observation = self.generate_symptom_scores(self.current_hidden_state)
        return observation

    def render(self, mode='human'):
        print(f"Patient's observed symptoms are: {self.generate_symptom_scores(self.current_hidden_state)}")
    
    def close(self):
        pass

