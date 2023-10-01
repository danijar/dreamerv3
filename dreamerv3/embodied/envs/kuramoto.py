import embodied
import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt

class KuramotoEnv(embodied.Env):
    DT = 0.01 # Time set for Runge-Kutta method
    def initialize_coupling_matrix(self):
        self.A = np.random.rand(self.N, self.N)
        self.A = (self.A + self.A.T) / 2  # Ensure symmetry
        np.fill_diagonal(self.A, 0)  # No self-coupling 

    def __init__(self, target_matrix, N=64, threshold=0.1, sim_steps_per_env_step=100, max_steps=1000):
        assert target_matrix.shape == (N, N), "Target matrix must be square with dimensions matching N"
        self.N = N  # Number of oscillators
        self.theta = np.random.uniform(0, 2 * np.pi, N)  # Initial phases
        self.initialize_coupling_matrix()
        self.target_matrix = target_matrix  # Target coupling matrix
        self.threshold = threshold  # Termination threshold for Frobenius norm
        self._done = True
        self.sim_steps_per_env_step = sim_steps_per_env_step
        self.phase_history = []  # Initialize phase history
        self.max_steps = max_steps
        self.current_step = 0


        # Define action and observation spaces
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(N, N), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(N, N, 1), dtype=np.float32)

    def simulation_step(self):
        """
        Advances the simulation by one time step using the 4th Order Runge-Kutta method.
        """
        # Define the derivative function for the Runge-Kutta method
        def derivative(theta):
            return np.sum(self.A * np.sin(np.outer(theta, np.ones(self.N)) - np.outer(np.ones(self.N), theta)), axis=1)
        
        # 4th Order Runge-Kutta method to update phases
        dt = self.DT # Time step
        k1 = dt * derivative(self.theta)
        k2 = dt * derivative(self.theta + 0.5 * k1)
        k3 = dt * derivative(self.theta + 0.5 * k2)
        k4 = dt * derivative(self.theta + k3)
        self.theta = self.theta + (k1 + 2*k2 + 2*k3 + k4) / 6
        
        # Store phase data
        self.phase_history.append(self.theta)
    
    def compute_correllogram(self):
        # Convert phase history to a numpy array for easier manipulation
        phase_data = np.array(self.phase_history)
        correllogram = np.corrcoef(phase_data, rowvar=False)  # Compute correlation matrix
        correllogram = np.expand_dims(correllogram, axis=-1)  # Add a channel dimension
        return correllogram
    
    def update_coupling_matrix(self, action):
        self.A += action
        self.A = (self.A + self.A.T) / 2  # Ensure symmetry
        np.fill_diagonal(self.A, 0)  # No self-coupling
        self.A = np.clip(self.A, 0, 1)  # Clip the values of A to be within [0, 1]


    def step(self, action):
        """
        Apply action, advance the simulation, and compute the correllogram, reward, and done flag.
        
        Parameters:
        action (np.ndarray): The action to apply to the coupling matrix.
        
        Returns:
        tuple: A tuple containing the correllogram, reward, done flag, and an empty dictionary.
        """
        
        # Update step count
        self.current_step += 1
        if self.current_step >= self.max_steps: 
            self._done = True

        # Apply action to the coupling matrix
        self.update_coupling_matrix(action)

        self.phase_history = [] # Reset phase history
        
        # Advance the simulation
        for _ in range(self.sim_steps_per_env_step):
            self.simulation_step()

        # Compute correllogram
        correllogram = self.compute_correllogram()
        
        # Compute reward based on closeness to the target matrix
        frobenius_norm = np.linalg.norm(self.A - self.target_matrix, 'fro')
        reward = -frobenius_norm
        
        # Check termination condition
        self._done = frobenius_norm < self.threshold
        
        return correllogram, reward, self._done, {}

    def reset(self):
        self.theta = np.random.uniform(0, 2 * np.pi, self.N)  # Reset phases
        self.initialize_coupling_matrix()
        self._done = False  # Reset done flag

        self.phase_history = []  # Reset phase history
        # Optionally, you might want to run a few simulation steps to get an initial correllogram
        for _ in range(self.sim_steps_per_env_step):
            self.simulation_step()
        initial_correllogram = self.compute_correllogram()  # Compute initial correllogram

        return initial_correllogram

    def render_phases(self, ax):
        ax.scatter(range(self.N), self.theta, c=self.theta, cmap='hsv', alpha=0.75)
        ax.set_title('Phases of Oscillators')
        ax.set_xlabel('Oscillator Index')
        ax.set_ylabel('Phase (radians)')
        ax.set_yticks([0, np.pi, 2 * np.pi])
        ax.set_yticklabels(['0', 'π', '2π'])

    def render_coupling_matrix(self, ax):
        im = ax.imshow(self.A, cmap='viridis', aspect='auto')
        ax.set_title('Coupling Matrix')
        ax.set_xlabel('Oscillator Index')
        ax.set_ylabel('Oscillator Index')
        return im  # Return the image object for colorbar

    def render_correllogram(self, ax):
        correllogram = self.compute_correllogram()
        im = ax.imshow(correllogram, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        ax.set_title('Correllogram')
        ax.set_xlabel('Oscillator Index')
        ax.set_ylabel('Oscillator Index')
        return im  # Return the image object for colorbar

    def render(self):
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        self.render_phases(axs[0])
        im1 = self.render_coupling_matrix(axs[1])
        fig.colorbar(im1, ax=axs[1], orientation='vertical', fraction=0.046, pad=0.04)
        im2 = self.render_correllogram(axs[2])
        fig.colorbar(im2, ax=axs[2], orientation='vertical', fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.show()

    def close(self):
        pass