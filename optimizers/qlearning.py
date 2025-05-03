"""
TODO: Finish Implementation
"""
import numpy as np
import random
from pointElec_simulation import pointElec_simulation # Used for point electrode simulation from NEURON moodule
from helper import firing_rate # Helper function for calculating the firing rate.

# Amplitude Range: 0 ~ 15 (0.3 steps == 50)
# Pulse Width: 0 ~ 980 (20 steps == 49)

# CONSTANTS
MAX_AMP = 15
MAX_PW = 980
AMP_STEPS = 1.5 # Adjusts by 1.5 nA per action
PW_STEPS = 90   # Adjusts by a pulse width of 20 us (micro-sec) per action

class NeuronEnv:
    def __init__(self, amplitude_range, pulse_width_range, num_amplitudes, num_pulse, cell):
        self.amplitude_values = np.linspace(amplitude_range[0], amplitude_range[1], num_amplitudes)
        self.pulse_width_values = np.linspace(pulse_width_range[0], pulse_width_range[1], num_pulse)
        self.num_amplitudes = num_amplitudes
        self.num_pulse = num_pulse
        self.state = None
        self.cell_ID = cell
        print("Amplitude Values: ", self.amplitude_values)
        print("Pulse Width Values: ", self.pulse_width_values)
        print("Number of Amplitude: ", self.num_amplitudes)
        print("Number of Pulses: ", self.num_pulse)


    def reset(self):
        # Reset to a central starting state
        amp_idx = self.num_amplitudes // 2
        pulse_idx = self.num_pulse // 2
        self.state = (self.amplitude_values[amp_idx], self.pulse_width_values[pulse_idx])
        print("Amplitude Index: ", amp_idx)
        print("PW Index", pulse_idx)
        print("State:", self.state)
        return self.state

    def step(self, action):
        # Action is defined as a tuple: (delta_amplitude_index, delta_pulse_index)
        print("Action: ", action)
        amp_idx, pulse_idx = self.state
        delta_amp, delta_pulse = action
        new_amp_idx = np.clip(amp_idx + delta_amp, 0, self.num_amplitudes - 1)
        new_pulse_idx = np.clip(pulse_idx + delta_pulse, 0, self.num_pulse - 1)
        self.state = (new_amp_idx, new_pulse_idx)
        print("New State: ", self.state)
        
        # Retrieve the actual amplitude and pulse width values from indices
        amplitude = self.amplitude_values[new_amp_idx]
        pulse_width = self.pulse_width_values[new_pulse_idx]
        
        # The reward is defined by the neuron firing rate from your simulation
        reward = simulate_neuron(amplitude, pulse_width)
        return self.state, reward

def simulate_neuron(amplitude, pulse_width):
    """
    Dummy function simulating the neuron model.
    In your implementation, this should call your NEURON simulation,
    update the neuron parameters, run the simulation, and return the firing rate.
    For this example, we simulate an optimum around amplitude=5 and pulse_width=2.5.
    """
    optimal_amplitude = 5.0
    optimal_pulse = 2.5
    firing_rate = np.exp(-((amplitude - optimal_amplitude)**2 + (pulse_width - optimal_pulse)**2))
    # Scale firing rate for demonstration purposes
    return firing_rate * 100

# def simulate_neuron(amp, pw, ):
#     mem_potential,t = pointElec_simulation(num_electrode=1, amplitude=amp, pulse_width=pw, period=100, total_time = 100, cell_type=self.cell_ID, plot_neuron_with_electrodes = False)
#             FR = firing_rate(mem_potential[0], t)
#             self.FR = FR

#     pass

# Define simulation parameters: ranges and discretization details
amplitude_range = (0, MAX_AMP)
pulse_width_range = (0, MAX_PW)
num_amplitudes = int((MAX_AMP/AMP_STEPS) + 1)  # e.g., 0, 0.3, ..., 15
num_pulse = int((MAX_PW/PW_STEPS) + 1)       # e.g., 0, 20, ..., 980

# Define actions: list of changes to the current state.
# Each action is a simple adjustment: increase or decrease amplitude or pulse width, or remain unchanged.
# (Amplitude Change, Pulse Width Change)
actions = [(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)]

# Q-Learning hyperparameters
alpha = 0.1            # Learning rate
gamma = 0.95           # Discount factor
epsilon = 1.0          # Exploration rate (epsilon-greedy)
epsilon_decay = 0.995  # Decay factor per episode
epsilon_min = 0.01     # Minimum exploration rate
num_episodes = 500     # Total number of episodes for training
MAX_STEPS = 50         # Maximum steps per episode


env = NeuronEnv(amplitude_range, pulse_width_range, num_amplitudes, num_pulse, None)

# Initialize Q-table with dimensions: 
# [num amplitude values x num pulse width values x num actions]
Q = np.zeros((num_amplitudes, num_pulse, len(actions)))

# Q-Learning training loop
for episode in range(num_episodes):
    state = env.reset()
    
    for step in range(MAX_STEPS):
        a_idx, p_idx = state
        
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action_index = random.randint(0, len(actions) - 1)
        else:
            action_index = np.argmax(Q[a_idx, p_idx, :])
            
        action = actions[action_index]
        new_state, reward = env.step(action)
        new_a_idx, new_p_idx = new_state
        
        # Q-learning update rule
        best_next_q = np.max(Q[new_a_idx, new_p_idx, :])
        Q[a_idx, p_idx, action_index] = Q[a_idx, p_idx, action_index] + alpha * (
            reward + gamma * best_next_q - Q[a_idx, p_idx, action_index]
        )
        
        state = new_state
        
    # Decay the exploration rate
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    if (episode + 1) % 50 == 0:
        print(f"Episode {episode + 1}/{num_episodes}, Epsilon: {epsilon:.3f}")

print("Training complete.")

# After training, we can extract the best parameters by looking for the state with the highest Q-value.
best_value = -np.inf
best_state = None
for i in range(num_amplitudes):
    for j in range(num_pulse):
        max_q_val = np.max(Q[i, j, :])
        if max_q_val > best_value:
            best_value = max_q_val
            best_state = (i, j)

optimal_amplitude = env.amplitude_values[best_state[0]]
optimal_pulse_width = env.pulse_width_values[best_state[1]]
print(f"Optimal amplitude: {optimal_amplitude}, Optimal pulse width: {optimal_pulse_width}")
