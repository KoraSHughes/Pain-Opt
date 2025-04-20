from matplotlib import pyplot as plt
from pointElec_simulation import pointElec_simulation # Used for point electrode simulation from NEURON moodule
from helper import firing_rate # Helper function for calculating the firing rate.
import numpy as np 
import random 
import ray # Used for multi-processing of NEURNO module

from matplotlib import pyplot as plt
from pointElec_simulation import pointElec_simulation
from helper import firing_rate
import numpy as np
import random 
import ray

@ray.remote
class Arm:
    def __init__(self, cell_ID, total_iter):
        self.m_cell_ID = cell_ID
        self.m_total_Iter = total_iter # Total number of iterations we want to execute for.
        self.m_FR = -1 # Initialized to -1 as no firing rate has been determined.
        self.m_ground_truth_reward = -1 # Initialized to -1 as no ground truth reward is determined yet.
        self.m_baseFR = 8 # Clarify how much this value can change later. Can we make this arbirtrary?
    
    def update_reward(self, t):
        if t != 0 and self.non_stationary and t in self.switch_points and self.switch_flag[t] == 0:
            #np.random.seed(self.cell_ID * t)
            self.FR += np.random.uniform(-0.4 * self.FR, 0.4 * self.FR)  
            self.FR = self.FR * 0.9 + self.FR_base * 0.1  
            self.switch_flag[t] = 1

        
    def get_reward(self, stim_amp, stim_pulse_width):
        # Initially, no firing rate is defined for the targeted neuron, so we calculate the FR.
        if self.FR == -1:
            mem_potential, t = pointElec_simulation(num_electrode=1, amplitude=stim_amp, pulse_width=stim_pulse_width, period=100, total_time = 100, cell_type=self.m_cell_ID, plot_neuron_with_electrodes = False)

    def get_ground_truth_reward(self,stim_amp, stim_pulse_width):
        if self.m_FR == -1: # Represents an undefined firing rate
            mem_potential,t = pointElec_simulation(num_electrode=1, amplitude=stim_amp, pulse_width=stim_pulse_width, period=100, total_time = 100, cell_type=self.m_cell_ID, plot_neuron_with_electrodes = False)
            FR = firing_rate(mem_potential[0], t)
            self.FR = FR       
        return self.apply_transfer_function(self.FR)
    
    # TODO: Update Transfer Function to match representation for neuron type and input currents.
    def apply_transfer_function(self,FR):
        if self.cell_ID == 35:
            return FR * 1
        elif self.cell_ID == 6:
            return FR * 0.5
        elif self.cell_ID == 7:
            return np.log(FR) + FR -1
        elif self.cell_ID == 36:
            return np.sqrt(FR)
        else:
            raise ValueError('Transfer function not recognized')
            

@ray.remote
class Arm:
    def __init__(self, noise_level,cell_ID,non_stationary,switch_points,total_iterations):       
        self.noise_level = noise_level
        self.cell_ID = cell_ID
        self.non_stationary = non_stationary
        self.total_iterations = total_iterations
        self.FR = -1
        self.ground_truth_reward = -2
        self.switch_points = switch_points
        self.switch_flag = np.zeros(total_iterations)
        self.FR_base = 8

    def update_reward(self,t):
        if t != 0 and self.non_stationary and t in self.switch_points and self.switch_flag[t] == 0:
            #np.random.seed(self.cell_ID * t)
            self.FR += np.random.uniform(-0.4 * self.FR, 0.4 * self.FR)  
            self.FR = self.FR * 0.9 + self.FR_base * 0.1  
            self.switch_flag[t] = 1

    def get_reward(self,stim_amp, stim_pulse_width):
        # simulation of firing rate, only calculate once. 
        if self.FR == -1:
            mem_potential,t = pointElec_simulation(num_electrode=1, amplitude=stim_amp, pulse_width=stim_pulse_width, period=100, total_time = 100, cell_type=self.cell_ID, plot_neuron_with_electrodes = False)
            FR = firing_rate(mem_potential[0], t)
            self.FR = FR
        # apply transfer function
        FR_deep = self.apply_transfer_function(self.FR)
        self.ground_truth_reward = FR_deep
        # add noise 
        noisy_FR_deep = FR_deep + FR_deep*np.random.normal(0,self.noise_level)
        return noisy_FR_deep
    
    def get_ground_truth_reward(self,stim_amp, stim_pulse_width):
        if self.FR == -1:
            mem_potential,t = pointElec_simulation(num_electrode=1, amplitude=stim_amp, pulse_width=stim_pulse_width, period=100, total_time = 100, cell_type=self.cell_ID, plot_neuron_with_electrodes = False)
            FR = firing_rate(mem_potential[0], t)
            self.FR = FR       
        return self.apply_transfer_function(self.FR)
    
    def apply_transfer_function(self,FR):
        if self.cell_ID == 35:
            return FR * 1
        elif self.cell_ID == 6:
            return FR * 0.5
        elif self.cell_ID == 7:
            return np.log(FR) + FR -1
        elif self.cell_ID == 36:
            return np.sqrt(FR)
        else:
            raise ValueError('Transfer function not recognized')