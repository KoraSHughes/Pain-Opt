"""
NOTE: This is an outdated implementation of the pain-informed bandit. Do NOT reference.
"""
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
    def __init__(self, cell_ID, total_iter, tweak_PW, init_amp, init_PW):
        self.m_cell_ID = cell_ID
        self.m_total_Iter = total_iter # Total iterations to train for
        self.m_FR = -1 # No firing rate determined holds value of -1
        self.m_ground_truth_reward = -1 # No ground truth holds value of -1
        self.m_baseFR = 8 # Clarify how much this value can change later. Can we make this arbirtrary?
        self.m_tweak_PW = False # Tweak amplitude (False) | Tweak pulse width (True)

        # Stores the historical values for the hyperparmeters being tuned
        # They are pre-allocated to improve processing speed.
        self.m_Amps = [None] * total_iter
        self.m_PWs = [None] * total_iter
        self.m_Amps[0] = init_amp
        self.m_PWs[0] = init_PW
    
    def update_reward(self, t):
        if t != 0 and self.non_stationary and t in self.switch_points and self.switch_flag[t] == 0:
            #np.random.seed(self.cell_ID * t)
            self.FR += np.random.uniform(-0.4 * self.FR, 0.4 * self.FR)  
            self.FR = self.FR * 0.9 + self.FR_base * 0.1  
            self.switch_flag[t] = 1

        
    def get_reward(self, tweak_PW, stim_Amp, stim_PW):
        """
        Inputs: tweak_PW - Bool for choosing to adjust pulse width or amplitude
                stim_Amp - The stimulating amplitude that we will test for
                stim_PW - The simulating pules width that we will test for
        Outputs: pained_FR - Appends a pain factor to the firing rate of the Neuron
        """
        # Initially, no firing rate is defined for the targeted neuron, so we calculate the FR.
        if self.m_FR == -1:
            # Should tweak_PW be given as a None value, then we change both hyperparameters
            if tweak_PW == None:
                mem_potential, t = pointElec_simulation(num_electrode=1, amplitude=stim_Amp, pulse_width=stim_PW, period=100, total_time = 100, cell_type=self.m_cell_ID, plot_neuron_with_electrodes = False)
            elif tweak_PW == True:
                mem_potential, t = pointElec_simulation(num_electrode=1, amplitude=self.m_Amps[0], pulse_width=stim_PW, period=100, total_time = 100, cell_type=self.m_cell_ID, plot_neuron_with_electrodes = False)
            else:
                mem_potential, t = pointElec_simulation(num_electrode=1, amplitude=stim_Amp, pulse_width=self.m_PWs[0], period=100, total_time = 100, cell_type=self.m_cell_ID, plot_neuron_with_electrodes = False)

        FR_deep = self.apply_transfer_function(self.m_FR)
        self.m_ground_truth_reward = FR_deep

        # Adds the pain metric here
        """
            The following are different pain distributions that we execute on:
            (1) Uses a normal distribution with mean(μ) = 5 & std(σ²) = 2 as
                the behavior strongly mimics our theorized pain behavior.
                This is NOT ideal as this should not be a metric confounded 
                with the determination of 

            (2) Usees
        """
        # pain_Induced_FR = FR_deep  + FR_deep*np.random
        return FR_deep

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
