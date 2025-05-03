from matplotlib import pyplot as plt
from pointElec_simulation import pointElec_simulation # Used for point electrode simulation from NEURON moodule
from helper import firing_rate # Helper function for calculating the firing rate.
import numpy as np 
import random 
import ray # Used for multi-processing of NEURNO module
from enum import Enum

# Imports the pain response from the participant using the painModel
from painModels import instant_pain
# CONSTANTS
MAX_PAIN = 10
ACCEPT_PAIN = 5 # Acceptable pain is anything less than or eq to this val

# Enum for setting which classifier to use.
class DISTR(Enum):
    EXP = 1
    BETA = 2

GAMMAS = [0.5, 1.0, 2.0, 5.0]

@ray.remote
class Arm:
    def __init__(self, cell_ID, total_iter, gamma):
        self.m_cell_ID = cell_ID
        self.m_total_Iter = total_iter # Total iterations to train for
        self.m_FR = -1 # No firing rate determined holds value of -1
        self.m_ground_truth_reward = -1 # No ground truth holds value of -1
        self.m_baseFR = 8 # Clarify how much this value can change later. Can we make this arbirtrary?
        
        # Pain informed variabels
        # The following variable(s) are used for a Beta Dist. pain model
        self.m_hiPain = 1 # This represents the beta value
        self.m_loPain = 1 # This represents the alpha value

        # The following variable(s) are used for the Exp Distr. pain model
        self.m_gamma = 1.5 # Default to 1.5

        # Stores the historical values for the hyperparmeters being tuned
        # They are pre-allocated to improve processing speed.
        self.m_Amp_Arr = []
        self.m_PW_Arr = []
        self.m_FR_Arr = []
    
    def update_reward(self, t, stim_Amp, stim_PW, distr):
        """
        Inputs: stim_Amp - The stimulating amplitude that we will test for
                stim_PW - The simulating pules width that we will test for
                distr - Specifies which distribution to include for pain scale
        Result: pained_FR - Appends a pain factor to the firing rate of the Neuron
        """


        # if t != 0 and self.non_stationary and t in self.switch_points and self.switch_flag[t] == 0:
        #     #np.random.seed(self.cell_ID * t)
        #     self.FR += np.random.uniform(-0.4 * self.FR, 0.4 * self.FR)  
        #     self.FR = self.FR * 0.9 + self.FR_base * 0.1  
        #     self.switch_flag[t] = 1
        pass

        
    def get_reward(self, stim_Amp, stim_PW, distr):
        """
        Inputs: stim_Amp - The stimulating amplitude that we will test for
                stim_PW - The simulating pules width that we will test for
                distr - Specifies which distribution to include for pain scale
        Result: pained_FR - Appends a pain factor to the firing rate of the Neuron
        """

        # Initially, no firing rate is defined for the targeted neuron, so we calculate the FR.
        if self.m_FR == -1:
            # Calculates the firing rate from the membrane potential value
            mem_potential, t = pointElec_simulation(num_electrode=1, amplitude=stim_Amp, pulse_width=stim_PW, period=100, total_time = 100, cell_type=self.m_cell_ID, plot_neuron_with_electrodes = False)
            self.m_FR = firing_rate(mem_potential[0], t)
        
        print("FR:", self.m_FR)
        # # Calculates the firing rate from the membrane potential value
        # mem_potential, t = pointElec_simulation(num_electrode=1, amplitude=stim_Amp, pulse_width=stim_PW, period=100, total_time = 100, cell_type=self.m_cell_ID, plot_neuron_with_electrodes = False)
        # self.m_FR = firing_rate(mem_potential[0], t)

        # Updates the tested electrode parameters for the arm
        self.m_Amp_Arr.append(stim_Amp)
        self.m_PW_Arr.append(stim_PW)
        self.m_FR_Arr.append(self.m_FR)

        # FR_deep = self.apply_transfer_function(self.m_FR) # May not be needed
        # self.m_ground_truth_reward = self.m_FR

        # Adds the pain-informed metric here
        """
            The following are different pain distributions that we execute on:
            (1) Uses a normal distribution with mean(μ) = 5 & std(σ²) = 2 as
                the behavior strongly mimics our theorized pain behavior.
                This is NOT ideal as this should not be a metric confounded 
                with the determination of 

            (2) Uses the pain model defined by Kora here to determine a pain
                score, then use a beta distribution to set high vs low pain
            
            (3) It could also be valuable to study what this paper does which
                it refers to as a Bayesian Jump Model(?)
                Link: https://pmc.ncbi.nlm.nih.gov/articles/PMC9633765/
                G-Scholar link: https://scholar.google.com/scholar?q=%22Bayesian+Jump+Frequency+model%22&hl=en&as_sdt=0&as_vis=1&oi=scholart
        """
        subj_Pain = instant_pain([stim_Amp, stim_PW])
        # The following implements the reward based on a Beta distribution
        norm_Pain = subj_Pain/MAX_PAIN

        print("Subjective Pain:", subj_Pain)
        print("Normalized Pain:", norm_Pain)
        
        # The following implements the reward based on a exponential distr.
        pain_Inf_Reward = self.apply_distr_func(norm_Pain, distr)

        print("Pain Informed Reward:", pain_Inf_Reward)
        return pain_Inf_Reward

    def get_ground_truth_reward(self, stim_amp, stim_pulse_width):
        if self.m_FR == -1: # Represents an undefined firing rate
            mem_potential,t = pointElec_simulation(num_electrode=1, amplitude=stim_amp, pulse_width=stim_pulse_width, period=100, total_time = 100, cell_type=self.m_cell_ID, plot_neuron_with_electrodes = False)
            FR = firing_rate(mem_potential[0], t)
            self.m_FR = FR       
        return self.m_FR
    
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
        
    def apply_distr_func(self, norm_Pain, distr):
        """
        Input(s):   norm_Pain - The normalized pain value between 0 and 1
                    distr - Specifies which distr for disincentivizing pain
        Output(s):  pained_Reward - Scales the firing rate according to the pain
        """
        # Applies the distribution function that corresponds to a desired freq
        pained_Reward = 0
        if(distr == DISTR.EXP):
            pained_Reward = self.m_FR + (self.m_FR * np.random.exponential(self.m_gamma * norm_Pain))

        elif(distr == DISTR.BETA):
            # Describes anything over 5 as a high pain score, setting the standard
            # Low Pain = alpha and High Pain = Beta
            if(norm_Pain >= 5):
                self.m_hiPain += 1
            else:
                self.m_loPain += 1
            
            pained_Reward = self.m_FR  + (self.m_FR*np.random.beta(self.m_loPain, self.m_hiPain))

        return pained_Reward
            
# @ray.remote
# class Arm:
#     def __init__(self, noise_level,cell_ID,non_stationary,switch_points,total_iterations):       
#         self.noise_level = noise_level
#         self.cell_ID = cell_ID
#         self.non_stationary = non_stationary
#         self.total_iterations = total_iterations
#         self.FR = -1
#         self.ground_truth_reward = -2
#         self.switch_points = switch_points
#         self.switch_flag = np.zeros(total_iterations)
#         self.FR_base = 8

#     def update_reward(self,t):
#         if t != 0 and self.non_stationary and t in self.switch_points and self.switch_flag[t] == 0:
#             #np.random.seed(self.cell_ID * t)
#             self.FR += np.random.uniform(-0.4 * self.FR, 0.4 * self.FR)  
#             self.FR = self.FR * 0.9 + self.FR_base * 0.1  
#             self.switch_flag[t] = 1

#     def get_reward(self,stim_amp, stim_pulse_width):
#         # simulation of firing rate, only calculate once. 
#         if self.FR == -1:
#             mem_potential,t = pointElec_simulation(num_electrode=1, amplitude=stim_amp, pulse_width=stim_pulse_width, period=100, total_time = 100, cell_type=self.cell_ID, plot_neuron_with_electrodes = False)
#             FR = firing_rate(mem_potential[0], t)
#             self.FR = FR
#         # apply transfer function
#         FR_deep = self.apply_transfer_function(self.FR)
#         self.ground_truth_reward = FR_deep
#         # add noise 
#         noisy_FR_deep = FR_deep + FR_deep*np.random.normal(0,self.noise_level)
#         return noisy_FR_deep
    
#     def get_ground_truth_reward(self, stim_amp, stim_pulse_width):
#         if self.FR == -1:
#             mem_potential,t = pointElec_simulation(num_electrode=1, amplitude=stim_amp, pulse_width=stim_pulse_width, period=100, total_time = 100, cell_type=self.cell_ID, plot_neuron_with_electrodes = False)
#             FR = firing_rate(mem_potential[0], t)
#             self.FR = FR       
#         return self.apply_transfer_function(self.FR)
    
#     def apply_transfer_function(self,FR):
#         if self.cell_ID == 35:
#             return FR * 1
#         elif self.cell_ID == 6:
#             return FR * 0.5
#         elif self.cell_ID == 7:
#             return np.log(FR) + FR -1
#         elif self.cell_ID == 36:
#             return np.sqrt(FR)
#         else:
#             raise ValueError('Transfer function not recognized')