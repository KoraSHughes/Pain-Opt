"""
TODO: Finish Implementation
"""

import numpy as np
import pandas as pd

# Import helper functions for use later.
# pointElec_Stimulation is for electrode simulation for NEURON
from pointElec_simulation import pointElec_simulation
from helper import firing_rate # Helper func for calculating firing rate

import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots

from scipy.stats import norm
# from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Imports custom pain-informed Bayes Opt
# import sys
# sys.path.append("..") # Adds the prior dir to the path to import sibling func
# sys.path.append("../simulation") # Adds the simulation dir to the path
from pain_info_bandit_arms import Arm

import time # Used to generate random seeds later.
import ray
# from simulation.helper import sample_switch_points
from enum import Enum
import plotly.graph_objs as go
import plotly.express as px
import time # Used to generate random seeds later.
np.random.seed(420) # Sets the seed for reproducibility

# Imports the pain model
from painModels import instant_pain


# Enum for setting which classifier to use.
class DISTR(Enum):
    EXP = 1
    BETA = 2

# Initialzes ray for multiprocessing
ray.init()

# CONSTANTS
MAX_AMP = 15
MAX_PW = 980
TOTAL_ITER = 300
EXPR_ITER = 10
MAX_PAIN = 10
ACCEPT_PAIN = 5 # Acceptable pain is anything less than or eq to this val
PLOT_COLS = ["immed_reward", "run_avg_ot", "immed_regret", "accume_regret_ot", "arm_chose"]
# CONSTANT for Exp Distr. Pain Reward Model, we test with these later.
GAMMAS = [0.5, 1.0, 2.0, 5.0]
PYR_CELL_IDs = [6, 8]
PV_CELL_IDS = [32, 36]

# Imports the pain informed bandit arm
# NOTE: Use cell_ID 6 or 36 for simulations, both are in cells/SaveState
arm1 = Arm.remote(cell_ID=PYR_CELL_IDs[0], total_iter=TOTAL_ITER, gamma=GAMMAS[1])
arm2 = Arm.remote(cell_ID=PYR_CELL_IDs[0], total_iter=TOTAL_ITER, gamma=GAMMAS[1])
arms = [arm1, arm2]

# Uncomment the following block to test reward retrieval for each arm at time point 0
iteration = 0
amp = 10
pulse_width = 98
painGamma = np.random.choice(GAMMAS)

actionList = []  # [amplitude, pulse_width]
rewardList = []  # reward

# Retrieve observed rewards (with noise) at iteration 0
observed_rewards = ray.get([
    arm1.get_reward.remote(amp, pulse_width, DISTR.EXP),
    arm2.get_reward.remote(amp, pulse_width, DISTR.EXP)
])

# Retrieve ground truth rewards (noise-free)
GT_rewards = ray.get([
    arm1.get_ground_truth_reward.remote(amp, pulse_width),
    arm2.get_ground_truth_reward.remote(amp, pulse_width)
])

for i in range(2):
    print('Arm %d reward at iteration 0: %.2f (observed), %.2f (ground truth)' % (i+1, observed_rewards[i], GT_rewards[i]))



def apply_distr_func(norm_Pain, distr, FR, gamma, loPain, hiPain):
    # Applies the distribution function that corresponds to a desired freq
    pained_Reward = 0
    if(distr == DISTR.EXP):
        pained_Reward = FR * np.random.exponential(gamma * norm_Pain)

    elif(distr == DISTR.BETA):
        # Describes anything over 5 as a high pain score, setting the standard
        # Low Pain = alpha and High Pain = Beta
        if(norm_Pain >= 5):
            hiPain += 1
        else:
            loPain += 1
        
        pained_Reward = FR  + (FR*np.random.beta(loPain, hiPain))

    return pained_Reward

def get_reward(amp, pw, distr):
    mem_potential, t = pointElec_simulation(num_electrode=1, amplitude=amp, pulse_width=pw, period=100, total_time = 100, cell_type=PYR_CELL_IDs[0], plot_neuron_with_electrodes = False)
    FR = firing_rate(mem_potential[0], t)
    print("The Firing Rate:", FR)
    subj_Pain = instant_pain([amp, pw])
    # The following implements the reward based on a Beta distribution
    norm_Pain = subj_Pain/MAX_PAIN
    
    # The following implements the reward based on a exponential distr.
    pain_Inf_Reward = apply_distr_func(norm_Pain, distr)

    return pain_Inf_Reward

# Model: Bayesian linear regression
model = BayesianRidge()

# Samples 10-points for the reward function for a Bayesian optimizer
for _ in range(EXPR_ITER):
    amp = np.random.randint(0, MAX_AMP)
    pw = np.random.randint(0, MAX_PW)

    print("Amp:", amp, "PW:", pw)

    reward = get_reward(amp, pw, DISTR.EXP)
    actionList.append([amp, pw])
    rewardList.append(reward)

# Old Thompson Sampling Code
def non_stationary_bandit(total_iterations):

    # Fits the model to the explored values 
    model.fit(actionList, rewardList)

    # Preallocates list for speed.
    immediate_Reward = [None] * total_iterations
    run_Avg_Reward = [None] * total_iterations
    immediate_Regret = [None] * total_iterations
    accume_Regret = [None] * total_iterations
    chosen_Arm = [None] * total_iterations
    arm_Rewards = [0] * len(arms)
    arm_Count = [0] * len(arms)
    # arm_Means = [0] * len(arms)
    # arm_Variance = [1] * len(arms)

    # history = [[]]*len(arms)
    alpha = [1]*len(arms) # Alpha represents the number of successful rewards
    beta = [1]*len(arms) # Beta tracks number of failures

    for t in range(total_iterations):

        # Useful for calculating the other parameters, they are constant for this.
        ground_Truths = [ray.get(each_Arm.get_ground_truth_reward.remote(amp, pulse_width)) for each_Arm in arms]
        
        # Please include the 2 lines below 
        # Updates the reward for each arm before sampling begins.
        for arm in arms:
            arm.update_reward.remote(t) # Update the reward of the arm at time t
        
        # Uses Thompson sampling by sampling based on a beta distribution and turning this into
        # a bernoulli bandit problem.
        sampled_means = [np.random.beta(alpha[i], beta[i]) for i in range(len(arms))]

        # Following used for debugging rewards
        # This doesn't work the way I expect
        # potential_Rewards = [ray.get(arm.get_reward.remote(amp, pulse_width)) for arm in arms]
        # print("(" + str(t) + ")", "Potential Rewards:", potential_Rewards, "| Ground Truths:", ground_Truths)
        
        #print("(" + str(t) + ")", "Sample Mean:", sampled_means, "|", arm_Rewards)
        arm_Idx = np.argmax(sampled_means)

        # Gets the reward, updates the reward tracking and number of times the arm was pulled.
        curr_Reward = ray.get(arms[arm_Idx].get_reward.remote(amp, pulse_width))
        arm_Rewards[arm_Idx] = ((arm_Rewards[arm_Idx] * arm_Count[arm_Idx]) + curr_Reward)/(arm_Count[arm_Idx] + 1)
        arm_Count[arm_Idx] += 1

        
        # if(len(history[arm_Idx]) > SLIDE_WIND):
        #     history[arm_Idx].pop()
        
        # # Adds the current reward to the history
        # history[arm_Idx].append(curr_Reward)

        # Increments alpha and beta based on the successful arm pull
        if(arm_Idx == np.argmax(arm_Rewards)):
            alpha[arm_Idx] += 1
        else:
            beta[arm_Idx] += 1


        #arm_Variance[arm_Idx] = arm_Variance[arm_Idx]/arm_Count[arm_Idx]

        # Updates the posterior mean and variance for the selected arm
        # and updates the new mean and new variance for the selected arm
        # alpha = arm_Count[arm_Idx]
        # beta = (arm_Rewards[arm_Idx] * arm_Count[arm_Idx])
        # new_mean = beta / alpha
        # arm_Means[arm_Idx] = new_mean
        # arm_Variance[arm_Idx] = arm_Variance[arm_Idx] / (alpha)

        # Step 5: Update the posterior (mean and variance) for the selected arm
        # alpha = arm_Count[arm_Idx]
        # Assuming you are using a Gaussian model, the posterior update formula would be like this:
        # arm_Means[arm_Idx] = (arm_Rewards[arm_Idx] * (alpha - 1) + curr_Reward) / alpha
        # arm_Variance[arm_Idx] = arm_Variance[arm_Idx] / alpha  # This is just a simple variance update

        # history.append([arm_Idx, curr_Reward])
        
        # Maintain sliding window size by removing the oldest entry when necessary
        # if t % SLIDE_WIND == 0 > SLIDE_WIND:
        #     history.pop(0)

        # Update model parameters defined to be returned by the function
        immediate_Reward[t] = curr_Reward
        if(t > 1):
            run_Avg_Reward[t] = (run_Avg_Reward[t - 1]*(t) + immediate_Reward[t])/(t + 1)
        else:
            run_Avg_Reward[t] = immediate_Reward[t]

        # Pairs the selected arm alongside the best ground truth arm at the current time step
        chosen_Arm[t] = [arm_Idx, np.argmax(ground_Truths)]

        # Regret is given as (best ground truth reward - selected arm reward)
        immediate_Regret[t] = ground_Truths[np.argmax(ground_Truths)] - curr_Reward
        
        # Stores accumulated regret over time 
        if(t > 1):
            accume_Regret[t] = (accume_Regret[t - 1] + immediate_Regret[t])
        else:
            accume_Regret[t] = immediate_Regret[t]

        # Debug/Performance String
        #print("Step (" + str(t) + "):", arm_Rewards, "|", ground_Truths)
    
    return immediate_Reward, run_Avg_Reward, immediate_Regret, accume_Regret, chosen_Arm

# # Thompson Sampling Implementation
# np.random.seed(int(time.time())) # Sets seed based on time for broad range seed testing
# outptTS = non_stationary_bandit(TOTAL_ITER)
# outptTS = pd.DataFrame(outptTS).T
# outptTS.columns = PLOT_COLS
# outptTS['function'] = ["TS"] * TOTAL_ITER
# outptTS['time_step'] = time_step
# outptTS['ideal_arm'] = np.array(outptTS['arm_chose'].tolist())[:, 1]
# outptTS['arm_chose'] = np.array(outptTS['arm_chose'].tolist())[:, 0]
# # exp_Data = pd.concat([exp_Data, ucbData], ignore_index=True)


# immed_reward_fig = px.scatter(outptTS, x="time_step", y="immed_reward", 
#                               title="Immmediate Reward over Time", color="function",
#                               labels={"time_step": "Time Steps", "immed_reward": "Immediate Reward"},
#                               trendline="ols")
# immed_reward_fig.show()

# run_avg_ot_fig = px.line(outptTS, x="time_step", y="run_avg_ot", 
#                          title="Running Avg Reward Over Time", color="function",
#                          labels={"time_step": "Time Steps", "run_avg_ot": "Running Avg"})
# run_avg_ot_fig.show()

# immed_regret_fig = px.scatter(outptTS, x="time_step", y="immed_regret", 
#                          title="Immediate Regret Over Time", color="function",
#                          labels={"time_step": "Time Steps", "immed_regret": "Immediate Regret"},
#                          trendline="ols")
# immed_regret_fig.show()

# accume_regret_ot_fig = px.line(outptTS, x="time_step", y="accume_regret_ot", 
#                          title="Accumulated Regret", color="function",
#                          labels={"time_step": "Time Steps", "accume_regret_ot": "Accumulated Regret"})
# accume_regret_ot_fig.show()

# arm_chose_fig = px.scatter(outptTS, x="time_step", y=['arm_chose'], 
#                          title="Arm Chosen Over Time", color="function",
#                          labels={"time_step": "Time Steps", "arm_chose": "Selected Arm"})
# arm_chose_fig.show()

# ideal_arm_fig = px.scatter(outptTS, x="time_step", y=["ideal_arm"], 
#                          title="Ideal Arm", color="function",
#                          labels={"time_step": "Time Steps", "ideal_arm": "Ideal Arm"})
# ideal_arm_fig.show()
ray.shutdown()