import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots

from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Imports custom pain-informed Bayes Opt
# from simupain_info_bandit_arms import Arm
from simulation.pain_info_bandit_arms import Arm

# CONSTANTS
TOTAL_ITER = 300
CELL_IDS = [6, 36] # We used in both HW3/HW4; now for analytical consistency.


# Imports the pain informed bandit arm
# NOTE: Use cell_ID 6 or 36 for simulations, both are in cells/SaveState
# Arm1 = Tweak amp, Arm2 = Tweak PW, Arm3 = Tweak Both
arm1 = Arm.remote(cell_ID=6, total_iter=TOTAL_ITER, )
arm1 = Arm.remote(noise_level=0.2, cell_ID=6, non_stationary=False,
                  switch_points=switch_points,total_iterations = total_iterations)
arm2 = 
arms = [arm1, arm2]

# Uncomment the following block to test reward retrieval for each arm at time point 0
iteration = 0
# Retrieve observed rewards (with noise) at iteration 0
observed_rewards = ray.get([
    arm1.get_reward.remote(amp, pulse_width),
    arm2.get_reward.remote(amp, pulse_width)
    # arm3.get_reward.remote(amp, pulse_width),
    # arm4.get_reward.remote(amp, pulse_width)
])

# Retrieve ground truth rewards (noise-free)
GT_rewards = ray.get([
    arm1.get_ground_truth_reward.remote(amp, pulse_width),
    arm2.get_ground_truth_reward.remote(amp, pulse_width)
    # arm3.get_ground_truth_reward.remote(amp, pulse_width),
    # arm4.get_ground_truth_reward.remote(amp, pulse_width)
])

for i in range(4):
    print('Arm %d reward at iteration 0: %.2f (observed), %.2f (ground truth)' % (i+1, observed_rewards[i], GT_rewards[i]))


# Old Thompson Sampling Code
def non_stationary_bandit(total_iterations):

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