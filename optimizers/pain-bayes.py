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

low = -10
high = 10

# Ground truth objective function
def objective(x): 
    return -(np.sin(x) + 0.1 * x ** 2)

# HELPERS
def unnorm(pts, data):
    """ returns a list of points that is originally normalized with respect to array-like data """
    # norm(x) = x-min/(max-min)
    # norm_inv(x) = x(max-min)+min
    mi = min(data)
    ma = max(data)
    return np.array([(p*(ma-mi) + mi) for p in pts])

def my_norm(points, data):
    """ returns a list representative of points but normalized with respect to data """
    mi = min(data)
    ma = max(data)
    return [(x-mi)/(ma-mi) for x in points]

# vis helper
def plot_BO(x, gt, means, pts, it):
    df = pd.DataFrame([[x[i], means[i], gt[i]] for i in range(len(means))], columns=["X", "GP Predicted", "Ground-Truth"])
    fig = px.line(df, x="X", y=["GP Predicted", "Ground-Truth"], title=f"Bayes Opt's Prediction vs. Reality for Iteration {it}")
    fig.add_trace(px.scatter(x=pts.keys(), y=pts.values()).data[0])
    fig.show()

def sample_initial_points(num_init=10):
    """ Randomly sample 10 initial data points in the range of [-10, 10] """
    diff = high-low
    return np.random.rand(num_init)*diff - high

def acquisition_function(means, std, best, explore=0.01):
    """ Expected Improvement (EI) acquisition function """
    improv = [(m-best-explore) for m in means]  # elem wise subtract & divide
    z = improv / std  
    ei = improv * norm.cdf(z) * norm.pdf(z)
    best_ratio = np.argmax(ei)/(len(ei)-1)  # location-ratio of best new val
    new_best = best_ratio*(high-low) + low  # converting ratio to point on our scale
    return new_best

def BO_loop(opt_trials=10, add_graph=True, verbose=True):
    # Define the GP model
    kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3))
    gp = GaussianProcessRegressor(kernel=kernel, optimizer=None) 

    starting_pts = sample_initial_points()
    eval_pts = dict([[pt, objective(pt)] for pt in starting_pts])
    acc_hist = []
    last_gt = []
    for it in range(opt_trials+1):
        # calculate latent
        y_pred = [] # my_norm(eval_pts.values())
        x_vals = []
        for key, val in eval_pts.items():
            y_pred = y_pred + my_norm([val], eval_pts.values())
            x_vals.append(key)
        gp.fit(np.array(x_vals).reshape(-1,1), y_pred)
        # define latent further
        x = np.arange(low, high, 1/10)
        m, s = gp.predict(x.reshape(-1, 1), return_std=True)
        means = unnorm(m, eval_pts.values())
        std = unnorm(s, eval_pts.values())
        # evaluation
        gt = [objective(pt) for pt in x]
        last_gt = [[x[i], gt[i]] for i in range(len(gt))]
        dev_perc = 100*abs(max(means)-max(gt))/(max(means)-min(means))  # %range of deviation from max's
        acc = [round(mean_squared_error(gt, means), 3), round(dev_perc, 3)]
        acc_hist.append(acc)
        if verbose:
            print("It", it, "-", len(eval_pts), "points with [MSE, deviation_perc] =", acc)
        if add_graph:
            plot_BO(x, gt, means, eval_pts, it)  # show the resulting graph
        # set up next iteration
        if it < opt_trials:
            # max([[k,v] for k, v in eval_pts.items()], key=lambda x: x[1])
            bp = max(list(eval_pts.values()))
            next_point = acquisition_function(means, std, bp)
            eval_pts[next_point] = objective(next_point)
            print("Adding point x =", next_point)
    return last_gt, acc_hist, eval_pts, gp
hist, accuracies, _, _ = BO_loop()