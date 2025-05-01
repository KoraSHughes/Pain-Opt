import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import DotProduct
import matplotlib.pyplot as plt
from updated_run import run_simulation          # NEURON wrapper
import plotly.express as px
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
from matplotlib import cm  # Colormap
import plotly.graph_objects as go
#import plotly.io as pio
#pio.renderers.default = 'browser'

from painModels import (
    instant_pain, add_relative_pain, calculate_pains
)

np.random.seed(23)
MAX_PAIN = 10
# pain helpers:
# Implementation: 
def ideal_pain_curve(trial_num, max_trial_num):  # helper for ideal pain curve
    """ -(4B/A^2) * x(x-A) :: A is the max#trials, B is the ideal height aka pain,
                                C is roughly where we want to start our experiment """
    x = trial_num
    height_mod = 0.5
    A = max_trial_num
    B = MAX_PAIN*0.6 - height_mod  # * Note: in ideal we only want 90% of our max pain?
    return (-4*B * x * (x-A)) / (A**2) + height_mod

def weight_ppain(params, last_params, trial_num, max_trial_num, max_params, w=[0.25, 0.2], lower_pain_mod=1):
    """ returns a weight to the uncertainty with a max deviating percentage of sum(w) 
    ie, at the highest pain possible, we are disencentivizing sampling by (sum(w)*100)%"""
    # scaling modifiers
    # if sum(w) > 1:
    #     temp = sum(w)
    #     w[0] = w[0]/temp
    #     w[1] = w[1]/temp
    # inst pain
    ipain = instant_pain(params, max_params, [5,3])  # TODO: tune weights
    if len(last_params) == 0:  # first stim event => no baseline pain
        lpain = 0
    else:
        lpain = instant_pain(last_params, max_params, [5,3])
    # * Note: include_lower incentivizes larger decreases in pain-scale which may inadvertantly incentivize higher jumps
    # relative pain
    ppain = add_relative_pain([lpain, ipain], 2, include_lower=False)[-1]
    modifier = w[0] * ppain/MAX_PAIN
    #assert modifier <= w[0]
    # memory of pain
    idpain = ideal_pain_curve(trial_num, max_trial_num)
    if ppain > idpain:
        modifier += w[1] * (abs(ppain - idpain)/MAX_PAIN)
    else:
        modifier += w[1]*lower_pain_mod * (abs(ppain - idpain)/MAX_PAIN)
    #assert modifier <= 1
    return (1-modifier)

def scale_pain(mygrid, last_params, trial, max_trials=20, max_params=[80, 30], w=[0.4, 0.8], lower_pain_mod=1):
    res = []
    for i in range(mygrid.shape[0]):
        params = [mygrid[i, 0], mygrid[i, 1]]  # amp, pw
        res.append(weight_ppain(params, last_params, trial, max_trials, max_params, w, lower_pain_mod))
    return np.array(res)

# ───────────────── helpers ──────────────────────────────────────────
def neuron_objective(X):
    X = np.atleast_2d(X)
    return np.array([run_simulation(float(a), float(pw),
                                    sim_duration=1000,
                                    stim_delay=5,
                                    plot_traces=False)   # keep UI clean
                     for a, pw in X])

def latin_hypercube(n, amp_bounds, pw_bounds):
    amp = np.random.uniform(*amp_bounds, n)
    pw  = np.random.uniform(*pw_bounds , n)
    return np.column_stack((amp, pw))

def expected_improvement(mu, sigma, best):
    z = (mu - best) / (sigma + 1e-9)
    return (mu - best) * norm.cdf(z) + sigma * norm.pdf(z)

def _plot_surface(gp, mu_norm, A, PW, X_samp, step):
    mu_raw = gp._y_train_mean + mu_norm * gp._y_train_std
    Z = mu_raw.reshape(A.shape)

    # Predicted GP mean at sampled points
    mu_samp_norm, _ = gp.predict(X_samp, return_std=True)
    mu_samp_raw = gp._y_train_mean + mu_samp_norm * gp._y_train_std

    fig = go.Figure()

    # Add surface
    fig.add_trace(go.Surface(
        x=A, y=PW, z=Z,
        colorscale='Viridis',
        colorbar=dict(title='GP mean firing-rate (Hz)')
    ))

    # Add sample points
    fig.add_trace(go.Scatter3d(
        x=X_samp[:, 0], y=X_samp[:, 1], z=mu_samp_raw,
        mode='markers',
        marker=dict(size=5, color='red', line=dict(color='black', width=1)),
        name='Samples'
    ))

    fig.update_layout(
        title=f"BO iteration {step}",
        scene=dict(
            xaxis_title='Amplitude',
            yaxis_title='Pulse-width',
            zaxis_title='Firing Rate (Hz)'
        ),
        width=800,
        height=600
    )

    fig.show()

def _plot_surface2(gp, mu_norm, A, PW, X_samp, step):
    """
    Contour-plot the *denormalised* GP mean surface.
    Works with a GP that was trained with `normalize_y=True`.
    """
    mu_raw = gp._y_train_mean + mu_norm * gp._y_train_std
    plt.figure(figsize=(6, 4))
    cs = plt.contourf(A, PW, mu_raw.reshape(A.shape),
                      levels=30, alpha=0.8)
    plt.colorbar(cs, label="GP mean firing-rate (Hz)")
    plt.scatter(X_samp[:, 0], X_samp[:, 1],
                c='red', edgecolor='k', s=55, label='Samples')
    plt.title(f"BO iteration {step}")
    plt.xlabel("Amplitude"); plt.ylabel("Pulse-width")
    plt.tight_layout(); plt.show()

def _plot_history(fr_history):
    plt.figure(figsize=(6, 3))
    plt.plot(fr_history, marker='o')
    plt.xlabel("Iteration"); plt.ylabel("Firing rate (Hz)")
    plt.tight_layout(); plt.show()

# ───────────────── BO loop ──────────────────────────────────────────
def BO_loop(opt_trials=25,
            amp_bounds=(2, 80),
            pw_bounds=(0.01, 30),
            grid=120,
            n_init=3,
            kernel_type = 'RBF',
            make_plots=True,
            w=[0.5,0.5],
            lower_pain_mod = 1):
    # GP definition
    if kernel_type == 'RBF':
        kernel = RBF(length_scale=[2.75, 250], length_scale_bounds=(1e-3, 1e5))
    else:
        kernel = DotProduct(sigma_0=1)

    gp = GaussianProcessRegressor(kernel=kernel,
                                  normalize_y=True,
                                  optimizer=None)

    # prediction grid
    A, PW = np.meshgrid(np.linspace(*amp_bounds, grid),
                        np.linspace(*pw_bounds,  grid), indexing='ij')
    X_grid = np.c_[A.ravel(), PW.ravel()]

    # ── initial Latin-hypercube sample ────────────────────────────
    X = latin_hypercube(n_init, amp_bounds, pw_bounds)
    X = np.array([[1,100],[1.5,150],[0.5,200]])
    #y = neuron_objective(X)
    y = np.array([[4],[6],[4]])
    gp.fit(X, y)

    best_hist = [y.max()]          # record best raw value each iter

    if make_plots:
        mu0, _ = gp.predict(X_grid, return_std=True)
        _plot_surface(gp, mu0, A, PW, X, step=0)

    max_bounds = (amp_bounds[1], pw_bounds[1])
    last_params = []
    # ── optimisation loop ────────────────────────────────────────
    for k in range(1, opt_trials + 1):
        oldmu, sd = gp.predict(X_grid, return_std=True)
        pain_factors = scale_pain(X_grid, last_params, k, opt_trials, max_bounds, w, lower_pain_mod)
        print(k/opt_trials)
        print(last_params)
        mu = np.multiply(pain_factors, oldmu)  # scale means
        ei = expected_improvement(mu, sd, np.max(best_hist))

        x_next = X_grid[np.argmax(ei)]
        last_params = x_next
        y_next = neuron_objective(x_next)[0]

        # update datasets
        X = np.vstack([X, x_next])
        y = np.append(y, y_next)
        best_hist.append(y_next)

        gp.fit(X, y)

        if make_plots:
            mu_k, _ = gp.predict(X_grid, return_std=True)
            pain_factors = scale_pain(X_grid, last_params, k, opt_trials, max_bounds, w=[0.5,0.5])
            pain_mu = np.multiply(pain_factors, mu_k)

            print('Old Heatmap:')
            _plot_surface(gp, mu_k, A, PW, X, step=k)
            print('New Heatmap:')
            _plot_surface(gp, pain_mu, A, PW, X, step=k)

    # ── save final GP mean / sd grids once ───────────────────────
    mu_grid, sd_grid = gp.predict(X_grid, return_std=True)
    pd.DataFrame(mu_grid.reshape(A.shape)).to_csv("mu_predictions.csv",
                                                  index=False, header=False)
    pd.DataFrame(sd_grid.reshape(A.shape)).to_csv("sd_predictions.csv",
                                                  index=False, header=False)

    if make_plots:
        _plot_history(best_hist)

    return X.tolist(), best_hist
# ───────────────────────────────────────────────────────────────────

amp_bounds = (0.5, 15)
pw_bounds  = (0.5, 980)

w1_ls = [3,0.5,1,0]
lpm_ls = [0.1,0.3,0.6,0.9]

sampled_pts, fr_history = BO_loop(opt_trials=10,
                                          amp_bounds=amp_bounds,
                                          pw_bounds=pw_bounds, 
                                          w=[0.5, 3],
                                          lower_pain_mod=0.5)

print(sampled_pts)
print(fr_history)
for idx, (a, p) in enumerate(sampled_pts, 1):
    print(f"{idx:02d}: amp={a:6.2f}  pw={p:7.2f}")

# Compute extra metrics
exp = calculate_pains(sampled_pts[3:], max_params=[amp_bounds[1], pw_bounds[1]], param_weights=[5,3], rpd_weight=5)
exp_names = list(exp.columns)
recalled_pain = exp.iloc[0][exp_names[-1]]
max_fr = max(fr_history)

print("Recalled Pain:", recalled_pain)
print("Max Firing Rate:", max_fr)

# Combine all data into a single DataFrame
df_combined = pd.DataFrame(sampled_pts[3:], columns=["amplitude", "pulse_width"])
df_combined["firing_rate"] = fr_history[1:]
df_combined["recalled_pain"] = recalled_pain
df_combined["max_firing_rate"] = max_fr

# Save combined CSV
df_combined.to_csv(f"combined_output_w1-3_lpm-01_10_runs.csv", index=False)

# Plot
px.line(exp, x=exp_names[2], y=exp_names[3:], title="Instantaneous Vs. Distorted & Recalled Pain").show()



'''
for i in w1_ls:
    for j in lpm_ls:
        print("W1:", i)
        print("LPM:", j)
        sampled_pts, fr_history = BO_loop(opt_trials=20,
                                          amp_bounds=amp_bounds,
                                          pw_bounds=pw_bounds, 
                                          w=[0.5, i],
                                          lower_pain_mod=j)
        print(sampled_pts)
        print(fr_history)
        for idx, (a, p) in enumerate(sampled_pts, 1):
            print(f"{idx:02d}: amp={a:6.2f}  pw={p:7.2f}")

        # Compute extra metrics
        exp = calculate_pains(sampled_pts[3:], max_params=[amp_bounds[1], pw_bounds[1]], param_weights=[5,3], rpd_weight=5)
        exp_names = list(exp.columns)
        recalled_pain = exp.iloc[0][exp_names[-1]]
        max_fr = max(fr_history)

        print("Recalled Pain:", recalled_pain)
        print("Max Firing Rate:", max_fr)

        # Combine all data into a single DataFrame
        df_combined = pd.DataFrame(sampled_pts[3:], columns=["amplitude", "pulse_width"])
        df_combined["firing_rate"] = fr_history[1:]
        df_combined["recalled_pain"] = recalled_pain
        df_combined["max_firing_rate"] = max_fr

        # Save combined CSV
        df_combined.to_csv(f"combined_output_w1-{i}_lpm-{j}_Dot_Product.csv", index=False)
        
        # Plot
        px.line(exp, x=exp_names[2], y=exp_names[3:], title="Instantaneous Vs. Distorted & Recalled Pain").show()
'''