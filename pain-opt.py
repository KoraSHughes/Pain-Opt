# optimiziation imports
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import numpy as np

# modelling imports
from updated_run import run_simulation          # NEURON wrapper
from painModels import (
    instant_pain, add_relative_pain, calculate_pains, MAX_PAIN
)

# visualization imports
import matplotlib.pyplot as plt
import plotly.express as px

np.random.seed(23)

# pain helpers:
def ideal_pain_curve(trial_num, max_trial_num):  # helper for ideal pain curve (used to add MOP-based weights )
    """ -(4B/A^2) * x(x-A) :: A is the max#trials, B is the ideal height aka pain,
                                C is roughly where we want to start our experiment """
    x = trial_num
    height_mod = 2
    A = max_trial_num
    B = MAX_PAIN*0.9 - height_mod  # * Note: in ideal we only want 90% of our max pain?
    return (-4*B * x * (x-A)) / (A**2) + height_mod

def weight_ppain(params, last_params, trial_num, max_trial_num, max_params, w=[0.25, 0.2]):
    """ returns a weight to the uncertainty with a max deviating percentage of sum(w) 
    ie, at the highest pain possible, we are disencentivizing sampling by (sum(w)*100)%"""
    # scaling modifiers
    if sum(w) > 1:
        temp = sum(w)
        w[0] = w[0]/temp
        w[1] = w[1]/temp
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
    assert modifier <= w[0]
    # memory of pain
    idpain = ideal_pain_curve(trial_num, max_trial_num)
    if ppain > idpain:
        modifier += w[1] * (abs(ppain - idpain)/MAX_PAIN)
    else:
        modifier += (w[1]/4) * (abs(ppain - idpain)/MAX_PAIN)
    assert modifier <= 1
    return (1-modifier)

def scale_pain(mygrid, last_params, trial, max_trials=20, max_params=[80, 30], w=[0.25, 0.2]):
    res = []
    for i in range(mygrid.shape[0]):
        params = [mygrid[i, 0], mygrid[i, 1]]  # amp, pw
        res.append(weight_ppain(params, last_params, trial, max_trials, max_params, w))
    return np.array(res)

# ──────────────────────────────────────────────────────────────────────
def neuron_objective(X):
    """
    X : (n,2) or (2,) – [amplitude, pulse‑width]
    Returns positive firing‑rate values (we maximise directly).
    """
    X = np.atleast_2d(X)
    vals = [run_simulation(float(a), float(pw),
                            sim_duration=1000,
                            stim_delay=5,
                            plot_traces=True)       # plotting restored
            for a, pw in X]
    return np.asarray(vals)

def latin_hypercube(n, amp_bounds, pw_bounds):
    amp = np.random.uniform(*amp_bounds, size=n)
    pw  = np.random.uniform(*pw_bounds , size=n)
    return np.column_stack((amp, pw))

def acquisition_xi(mu, sigma, best, xi=0.01):
    z = (mu - best - xi) / (sigma + 1e-8)
    return (mu - best - xi) * norm.cdf(z) + sigma * norm.pdf(z)

def acquisition(mu, sigma, best):
    z = (mu - best) / (sigma + 1e-8)
    return (mu - best) * norm.cdf(z) + sigma * norm.pdf(z)

# ────────────── helpers ───────────────────────────────────────────────
def _plot_surface(mu, sampled, A, PW, step):
    """Heat‑map of GP mean + sampled points."""
    plt.figure(figsize=(6, 4))
    cs = plt.contourf(A, PW, mu.reshape(A.shape), levels=30, alpha=0.75)
    plt.colorbar(cs, label="GP mean (Hz)")
    plt.scatter(sampled[:, 0], sampled[:, 1],
                c="red", edgecolor="k", s=60, label="Samples")
    plt.title(f"BO iteration {step}")
    plt.xlabel("Amplitude"); plt.ylabel("Pulse‑width")
    plt.tight_layout(); plt.show()


def _plot_history(fr_history):
    """Line plot of raw firing‑rate across optimisation iterations."""
    plt.figure(figsize=(6, 3))
    plt.plot(fr_history, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Firing rate (Hz)")
    plt.title("Firing rate across Bayesian‑optimisation iterations")
    plt.tight_layout(); plt.show()
# ──────────────────────────────────────────────────────────────────────


def BO_loop(opt_trials=25,
            amp_bounds=(2, 80),
            pw_bounds=(0.01, 30),
            grid=120,
            make_plots=True):
    kernel = RBF([10.0, 300.0], length_scale_bounds=(1e-3, 1e5))
    gp = GaussianProcessRegressor(kernel=kernel,
                                optimizer=None,
                                normalize_y=False)

    # dense prediction grid
    A, PW = np.meshgrid(np.linspace(*amp_bounds, grid),
                        np.linspace(*pw_bounds , grid), indexing='ij')
    X_grid = np.c_[A.ravel(), PW.ravel()]

    # ── initial Latin‑hypercube sample ───────────────────────────────
    X = latin_hypercube(5, amp_bounds, pw_bounds)
    y_raw = neuron_objective(X)               # firing‑rate in Hz

    fr_history = y_raw.tolist()               # record raw values

    mu_y, sd_y = y_raw.mean(), y_raw.std() + 1e-9
    y = (y_raw - mu_y) / sd_y
    gp.fit(X, y)

    if make_plots:
        mu0, _ = gp.predict(X_grid, return_std=True)
        _plot_surface(mu_y + mu0*sd_y, X, A, PW, step=0)

    # ── BO iterations ────────────────────────────────────────────────
    max_bounds = (amp_bounds[1], pw_bounds[1])
    last_params = []
    for k in range(1, opt_trials + 1):
        oldmu, sd = gp.predict(X_grid, return_std=True)
        # ! derive pain factors
        pain_factors = scale_pain(X_grid, last_params, k, opt_trials, max_bounds)
        mu = np.multiply(pain_factors, oldmu)  # scale means

        ei = acquisition(mu, sd, y.max())

        x_next = X_grid[np.argmax(ei)]
        last_params = x_next
        y_next_raw = neuron_objective(x_next)[0]
        fr_history.append(y_next_raw)

        y_next = (y_next_raw - mu_y) / sd_y
        X = np.vstack([X, x_next])
        y = np.append(y, y_next)
        gp.fit(X, y)

        if make_plots:
            mu_pred, _ = gp.predict(X_grid, return_std=True)
            _plot_surface(mu_y + mu_pred*sd_y, X, A, PW, step=k)

    # ── final history plot ───────────────────────────────────────────
    if make_plots:
        _plot_history(fr_history)

    return X.tolist()

# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    amp_bounds = (0,15)
    pw_bounds = (0,980)
    max_params = [amp_bounds[1], pw_bounds[1]]
    test_weights= [5,3]
    history = BO_loop(amp_bounds=amp_bounds, pw_bounds=pw_bounds, opt_trials=20)   # change trials as you like
    for i, (a, p) in enumerate(history, 1):
        print(f"{i:02d}: amp={a:6.2f}  pw={p:6.2f}")

exp = calculate_pains(history, max_params, test_weights, 5)
exp_names = list(exp.columns)
px.line(exp, x=exp_names[2], y=exp_names[3:], title="Instantaneous Vs. Distorted & Recalled Pain")
