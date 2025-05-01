import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import DotProduct
import matplotlib.pyplot as plt
from updated_run import run_simulation          # NEURON wrapper
from painModels import calculate_pains
import plotly.express as px
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
from matplotlib import cm  # Colormap
import plotly.graph_objects as go

from painModels import (
    instant_pain, add_relative_pain, calculate_pains
)

np.random.seed(23)

# ───────────────── helpers ──────────────────────────────────────────
def neuron_objective(X):
    X = np.atleast_2d(X)
    return np.array([run_simulation(float(a), float(pw),
                                    sim_duration=1000,
                                    stim_delay=5,
                                    plot_traces=True)   # keep UI clean
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
            make_plots=True):
    # GP definition
    kernel = RBF(length_scale=[2.75, 250], length_scale_bounds=(1e-3, 1e5))
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

    # ── optimisation loop ────────────────────────────────────────
    for k in range(1, opt_trials + 1):
        mu, sd = gp.predict(X_grid, return_std=True)
        ei = expected_improvement(mu, sd, np.max(best_hist))

        x_next = X_grid[np.argmax(ei)]
        y_next = neuron_objective(x_next)[0]

        # update datasets
        X = np.vstack([X, x_next])
        y = np.append(y, y_next)
        best_hist.append(y_next)

        gp.fit(X, y)

        if make_plots:
            mu_k, _ = gp.predict(X_grid, return_std=True)
            _plot_surface(gp, mu_k, A, PW, X, step=k)

    # ── save final GP mean / sd grids once ───────────────────────
    mu_grid, sd_grid = gp.predict(X_grid, return_std=True)
    pd.DataFrame(mu_grid.reshape(A.shape)).to_csv("mu_predictions.csv",
                                                  index=False, header=False)
    pd.DataFrame(sd_grid.reshape(A.shape)).to_csv("sd_predictions.csv",
                                                  index=False, header=False)

    if make_plots:
        _plot_history(best_hist)

    return X.tolist(), best_hist
# ────────────────────────────────────────────────────────────────────

amp_bounds = (0, 15)
pw_bounds  = (0, 980)

sampled_pts, fr_history = BO_loop(opt_trials=10,
                                    amp_bounds=amp_bounds,
                                    pw_bounds=pw_bounds)

for i, (a, p) in enumerate(sampled_pts, 1):
    print(f"{i:02d}: amp={a:6.2f}  pw={p:7.2f}")

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
#df_combined.to_csv(f"combined_output_Trad_Bayes.csv", index=False)

# Plot
px.line(exp, x=exp_names[2], y=exp_names[3:], title="Instantaneous Vs. Distorted & Recalled Pain").show()
