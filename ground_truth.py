import numpy as np
import matplotlib.pyplot as plt
from updated_run import run_simulation
import pandas as pd

def firing_rate_grid(x_samples,
                     x_range,
                     y_samples,
                     y_range,
                     sim_duration=40,
                     stim_delay=5,
                     show_heatmap=True,
                     csv_path=None):          # <-- optional CSV output
    """
    Returns
    -------
    amps : 1‑D np.ndarray
    pws  : 1‑D np.ndarray
    fr_matrix : 2‑D np.ndarray
    """
    # ── sample space ────────────────────────────────────────────────────────
    amp_min, amp_max = x_range
    pw_min,  pw_max  = y_range

    #amps = np.logspace(np.log10(amp_min),
    #                   np.log10(amp_max),
    #                   num=x_samples)
    amps = np.linspace(amp_min, amp_max, num=x_samples)
    pws  = np.linspace(pw_min, pw_max, num=y_samples)

    fr_matrix = np.empty((x_samples, y_samples))

    # ── sweep ───────────────────────────────────────────────────────────────
    for i, amp in enumerate(amps):
        for j, pw in enumerate(pws):
            fr_matrix[i, j] = run_simulation(amp,
                                             pw,
                                             sim_duration=sim_duration,
                                             stim_delay=stim_delay,
                                             plot_traces=False)

    # ── optional CSV save ───────────────────────────────────────────────────
    if csv_path is not None:
        df = pd.DataFrame(fr_matrix, index=amps, columns=pws)
        df.index.name   = "Amplitude_(µA)"
        df.columns.name = "PulseWidth_(ms)"
        df.to_csv(csv_path)

    # ── visualisation ───────────────────────────────────────────────────────
    if show_heatmap:
        fig, ax = plt.subplots()
        img = ax.imshow(fr_matrix,
                        origin='lower',
                        aspect='auto',
                        extent=[pw_min, pw_max, amp_min, amp_max],
                        cmap='viridis')
        #ax.set_yscale('log')
        ax.set_xlabel('Pulse‑width (ms)')
        ax.set_ylabel('Amplitude (µA)')
        ax.set_title('Neuron firing‑rate heatmap')
        fig.colorbar(img, ax=ax, label='Firing rate (Hz)')
        plt.show()

    return amps, pws, fr_matrix

amps, pws, fr = firing_rate_grid(
        x_samples=100,
        x_range=(0.1, 40),
        y_samples=30,
        y_range=(0, 30),
        csv_path="firing_rates2.csv") 