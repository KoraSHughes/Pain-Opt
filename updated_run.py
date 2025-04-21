from neuron import h,gui
import numpy as np
from scipy.signal import find_peaks

def run_simulation(amplitude, pulse_width, sim_duration=10, stim_delay=5, plot_traces=False):
    print(amplitude)
    print(pulse_width)
    
    h.load_file("stdrun.hoc")
    h.load_file("import3d.hoc")
    h.load_file("constants.hoc")
    h.load_file("morphology.hoc")
    h.load_file("biophysics.hoc")
    h.load_file("template.hoc")

    cell = h.cADpyr229_L23_PC_8ef1aa6602(0)

    stim = h.IClamp(0.5, sec=cell.soma[0])
    stim.delay = stim_delay
    stim.dur = pulse_width
    stim.amp = amplitude

    v_vec = h.Vector().record(cell.axon[18](0.5)._ref_v)
    t_vec = h.Vector().record(h._ref_t)

    h.tstop = sim_duration
    h.cvode_active(0)
    h.run()

    v = np.array(v_vec)
    t = np.array(t_vec)

    num_peaks, _ = find_peaks(v[:-1], prominence=40)
    firing_rate = len(num_peaks)/ (h.tstop / 1000)
    print(firing_rate)

    if plot_traces:
        import matplotlib.pyplot as plt
        plt.plot(t, v)
        plt.xlabel("Time (ms)")
        plt.ylabel("Membrane potential (mV)")
        plt.title("Vm trace")
        plt.grid(True)
        plt.show()

    return firing_rate
