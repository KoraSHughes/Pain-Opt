# Note: all of these are assumptions, in future implementations we can make these unknown
""" Intended application
Run optimization trial & store attempted stimulation params in an array.
Calculate the institaneous pain per stimulation event in that trial using instant_pain().
Feed that into the pain_function() to inject the relative pain.
Evaluate model pain via pain_recall()
"""

MAX_PAIN = 10
def instant_pain(params, max_params=[5, 50], weights=[3,2]):
    """ Input: params = [amplitude, pulse_width], maximum bounds of those params, weight
        Output: [pain]"""
    # *Note: amplitude in microvolts, pulse-width in microseconds
    assert len(params) == len(weights) == len(max_params), "input parameters, weights, and maximum params must be identical in length"
    pain = 0
    for i in range(len(params)):  # for each parameter to our waveform
        pain += weights[i] * (params[i]/max_params[i])
    return pain

def relative_pain(last_pain, curr_pain, include_lower=True, deg=2):
    """ principle 2 - relative pain distortion - RPD
    calculates the distortion in pain reported """
    if curr_pain > last_pain:
        return (abs(curr_pain - last_pain)/MAX_PAIN)**deg
    elif include_lower:
        return -1 * (abs(curr_pain - last_pain)/MAX_PAIN)**deg
    return 0

def add_relative_pain(painful_trials, weight=2, include_lower=True):
    """ inject the relative pain into a set of painful trials 
    Note: weight is the maximum change in pain that can occur due to RPD """
    pain = []
    for i, p in enumerate(painful_trials):
        this_pain = p
        last_pain = 0 if i == 0 else painful_trials[i-1]
        this_pain += relative_pain(last_pain, this_pain, include_lower) * weight
        # add bounds for pain scale
        if this_pain > MAX_PAIN:
            this_pain = MAX_PAIN
        elif this_pain < 0:
            this_pain = 0
        pain.append(this_pain)
    return pain

""" end of study pain recall EOS_recall --> mean, peak, end
# estimated to be the average reported EOD/peak
# https://pmc.ncbi.nlm.nih.gov/articles/PMC3012151/table/T1/
according to these estimates: people report a EOD pain of:
- 84% of their peak pain
- 106% of their end pain
- 112% of their mean pain
"""
peak_mod = 46.5 / (41.6+13.9)
end_mod = 46.5 / (41.6+2.3)
def memory_pain(pain_scores):
    """ principle 1 - memory of pain - MOP
    calculates the patient's memory of a set of painful events
    in accordance with the peak-end rule """
    return (peak_mod*max(pain_scores) + end_mod*pain_scores[-1]) / 2

mean_mod = 46.5 / 41.6
def pain_recall(pain_scores):
    """ leverages memory of pain along with average pain to estimate the
        total pain the patient recalls
        takes a weighted average of the peak, end, and mean estimates
    """
    pp = peak_mod*max(pain_scores)
    ep = end_mod*pain_scores[-1]
    mp = mean_mod*sum(pain_scores)/len(pain_scores)
    return (1.07*mp + 0.3*pp + 0.1*ep)/(1.07+0.3+0.1)


# helper for testing
import pandas as pd
def calculate_pains(experiment, max_params=[5,50], param_weights=[5,3], rpd_weight=2):
    # TODO: make names versatile in case the params change
    exp_names = ["Amplitude (nA)", "Pulse Width (ms)", "Stimulation Event",
            "Instant Pain", "Percieved Pain", "Memory of Pain", "Mean Inst Pain",
            "Mean Perc Pain", "Total Recalled Pain"]
    stim_events = len(experiment)
    # calculate instantenous pain
    inst_pain = []
    for amp, pw in experiment:
        this_pain = instant_pain([amp, pw], max_params, param_weights)
        inst_pain.append(this_pain)
    distorted_pain = add_relative_pain(inst_pain, rpd_weight, True)
    # calculate memory of/recalled pain
    mem_pain = memory_pain(distorted_pain)
    mean_inst_pain = sum(inst_pain)/stim_events
    mean_perc_pain = sum(distorted_pain)/stim_events
    recalled_pain = pain_recall(distorted_pain)

    # construct dataframe
    exp = pd.DataFrame(experiment, columns=exp_names[:2])
    exp[exp_names[2]] = [i+1 for i in range(stim_events)]  # add labeling for order
    exp[exp_names[3]] = inst_pain  # add inst pain
    exp[exp_names[4]] = distorted_pain  # add pain with RPD distortion
    exp[exp_names[5]] = [mem_pain for i in range(stim_events)]  # add memory pain
    exp[exp_names[6]] = [mean_inst_pain for i in range(stim_events)]  # add mean pain
    exp[exp_names[7]] = [mean_perc_pain for i in range(stim_events)]  # add mean pain
    exp[exp_names[8]] = [recalled_pain for i in range(stim_events)]  # add recalled pain
    return exp