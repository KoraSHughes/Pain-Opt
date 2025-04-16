# Note: all of these are assumptions, in future implementations we can make these unknown
""" Intended application
Run optimization trial & store attempted stimulation params in an array.
Calculate the pain per stimulation event in that trial using pain_function().
Evaluate model pain via total_pain()
"""

MAX_PAIN = 10
def pain_function(trials, max_params=[200, 500], weights=[3,2], include_rpd=True):
    """
    Input: array of model parameters for each n stimulation events
            can also weight the degree to which parameters contribute to pain
    Output: absolute pain score per stim event
    """
    # *Note: amplitude in microvolts, pulse-width in microseconds
    pain = []
    for i in range(len(trials)):
        this_pain = instant_pain(trials[i], max_params, weights)
        if include_rpd:  # tldr exaserbates the change in percieved pain +-[0,1]
            last_pain = 0 if len(pain) == 0 else pain[-1]
            this_pain += RPD(last_pain, this_pain)
            if this_pain > MAX_PAIN:
                this_pain = MAX_PAIN
            elif this_pain < 0:
                this_pain = 0
        pain.append(this_pain)
    return pain

def instant_pain(params, max_params, weights):
    """ params = [amplitude, pulse_width] """
    assert len(params) == len(weights) == len(max_params), "input parameters, weights, and maximum params must be identical in length"
    pain = 0
    for i in range(len(params)):  # for each parameter to our waveform
        pain += weights[i] * (params[i]/max_params[i])
    return pain

def RPD(last_pain, curr_pain, include_lower=True, deg=2):
    """ principle 2 - relative pain distortion
    calculates the distortion in pain reported """
    if curr_pain > last_pain:
        return (abs(curr_pain - last_pain)/MAX_PAIN)**deg
    elif include_lower:
        return -1 * (abs(curr_pain - last_pain)/MAX_PAIN)**deg
    return 0

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
def MOP(pain_scores):
    """ memory of pain according to peak-end rule """
    return peak_mod*max(pain_scores) + end_mod*pain_scores[-1]
mean_mod = 46.5 / 41.6
def pain_recall(pain_scores, include_mean=True):
    """ leverages memory of pain along with average pain to estimate the
        total pain the patient recalls
        takes a weighted average of the peak, end, and mean estimates
    """
    mop = MOP(pain_scores)
    if include_mean:
        pain_mean = mean_mod*sum(pain_scores)/len(pain_scores)
        return (mop + pain_mean)/3
    return mop/2
