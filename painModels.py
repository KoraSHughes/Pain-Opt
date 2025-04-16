# Note: all of these are assumptions, in future implementations we can make these unknown

def pain_function(trials, max_params=[200, 500], weights=[3,2]):
    """
    Input: array of model parameters for each n stimulation events
            can also weight the degree to which parameters contribute to pain
    Output: absolute pain score per stim event
    """
    # *Note: amplitude in microvolts, pulse-width in microseconds
    pain = []
    for i in range(len(trials)):
        instant_pain(trials[i], max_params, weights)
    return pain

def instant_pain(params, max_params, weights):
    """ params = [amplitude, pulse_width] """
    assert len(params) == len(weights) == len(max_params), "input parameters, weights, and maximum params must be identical in length"
    pain = 0
    for i in range(len(params)):  # for each parameter to our waveform
        pain += weights[i] * (params[i]/max_params[i])
    return pain


def MOP(pain_scores, mod=0.2):
    """ function describing principle 1 - memory of pain """
    # https://pmc.ncbi.nlm.nih.gov/articles/PMC3012151/table/T2/
    pain = max(pain_scores)*0.19 + pain_scores[-1]*0.14
    return pain*mod

def RPD(pain_scores, mod=0.7, deg=2, include_lower=False):
    """ principle 2 - relative pain distortion
    calculates the pain accrued over time """
    pain = 0
    for i in range(1, len(pain_scores)):
        if include_lower or (pain_scores[i] > pain_scores[i-1]):
            pain += abs(pain_scores[i] - pain_scores[i-1])**deg
    return pain*mod

def combined_pain(pain_scores, mod=[1,1]):
    """ calculates a combination of pain accrued over a trial """
    return MOP(pain_scores, mod[0]) + RPD(pain_scores, mod[1])


""" Intended application
Run optimization trial & store attempted stimulation params in an array.
Calculate the pain per stimulation event in that trial using pain_function
Use MOP, RPD, & combined_pain to understand the pain-quality of the experiment
"""