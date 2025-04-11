def pain_function(trials, weights=[3,2], max_amp=200, max_pw=5):
    """
    Input: array of model parameters for each n stimulation events
            can also weight the degree to which parameters contribute to pain
    Output: absolute pain score per stim event
    """
    # *Note: amplitude in microvolts, pulse-width in microseconds
    pain = []
    for amp, pw in trials:
        this_pain = 0
        this_pain += weights[0]*amp/max_amp
        this_pain += weights[1]*pw/max_pw
        pain.append(this_pain)
    return pain


def MOP(pain_scores, mod=0.2):
    """ function describing principle 1 - memory of pain """
    pain = max(pain_scores)*0.1 + pain_scores[-1]*0.5
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