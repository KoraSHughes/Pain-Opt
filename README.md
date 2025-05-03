# Pain-Opt
An investigation into pain-aware sampling in Bayesian informed neural stimulation experiments
## Abstract
Adaptive sampling and optimal control methods often overlook constraints needed to ensure that sampled parameters do not compromise system integrity. This is especially significant in the context of human trials, where reinforcement learning methods may incentivize actions that are effective for reward maximization but risk causing physical/psychological damage to patients. In neurostimulation experiments, this means that adaptive sampling techniques such as Bayesian Optimization may incentivize the injection of large current waveforms that inadvertently cause tissue damage or severe pain. Though hard limitations on waveform exploration are applied to ensure patient safety, they often still incentivize the most painful solutions and fail to take into account more complex, time-variant models of pain. To remedy this, we propose PainOpt; a pain-informed adaptive sampling method and data-driven model that leverages a time-variant understanding of patient pain to minimize recalled pain. When benchmarked against traditional sampling methods with no consideration of pain, we observe a 37+% reduction in recalled pain while preserving the original treatment objective.

## Repository Navigation
- Main Code:
  - painOpt.py :: main pain-informed optimization code
    - painModels.py :: main models of pain
  - updated_run.py :: main traditional optimizers to be tested
- Visualizations:
  - analysis.ipynb :: main visualization of the basic pain models & tester code for the pain-recall
  - pain-injection-tests.ipynb :: investigation of pain-informed weights + visual examples
  - results/* :: additional figures saved from the paper
- Simulation:
  - ...
- Optimizers:
  - ax-bo.ipynb :: Early Bayesian Optimization algorithm once again for axis
  - custom-bo-example :: Working prototype of earlier implementation of PainOpt
  - hard_bayes.py :: Bayesian Optimization implementation test example
  - pain_info_bandit_arms :: Multi-Armed bandit with pain-influenced rewards
  - pointElec_simulation.py :: NEURON connected point electrode simulation
  - qlearning.py :: Basic Q-Learning Algorithm Implementation (Unfinished)
  - thompson.py :: Thompson Sampling Implementation (Weak Reward)
  - trad-bayes.py :: Traditional Bayesian Optimization Implementation

### Execution
- Thompson Sampling: python3 thompson.py
- Q-Learning: python3 qlearning.py (Needs updating has errors)
