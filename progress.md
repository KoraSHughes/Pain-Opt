# Kora
# Rabira
# Ezekiel
Working on creating the Thompson Sampling (TS) model. I have it so far setup to have the following arms:
- An arm that changes both amplitude and pulse width for every iteration
- An arm that only changes pulse width and keeps a static amplitude
- An arm that only changes amplitude but keeps a static pulse width

Of those, we want to identify which technique gives the highest firing rate. If we were to incoropoate the pain model
I plan on using one of the following:
- Poisson Distr: 
- Normal Distr: mean(μ) = 5 & std(σ²) = 2
- Linear Function: Given from Kora's calculation