# Non-Parametric Estimator

Nadaraya-Watson estimator for stocks correlation.

Non-parametric approaches can be flexible, and valuable, when computing correlation between stocks, which, in reality is not constant over time. The time dependance feature supposes variation, thus allowing for a clear analysis of the time series's dynamic nature. 

$E(r_{it}) \approx \dfrac{1}{T} \sum r_{it}$

$Var(r_{it}) \approx \dfrac{1}{T} \sum r_{it}^2 - E(r_{it})^2$

$Cov(r_{1t},r_{2t}) \approx \dfrac{1}{T} \sum (r_{1t} - E(r_{1t})) (r_{2t} - E(r_{2t}))$

Alternatively, the daily correlation could be computed from the intra-day data. If t indexes the day, and jt indexes J equi-distant observations within day t, realized measures could be: 

$Var(r_{t}) \approx \sum r_{jt}^2 $

$Cov(r_{1t},r_{2t}) \approx \sum r_{1j} r_{2j}$
