# Non-Parametric Correlation

Nadaraya-Watson estimator for stocks correlation. Non-parametric approaches can be flexible, and valuable, when computing the correlation between stocks, which, in reality is not constant over time. The time dependance feature supposes variation, thus allowing a clear analysis of the time series's dynamic nature. 

$E(r_{it}) \approx 1/T \sum r_{it}$

$Var(r_{it}) \approx \dfrac{1}{T} \sum r_{it}^2 - E(r_{it})^2$
