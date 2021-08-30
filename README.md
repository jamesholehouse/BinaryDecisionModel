# BinaryDecisionModel
Modules for simulation and analytics of the mean-field binary decision model [1,2]. Code written in Julia. In order to view the project check out the `.ipynb` files. A summary of each of these files is detailed below:

- `How-to-Analytic-Binary-Decisions.ipynb`: details usage of the analytic solution in order to produce figures for the paper (yet to be uploaded to Arxiv). Shows that solution agrees with Monte Carlo simulation provided by the SSA [3].
- `How-to-SSA-BinDecModel.ipynb`: details on how to run the SSA for the mean-field binary decision model.
- `Max-likelihood-estimation.ipynb`: details how to construct a likelihood function from the analytic solution, from which MLE is performed using adaptive differential evolution algorithms from `BlackBoxOptim` [4].
- `MF_BDM_Analytic.jl`: code for the analytic time-dependent solution.
- `SSABinaryDecision.jl`: code for the SSA.

Any comments or issues please get in touch at james.holehouse@ed.ac.uk or jamesholehouse1@gmail.com.

[1] Brock, W. A. & Durlauf, S. N. Discrete choice with social interactions. The Review of Economic Studies 68, 235–260 (2001).

[2] Bouchaud, J.-P. Crises and collective socio-economic phenomena: simple models and challenges. Journal of Statistical Physics 151, 567–606 (2013).

[3]  Gillespie, D. T. Stochastic simulation of chemical kinetics. Annu. Rev. Phys. Chem. 58, 35–55 (2007).

[4]  Feldt, R. BlackBoxOptim. https://github.com/robertfeldt/BlackBoxOptim.jl. [Online; accessed 30-August-2021].
