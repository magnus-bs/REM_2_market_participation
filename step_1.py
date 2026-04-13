
"""
Title:
Renewables in Electricity Markets 2: Market Participation
Step 1: Participation in Day-ahead and Balancing Markets

Authors:
Magnus B. Sørensen
Freja R. Søndergaard
Sebastian C. Stokbro
Sámuel Gregersen

Overview:
Task 1.1) Offering Strategy Under a One-Price Balancing Scheme
Task 1.2) Offering Strategy Under a Two-Price Balancing Scheme
Task 1.3) Ex-post Analysis
Task 1.4) Risk-Averse Offering Strategy
"""


#%% Libraries

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import gurobipy as gb
import gurobipy as GRB
import plot_functions.py as pf
import useful_functions.py as uf

#%% Scenario Generation

FARM_CAPACITY_MW = 500
N_DAYS = 5 # per season, total 20 days

wind_scenarios = uf.wind_scenario_generation(FARM_CAPACITY_MW=FARM_CAPACITY_MW, N_DAYS=N_DAYS)








# Balancing Prices:




#%% ------------------------------------------------------------------------------
#                                      Task 1.1                                 
#   ------------------------------------------------------------------------------

# --------------------
# Sets
# --------------------
T = range(24)
Omega = range(200)

# --------------------
# Parameters
# --------------------
P_nom = 500

P_real = {(t,w): ... for t in T for w in Omega}
lambda_DA = {(t,w): ... for t in T for w in Omega}
y = {(t,w): ... for t in T for w in Omega}

pi = {w: 1/len(Omega) for w in Omega}

lambda_up = {(t,w): 1.25 * lambda_DA[t,w] for t in T for w in Omega}
lambda_down = {(t,w): 0.85 * lambda_DA[t,w] for t in T for w in Omega}

lambda_imb = {
    (t,w): lambda_up[t,w] if y[t,w] == 1 else lambda_down[t,w]
    for t in T for w in Omega
}

# --------------------
# Model
# --------------------s
op_model = gb.Model("Profit Maximization")

# Decision variable
p_DA = op_model.addVars(T, lb=0, ub=P_nom, name="p_DA")

# --------------------
# Objective
# --------------------
op_model.setObjective(
    gb.quicksum(
        pi[w] * (
            lambda_DA[t,w] * p_DA[t]
            + lambda_imb[t,w] * (P_real[t,w] - p_DA[t])
        )
        for t in T for w in Omega
    ),
    GRB.MAXIMIZE
)

# --------------------
# Solve
# --------------------
op_model.optimize()


#%% ------------------------------------------------------------------------------
#                                      Task 1.2                                 
#   ------------------------------------------------------------------------------







#%% ------------------------------------------------------------------------------
#                                      Task 1.3                                
#   ------------------------------------------------------------------------------






#%% ------------------------------------------------------------------------------
#                                      Task 1.4                                 
#   ------------------------------------------------------------------------------


