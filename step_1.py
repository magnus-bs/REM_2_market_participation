
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
import plot_functions as pf
import useful_functions as uf
import random
from itertools import product


#%%----------------------------
# Scenario Generation
#------------------------------

FARM_CAPACITY_MW = 500
N_DAYS = 5 # per season, total 20 days

df_wind_scenarios = uf.wind_scenario_generation(FARM_CAPACITY_MW=FARM_CAPACITY_MW, N_DAYS=N_DAYS)
df_price_scenarios = uf.price_scenario_generation()
df_imbalance_scenarios = uf.imbalance_scenario_generation()

# Set index to Hour
df_wind = df_wind_scenarios.set_index("Hour")
df_price = df_price_scenarios.set_index("Hour")
df_imb = df_imbalance_scenarios.set_index("Hour")

# Extract column names for each scenario type
Omega_prod = df_wind.columns
Omega_price = df_price.columns
Omega_imb = df_imb.columns

# Build full scenario space as Cartesian product of the three scenario types
Omega_full = list(product(Omega_prod, Omega_price, Omega_imb))

# Sample scenarios
SAMPLE_SIZE= 200
random.seed(42)

Omega_sample = random.sample(Omega_full, SAMPLE_SIZE)

Omega = range(SAMPLE_SIZE)
T = range(24)

#%%----------------------------
# Build parameters
#------------------------------
P_real = {}
lambda_DA = {}
y = {}

# Map sampled scenarios to their corresponding values in the dataframes
for w, (p, pr, im) in enumerate(Omega_sample):
    for t in T:
        P_real[(t, w)] = df_wind.loc[t, p]
        lambda_DA[(t, w)] = df_price.loc[t, pr]
        y[(t, w)] = df_imb.loc[t, im]

# Define imbalance prices based on the imbalance direction
lambda_up = {
    (t, w): 1.25 * lambda_DA[(t, w)]
    for t in T for w in Omega
}

lambda_down = {
    (t, w): 0.85 * lambda_DA[(t, w)]
    for t in T for w in Omega
}

lambda_imb = {
    (t, w): lambda_up[(t, w)] if y[(t, w)] > 0 else lambda_down[(t, w)]
    for t in T for w in Omega
}

# Define equal probabilities for each scenario
pi = {w: 1 / len(Omega) for w in Omega}


#%% ------------------------------------------------------------------------------
#                                      Task 1.1                                 
#   ------------------------------------------------------------------------------

# -------- Define model ---------
op_model = gb.Model("Profit_Maximization")

# ---- Add decision variables ----
p_DA = op_model.addVars(T, lb=0, ub=FARM_CAPACITY_MW, name="p_DA")

# ------ Objective Function ------
op_model.setObjective(
    gb.quicksum(
        pi[w] * (
            lambda_DA[(t, w)] * p_DA[t]
            + lambda_imb[(t, w)] * (P_real[(t, w)] - p_DA[t])
        )
        for t in T for w in Omega
    ),
    gb.GRB.MAXIMIZE
)
#%% --------- Solution ---------
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


