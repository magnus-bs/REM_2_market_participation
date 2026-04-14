
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

# Settings:
FARM_CAPACITY_MW = 500
IN_SAMPLE_SIZE= 200
random.seed(42)

# Load scenarios
df_wind_scenarios = uf.wind_scenario_generation(SCENARIOS_PER_SEASON= 5, FARM_CAPACITY_MW=FARM_CAPACITY_MW, data_folder='Data')
df_price_scenarios = uf.price_scenario_generation(SCENARIOS_PER_SEASON = 5, data_folder='Data')
df_imbalance_scenarios = uf.imbalance_scenario_generation(N_SCENARIOS=4, hours_per_day=24, data_folder='Data')

# Extract column names for each scenario type
Omega_wind = df_wind_scenarios.columns
Omega_price = df_price_scenarios.columns
Omega_imb = df_imbalance_scenarios.columns

# Build full scenario space as Cartesian product of the three scenario types
Omega_full = list(product(Omega_wind, Omega_price, Omega_imb))

# Set of In-sample scenarios for optimisation
Omega_in = random.sample(Omega_full, IN_SAMPLE_SIZE)

# Out-of-sample scenarios for ex-post analysis
Omega_out = list(set(Omega_full) - set(Omega_in))

# Build parameters for in-sample and out-of-sample scenarios
P_real_in, lambda_DA_in, y_imb_in, lambda_imb_in, pi_in = uf.build_parameters(Omega_in, df_wind_scenarios, df_price_scenarios, df_imbalance_scenarios)
P_real_out, lambda_DA_out, y_imb_out, lambda_imb_out, pi_out = uf.build_parameters(Omega_out, df_wind_scenarios, df_price_scenarios, df_imbalance_scenarios)


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


