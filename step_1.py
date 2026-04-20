
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
import random
from itertools import product

import plot_functions as pf
import useful_functions as uf
import plot_settings


#%%----------------------------
# Scenario Generation
#------------------------------

# Settings:
FARM_CAPACITY_MW = 500
IN_SAMPLE_SIZE= 200
SEED = 42
random.seed(SEED)

# Load scenarios
df_wind_scenarios = uf.wind_scenario_generation(N_SCENARIOS = 20, FARM_CAPACITY_MW=FARM_CAPACITY_MW, data_folder='Data', RANDOM_SEED=SEED)
df_price_scenarios = uf.price_scenario_generation(N_SCENARIOS = 20, data_folder='Data', RANDOM_SEED=SEED)
df_imbalance_scenarios = uf.imbalance_scenario_generation(N_SCENARIOS=4, hours_per_day=24, data_folder='Data')



#%%

def plot_scenarios(df_wind_scenarios, df_price_scenarios, df_imbalance_scenarios, figsize=(10, 3.2), dpi=300):
    
    fig, ax1 = plt.subplots(figsize=figsize, dpi=dpi)
    ax2 = ax1.twinx()

    # --- Wind (left axis) ---
    for col in df_wind_scenarios.columns:
        ax1.plot(df_wind_scenarios.index, df_wind_scenarios[col], color='skyblue', alpha=0.8, linewidth=1)
    #ax1.plot(df_wind_scenarios.index, df_wind_scenarios.mean(axis=1), color='skyblue', marker='o', label='Average Wind Power')
    ax1.set_ylabel('Power (MW)', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')

    # --- Prices (right axis) ---
    for col in df_price_scenarios.columns:
        ax2.plot(df_price_scenarios.index, df_price_scenarios[col], color='coral', alpha=0.6, linewidth=1)
    #ax2.plot(df_price_scenarios.index, df_price_scenarios.mean(axis=1), color='coral', marker='o', label='Average Day-Ahead Price')
    ax2.set_ylabel('Price (EUR/MWh)', color='coral')
    ax2.tick_params(axis='y', labelcolor='coral')

    # --- Formatting ---
    ax1.set_xlabel('Hour of the Day')
    ax1.set_xticks(df_wind_scenarios.index)
    ax1.grid(alpha=0.3)

    # Legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elements = [
        Line2D([0], [0], color='skyblue', label='Wind Power (MW)'),
        Line2D([0], [0], color='coral', label='DA Price (EUR/MWh)'),
    ]
    ax1.legend(handles=legend_elements, loc='upper left')

    plt.tight_layout()
    plt.show()

plot_scenarios(df_wind_scenarios, df_price_scenarios, df_imbalance_scenarios)


#%%



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

# Set of hours
T = range(24)

# -------- Define model ---------
op_model = gb.Model("Profit_Maximization_op")

# ---- Add decision variables ----
p_DA = op_model.addVars(T, lb=0, ub=FARM_CAPACITY_MW, name="p_DA")

# ------ Objective Function ------

op_model.setObjective(
    gb.quicksum(
        pi_in[w] * (
            lambda_DA_in[(t, w)] * p_DA[t]
            + lambda_imb_in[(t, w)] * (P_real_in[(t, w)] - p_DA[t])
        )
        for t in T for w in Omega_in
    ),
    gb.GRB.MAXIMIZE
)

#%% --------- Solution ---------
op_model.optimize()

#%% --------- Results ----------

print("RESULTS OF ONE-PRICE BALANCING SCHEME:")
print(40*"-")

# Extract optimal day-ahead offering for each hour
p_DA_optimal = {t: p_DA[t].X for t in T}
print("Optimal Day-ahead Offering (MW):")
for t in T:
    print(f"Hour {t}: {p_DA_optimal[t]:.2f} MW")

#da ahead price and imbalnce price in each hour
lambda_DA_avg = {
    t: sum(pi_in[w] * lambda_DA_in[(t, w)] for w in Omega_in)
    for t in T
}

lambda_imb_avg = {
    t: sum(pi_in[w] * lambda_imb_in[(t, w)] for w in Omega_in)
    for t in T
}

# plot optimal offering strategy
pf.plot_optimal_offering(T, p_DA_optimal, lambda_DA_avg, lambda_imb_avg)

# Calculate expected profit under the optimal offering strategy
expected_profit = op_model.ObjVal
print(f"\nExpected Profit under Optimal Offering Strategy: {expected_profit:.2f} DKK")

# Compute and plot profit distribution across in-sample scenarios
pf.plot_profit_distribution(T, Omega_in, p_DA_optimal, P_real_in, lambda_DA_in, lambda_imb_in)

#%% ------------------------------------------------------------------------------
#                                      Task 1.2                                 
#   ------------------------------------------------------------------------------


# -------- Define model ---------
tp_model = gb.Model("Profit_Maximization_tp")

# ---- Add decision variables ----
p_DA = tp_model.addVars(T, lb=0, ub=FARM_CAPACITY_MW, name="p_DA")
Delta_tw_up = tp_model.addVars(T, Omega_in, lb=0, name="Delta_up")
Delta_tw_down = tp_model.addVars(T, Omega_in, lb=0, name="Delta_down")

#%%

# ------ Objective Function ------
tp_model.setObjective(
    gb.quicksum(
        pi_in[w] * (
            lambda_DA_in[(t, w)] * p_DA[t]
            + y_imb_in[(t, w)] * (lambda_DA_in[(t, w)] * Delta_tw_up[(t, w)]
                                 - lambda_imb_in[(t, w)]*Delta_tw_down[(t, w)]) 
            + (1-y_imb_in[(t, w)]) * (lambda_imb_in[(t, w)] * Delta_tw_up[(t, w)]
                                      - lambda_DA_in[(t, w)]*Delta_tw_down[(t, w)])
            
        )
        for t in T for w in Omega_in
    ),
    gb.GRB.MAXIMIZE
)

# ------ Constraints ------
for t in T:
    for w in Omega_in:
        tp_model.addConstr(
            P_real_in[(t, w)]-p_DA[t] == Delta_tw_up[(t, w)] - Delta_tw_down[(t, w)],
            name=f"Balance_Constraint_t{t}_w{w}"
        )

#%% --------- Solution ---------
tp_model.optimize()


#%% --------- Results ----------

print("RESULTS OF TWO-PRICE BALANCING SCHEME:")
print(40*"-")

# Extract optimal day-ahead offering for each hour
p_DA_optimal = {t: p_DA[t].X for t in T}
print("Optimal Day-ahead Offering (MW):")
for t in T:
    print(f"Hour {t}: {p_DA_optimal[t]:.2f} MW")

#da ahead price and imbalnce price in each hour
lambda_DA_avg = {
    t: sum(pi_in[w] * lambda_DA_in[(t, w)] for w in Omega_in)
    for t in T
}

lambda_imb_avg = {
    t: sum(pi_in[w] * lambda_imb_in[(t, w)] for w in Omega_in)
    for t in T
}

# plot optimal offering strategy
pf.plot_optimal_offering(T, p_DA_optimal, lambda_DA_avg, lambda_imb_avg)

# Calculate expected profit under the optimal offering strategy
expected_profit = tp_model.ObjVal
print(f"\nExpected Profit under Optimal Offering Strategy: {expected_profit:.2f} DKK")

# Compute and plot profit distribution across in-sample scenarios
pf.plot_profit_distribution(T, Omega_in, p_DA_optimal, P_real_in, lambda_DA_in, lambda_imb_in)







#%% ------------------------------------------------------------------------------
#                                      Task 1.3                                
#   ------------------------------------------------------------------------------






#%% ------------------------------------------------------------------------------
#                                      Task 1.4                                 
#   ------------------------------------------------------------------------------


