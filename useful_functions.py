

import os
import pandas as pd
import numpy as np


def load_data(data_folder, GENERATORS, WIND_GENERATORS, CONV_GENERATORS, P_wind, C_wind, w_nodes):
    
    # -------------------- Load raw data -------------------- #
    df_demand = pd.read_excel(os.path.join(data_folder, 'demand.xlsx')) # Demand data, total and per node per hour
    df_gen_tech = pd.read_excel(os.path.join(data_folder, 'generator_technical_data.xlsx')) # generator technical data
    df_gen_cost = pd.read_excel(os.path.join(data_folder, 'generator_costs_initial_state.xlsx')) # generator cost data
    df_cf_wind = pd.read_csv(os.path.join(data_folder, 'cf_wind.csv')) # wind capacity factors per hour
    df_trans = pd.read_excel(os.path.join(data_folder, 'transmission_data.xlsx')) # generator technical data
