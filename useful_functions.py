
import os
import pandas as pd
import numpy as np

def get_season(month):
    # Fct to assign season based on month
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"

def wind_scenario_generation(FARM_CAPACITY_MW, N_DAYS, data_folder='Data'):
    """
    Generates wind power scenarios based on historical data, normalizes them to a specified farm capacity, 
    and samples a specified number of days per season. 
    The generated scenarios are saved to a CSV file for future use.
    """

    # Return if csv already exists
    if os.path.exists(os.path.join(data_folder, 'wind_scenarios.csv')):
        print("Wind scenarios already generated. Loading from file.")
        return pd.read_csv(os.path.join(data_folder, 'wind_scenarios.csv'))

    # Load data on power and capacities
    df_wind_pwr = pd.read_excel(os.path.join(data_folder, 'wind_pwr.xlsx')) # Wind power data
    df_wind_cap = pd.read_csv(os.path.join(data_folder, 'wind_cap.csv'), delimiter=';') # Wind power data

    # Parse timestamps as UTC and convert to Danish time
    df_wind_pwr["startTime"] = pd.to_datetime(df_wind_pwr["startTime"], utc=True).dt.tz_convert("Europe/Copenhagen")
    df_wind_cap["startTime"] = pd.to_datetime(df_wind_cap["startTime"], utc=True).dt.tz_convert("Europe/Copenhagen")

    # Set index and resample to hourly mean
    df_wind_pwr_h = (
        df_wind_pwr.set_index("startTime")["Wind power production - real-time data"]
        .resample("h")
        .mean()
        .reset_index()
        .rename(columns={"startTime": "TimeDK", "Wind power production - real-time data": "power_MW"})
    )
    df_wind_cap_h = (
        df_wind_cap.set_index("startTime")["Total production capacity used in the wind power forecast"]
        .resample("h")
        .mean()
        .reset_index()
        .rename(columns={"startTime": "TimeDK", "Total production capacity used in the wind power forecast": "capacity_MW"})
    )

    # Checked missing data, not significant
    # Filter out dates where any hour has negative power
    invalid_dates = df_wind_pwr_h.loc[df_wind_pwr_h['power_MW'] < 0, 'TimeDK'].dt.date.unique()
    df_wind_pwr_h = df_wind_pwr_h[~df_wind_pwr_h['TimeDK'].dt.date.isin(invalid_dates)]

    # Merge power and capacity data
    df_wind_h = df_wind_pwr_h.merge(df_wind_cap_h, on="TimeDK", how="left")

    # Normalise:
    df_wind_h["power_norm_MW"] = (df_wind_h["power_MW"] / df_wind_h["capacity_MW"]) * FARM_CAPACITY_MW

    # Randomly Sample 5 days per season
    df_wind_h["date"] = df_wind_h["TimeDK"].dt.date
    df_wind_h["season"] = df_wind_h["TimeDK"].dt.month.map(get_season)
    sampled_days = (
        df_wind_h.groupby("season")["date"]
        .apply(lambda x: pd.Series(x.unique()).sample(n=N_DAYS, random_state=42).values)
        .explode()
        .reset_index()
        .rename(columns={0: "date"})
    )

    # Pivot to get a day per column and hours as rows
    # Merge sampled days back, extract hour, then pivot
    df_wind_scenarios = (
        df_wind_h.merge(sampled_days, on="date")
        .assign(hour=lambda x: x["TimeDK"].dt.hour)
        .groupby(["hour", "date"])["power_norm_MW"].mean()
        .reset_index()
        .pivot(index="hour", columns="date", values="power_norm_MW")
        .rename_axis(index="Hour", columns=None)
        .reset_index()
    )

    # Save to csv for later use
    df_wind_scenarios.to_csv('Data/wind_scenarios.csv', index=False)

    return df_wind_scenarios



def sys_state_scenario_generation(num_scenarios, hours_per_day=24, data_folder='Data'):
    """
    Generates system state scenarios (SI) for a specified number of scenarios and hours per day.
    The generated scenarios are saved to a CSV file for future use.
    """

    # Return if csv already exists
    if os.path.exists(os.path.join(data_folder, 'imbalance_scenarios.csv')):
        print("Imbalance scenarios already generated. Loading from file.")
        return pd.read_csv(os.path.join(data_folder, 'imbalance_scenarios.csv'))

    data = []

    for s in range(1, num_scenarios+1):
        SI = np.random.binomial(1, 0.5, hours_per_day)  # 0 or 1
        for h in range(1, hours_per_day+1):
            data.append([s, h, SI[h-1]])

    df = pd.DataFrame(data, columns=["scenario"], rows=["hour"], data=["SI"])
    df.to_csv(os.path.join(data_folder, "imbalance_scenarios.csv"), index=False)
    
    return df


def imb_price_scenario_generation(da_price_scenarios, df_sys_state_scenarios):
    """
    Generates imbalance price scenarios based on given DA price and system state scenarios.
    """
    return None

    

