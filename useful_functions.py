
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

def wind_scenario_generation(SCENARIOS_PER_SEASON, FARM_CAPACITY_MW, data_folder='Data'):
    """
    Generates wind power scenarios based on historical data, normalizes them to a specified farm capacity, 
    and samples a specified number of days per season. 
    The generated scenarios are saved to a CSV file for future use.
    """

    RANDOM_SEED = 42

    # Return if csv already exists
    if os.path.exists(os.path.join(data_folder, 'wind_scenarios.csv')):
        print("Wind scenarios already generated. Loading from file.")
        return pd.read_csv(os.path.join(data_folder, 'wind_scenarios.csv'), index_col="Hour")

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
        .apply(lambda x: pd.Series(x.unique()).sample(n=SCENARIOS_PER_SEASON, random_state=RANDOM_SEED).values)
        .explode()
        .reset_index()
        .rename(columns={0: "date"})
    )

    # Pivot to get a day per column and hours as rows
    # Merge sampled days back, extract hour, then pivot
    df_scenarios = (
        df_wind_h.merge(sampled_days, on="date")
        .assign(hour=lambda x: x["TimeDK"].dt.hour)
        .groupby(["hour", "date"])["power_norm_MW"].mean()
        .reset_index()
        .pivot(index="hour", columns="date", values="power_norm_MW")
        .rename_axis(index="Hour", columns=None)
        .reset_index()
    )

    # Set hour as index:
    df_scenarios.set_index("Hour", inplace=True)

    # Save to csv for later use
    df_scenarios.to_csv('Data/wind_scenarios.csv', index=True)

    print(f"Saved {len(df_scenarios.columns)} days to {os.path.join(data_folder, 'wind_scenarios.csv')}")
    print(sampled_days.groupby("season")["date"].count())

    return df_scenarios



def price_scenario_generation(SCENARIOS_PER_SEASON = 5, data_folder='Data'):

    RANDOM_SEED = 42

    # Return if csv already exists
    if os.path.exists(os.path.join(data_folder, 'price_scenarios.csv')):
        print("DA price scenarios already generated. Loading from file.")
        return pd.read_csv(os.path.join(data_folder, 'price_scenarios.csv'), index_col="Hour")

    # Load, filter, parse
    df = (
        pd.read_csv(os.path.join(data_folder, 'spotprices.csv'))
        .pipe(lambda x: x[x["Area"] == "BZN|DK2"])
        .assign(
            price=lambda x: pd.to_numeric(x["Day-ahead Price (EUR/MWh)"], errors="coerce"),
            TimeDK=lambda x: pd.to_datetime(
                x["MTU (CET/CEST)"].str.split(" - ").str[0],
                format="%d/%m/%Y %H:%M:%S", errors="coerce"
            ).dt.tz_localize("Europe/Copenhagen", ambiguous="infer", nonexistent="shift_forward"),
        )
        .dropna(subset=["price", "TimeDK"])
        .assign(
            date=lambda x: x["TimeDK"].dt.date,
            hour=lambda x: x["TimeDK"].dt.hour,
            season=lambda x: x["TimeDK"].dt.month.map(get_season),
        )
    )

    # Keep complete 24h days only
    valid_days = df.groupby("date")["hour"].nunique()
    df = df[df["date"].isin(valid_days[valid_days == 24].index)]

    # Randomly sample 5 unique days per season.
    unique_days = df[["date", "season"]].drop_duplicates().sort_values(["season", "date"])

    selected_days = []
    for season in ["Winter", "Spring", "Summer", "Autumn"]:
        season_days = unique_days[unique_days["season"] == season]
        if len(season_days) < SCENARIOS_PER_SEASON:
            raise ValueError(f"Not enough full days in {season}: found {len(season_days)}, need {SCENARIOS_PER_SEASON}.")
        selected_days.append(season_days.sample(n=SCENARIOS_PER_SEASON, random_state=RANDOM_SEED, replace=False))

    sampled_days = pd.concat(selected_days, ignore_index=True)

    df_scenarios = (
        df.merge(sampled_days, on=["date", "season"])
        [["date", "hour", "price"]]
        .pivot(index="hour", columns="date", values="price")
        .rename_axis(index="Hour", columns=None)
    )

    df_scenarios.to_csv(os.path.join(data_folder, 'price_scenarios.csv'), index=True)
    print(f"Saved {len(df_scenarios.columns)} days to {os.path.join(data_folder, 'price_scenarios.csv')}")
    print(sampled_days.groupby("season")["date"].count())

    return df_scenarios


def imbalance_scenario_generation(N_SCENARIOS, hours_per_day=24, data_folder='Data'):

    out_path = os.path.join(data_folder, 'imbalance_scenarios.csv')

    if os.path.exists(out_path):
        print("Imbalance scenarios already generated. Loading from file.")
        return pd.read_csv(out_path, index_col="Hour")

    data = []
    for s in range(1, N_SCENARIOS + 1):
        SI = np.random.binomial(1, 0.5, hours_per_day)
        for h in range(0, hours_per_day):
            data.append([s, h, SI[h - 1]])

    df_scenarios = (
        pd.DataFrame(data, columns=["scenario", "hour", "SI"])
        .pivot(index="hour", columns="scenario", values="SI")
        .rename_axis(index="Hour", columns=None)
    )

    df_scenarios.to_csv(out_path, index=True)

    return df_scenarios


def build_parameters(Omega_set, df_wind_scenarios, df_price_scenarios, df_imbalance_scenarios, hours = 24):
    P_real = {}
    lambda_DA = {}
    y_imb = {}

    # Define time periods
    T = range(hours)

    # Set size
    SAMPLE_SIZE = len(Omega_set)

    # Map sampled scenarios to their corresponding values in the dataframes
    for w, (p, pr, im) in enumerate(Omega_set):
        for t in T:
            P_real[(t, w)] = df_wind_scenarios.loc[t, p]
            lambda_DA[(t, w)] = df_price_scenarios.loc[t, pr]
            y_imb[(t, w)] = df_imbalance_scenarios.loc[t, im]

    # Define imbalance prices based on the imbalance direction
    lambda_up = {
        (t, w): 1.25 * lambda_DA[(t, w)]
        for t in T for w in range(SAMPLE_SIZE)
    }

    lambda_down = {
        (t, w): 0.85 * lambda_DA[(t, w)]
        for t in T for w in range(SAMPLE_SIZE)
    }

    lambda_imb = {
        (t, w): lambda_up[(t, w)] if y_imb[(t, w)] > 0 else lambda_down[(t, w)]
        for t in T for w in range(SAMPLE_SIZE)
    }

    # Define equal probabilities for each scenario
    pi = {w: 1 / len(Omega_set) for w in range(SAMPLE_SIZE)}

    return P_real, lambda_DA, y_imb, lambda_imb, pi
    

