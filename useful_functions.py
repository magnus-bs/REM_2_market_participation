
import os
import pandas as pd
import numpy as np

def wind_scenario_generation(N_SCENARIOS, FARM_CAPACITY_MW, data_folder='Data', RANDOM_SEED = 42):
    """
    Generates wind power scenarios based on historical data, normalizes them to a specified farm capacity, 
    divides the year into N_SCENARIOS parts and samples a day randomly from each. 
    The generated scenarios are saved to a CSV file for future use.
    """

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

    # Divide the year into N_SCENARIOS periods and sample a day randomly from each
    df_wind_h["date"] = df_wind_h["TimeDK"].dt.date
    all_dates = pd.Series(df_wind_h["date"].unique())
    all_dates = all_dates.sort_values().reset_index(drop=True)
    all_dates_df = pd.DataFrame({"date": all_dates})
    all_dates_df["period"] = pd.cut(all_dates_df.index, bins=N_SCENARIOS, labels=False)

    sampled_days = (
        all_dates_df.groupby("period")["date"]
        .apply(lambda x: x.sample(n=1, random_state=RANDOM_SEED))
        .reset_index(drop=True)
        .to_frame()
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

    return df_scenarios



def price_scenario_generation(N_SCENARIOS, data_folder='Data', RANDOM_SEED = 42):
    """
    Generates day-ahead price scenarios based on historical data, samples a day per period with the data divided into N_SCENARIOS periods,
    and saves the generated scenarios to a CSV file for future use.
    """

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
        )
    )

    # Keep complete 24h days only
    valid_days = df.groupby("date")["hour"].nunique()
    df = df[df["date"].isin(valid_days[valid_days == 24].index)]

    # Filter out days with negative prices
    neg_price_dates = df.loc[df['price'] < 0, 'TimeDK'].dt.date.unique()
    df = df[~df['TimeDK'].dt.date.isin(neg_price_dates)]

    # Divide into 20 periods and sample 1 day from each
    all_dates_df = pd.DataFrame({"date": sorted(df["date"].unique())}).reset_index(drop=True)
    all_dates_df["period"] = pd.cut(all_dates_df.index, bins=N_SCENARIOS, labels=False)
    sampled_days = (
        all_dates_df.groupby("period")["date"]
        .apply(lambda x: x.sample(n=1, random_state=RANDOM_SEED))
        .reset_index(drop=True)
        .to_frame()
    )

    df_scenarios = (
        df.merge(sampled_days, on=["date"])
        [["date", "hour", "price"]]
        .pivot(index="hour", columns="date", values="price")
        .rename_axis(index="Hour", columns=None)
    )

    df_scenarios.to_csv(os.path.join(data_folder, 'price_scenarios.csv'), index=True)
    print(f"Saved {len(df_scenarios.columns)} days to {os.path.join(data_folder, 'price_scenarios.csv')}")

    return df_scenarios


def imbalance_scenario_generation(N_SCENARIOS, hours_per_day=24, data_folder='Data'):
    """
    Generates imbalance scenarios by simulating a binary imbalance signal (SI) for each 
    hour of the day across a specified number of scenarios.
    The generated scenarios are saved to a CSV file for future use.
    """

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
    

