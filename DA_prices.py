import pandas as pd

INPUT_CSV = "GUI_ENERGY_PRICES_202312312300-202412312300.csv"
OUTPUT_CSV = "DK2_selected_20_days_seasonal.csv"
RANDOM_SEED = 42
SAMPLES_PER_SEASON = 5


def season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def main() -> None:
    df = pd.read_csv(INPUT_CSV)

    # Keep the day-ahead DK2 rows only.
    df = df[df["Area"] == "BZN|DK2"].copy()
    df["price"] = pd.to_numeric(df["Day-ahead Price (EUR/MWh)"], errors="coerce")
    df = df.dropna(subset=["price"])

    # Parse interval start as timestamp from "MTU (CET/CEST)".
    df["start_time"] = pd.to_datetime(
        df["MTU (CET/CEST)"].str.split(" - ").str[0],
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    df = df.dropna(subset=["start_time"])

    df["date"] = df["start_time"].dt.date
    df["hour"] = df["start_time"].dt.hour
    df["month"] = df["start_time"].dt.month
    df["season"] = df["month"].map(season_from_month)

    # Keep complete 24h days only.
    daily_counts = df.groupby("date")["hour"].nunique()
    valid_days = daily_counts[daily_counts == 24].index
    df = df[df["date"].isin(valid_days)].copy()

    # Randomly sample 5 unique days per season.
    unique_days = (
        df[["date", "season"]]
        .drop_duplicates()
        .sort_values(["season", "date"])
    )

    selected_days = []
    for season in ["winter", "spring", "summer", "autumn"]:
        season_days = unique_days[unique_days["season"] == season]
        if len(season_days) < SAMPLES_PER_SEASON:
            raise ValueError(
                f"Not enough full days in {season}. "
                f"Found {len(season_days)}, need {SAMPLES_PER_SEASON}."
            )
        sampled = season_days.sample(
            n=SAMPLES_PER_SEASON,
            random_state=RANDOM_SEED,
            replace=False,
        )
        selected_days.append(sampled)

    selected_days_df = pd.concat(selected_days, ignore_index=True)
    selected_days_df = selected_days_df.sort_values(["season", "date"])

    # Join selected days back to hourly prices and export one CSV.
    result = df.merge(selected_days_df, on=["date", "season"], how="inner")
    result = result.sort_values(["season", "date", "hour"])
    result = result[["season", "date", "hour", "price"]]

    result.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved {len(selected_days_df)} days ({len(result)} hourly rows) to {OUTPUT_CSV}")
    print(selected_days_df.groupby("season")["date"].count())


if __name__ == "__main__":
    main()
