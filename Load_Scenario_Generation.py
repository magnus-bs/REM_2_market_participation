"""
Synthetic load scenario generation for Assignment 2, Step 2.

This script creates feasible stochastic baseline load trajectories for one
bidding hour (minute-level resolution) in the DK2 FCR-D UP problem.
The generated data is synthetic and not real measured load data.

Purpose in the assignment:
- Build feasible stochastic scenarios for Step 2 optimization and validation.
- Keep generation simple and transparent for group work.
- Separate scenario generation from bidding logic (ALSO-X/CVaR are not here).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# Parameters
# =============================================================================
N_TOTAL = 300
N_IN = 100
N_OUT = 200
N_MINUTES = 60

LOAD_MIN = 220.0
LOAD_MAX = 600.0
MAX_DELTA = 35.0

RANDOM_SEED = 42
# Slightly narrower initial range and centered stochastic deltas give paths that
# are still random but less "jumpy" than uniform full-range initialization.
INITIAL_LOAD_MIN = 280.0
INITIAL_LOAD_MAX = 540.0
DELTA_STD = 14.0

DATA_DIR = Path("Data")
MASTER_FILE = DATA_DIR / "load_profiles.csv"
IN_SAMPLE_FILE = DATA_DIR / "load_profiles_in_sample.csv"
OUT_SAMPLE_FILE = DATA_DIR / "load_profiles_out_of_sample.csv"

# Nice-to-have: upward flexibility export (F_up = load - 220)
FLEX_FILE = DATA_DIR / "load_profiles_f_up.csv"

# Plot output
PLOT_FILE = DATA_DIR / "load_profiles_examples.png"
N_EXAMPLE_PLOTS = 10


def generate_single_profile(
    rng: np.random.Generator,
    n_minutes: int = N_MINUTES,
    load_min: float = LOAD_MIN,
    load_max: float = LOAD_MAX,
    max_delta: float = MAX_DELTA,
) -> np.ndarray:
    """
    Generate one feasible 60-minute load profile under all constraints.

    Modeling choice:
    - Start from a central load band (not full [LOAD_MIN, LOAD_MAX]) to reduce
      immediate boundary saturation in synthetic trajectories.
    - Use centered normal increments, clipped to +/- MAX_DELTA, to keep most
      minute steps moderate while still allowing occasional larger moves.
    """
    profile = np.empty(n_minutes, dtype=float)
    profile[0] = rng.uniform(INITIAL_LOAD_MIN, INITIAL_LOAD_MAX)

    for t in range(1, n_minutes):
        delta = float(np.clip(rng.normal(loc=0.0, scale=DELTA_STD), -max_delta, max_delta))
        candidate = profile[t - 1] + delta
        profile[t] = np.clip(candidate, load_min, load_max)

    return profile


def generate_profiles(
    n_profiles: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a matrix with shape (n_profiles, N_MINUTES)."""
    profiles = np.vstack([generate_single_profile(rng) for _ in range(n_profiles)])
    return profiles


def validate_profiles(
    profiles: np.ndarray,
    n_total: int = N_TOTAL,
    n_minutes: int = N_MINUTES,
    load_min: float = LOAD_MIN,
    load_max: float = LOAD_MAX,
    max_delta: float = MAX_DELTA,
) -> None:
    """Validate dimensions, NaNs, bounds, and step-size constraints."""
    assert profiles.shape == (n_total, n_minutes), (
        f"Expected shape {(n_total, n_minutes)}, got {profiles.shape}."
    )
    assert not np.isnan(profiles).any(), "Profiles contain NaN values."
    assert np.all(profiles >= load_min), "Some load values are below LOAD_MIN."
    assert np.all(profiles <= load_max), "Some load values are above LOAD_MAX."

    deltas = np.diff(profiles, axis=1)
    assert np.all(np.abs(deltas) <= max_delta + 1e-10), (
        "Some consecutive minute changes exceed MAX_DELTA."
    )


def build_master_dataframe(profiles: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    """
    Create the scenario dataframe with metadata.

    We shuffle profiles before assigning sample labels so the in/out split is a
    random partition of one common scenario pool (cleaner for out-of-sample
    interpretation than labeling by generation order).
    """
    minute_cols = [f"m{i}" for i in range(1, N_MINUTES + 1)]
    shuffled_profiles = profiles[rng.permutation(len(profiles))]
    df = pd.DataFrame(shuffled_profiles, columns=minute_cols)
    df.insert(0, "sample_type", ["in_sample"] * N_IN + ["out_of_sample"] * N_OUT)
    df.insert(0, "scenario_id", np.arange(1, N_TOTAL + 1, dtype=int))
    return df


def validate_split(df: pd.DataFrame) -> None:
    """Validate exact in/out sample counts."""
    assert len(df) == N_TOTAL, f"Expected {N_TOTAL} rows, got {len(df)}."
    counts = df["sample_type"].value_counts()
    assert counts.get("in_sample", 0) == N_IN, (
        f"Expected {N_IN} in-sample profiles, got {counts.get('in_sample', 0)}."
    )
    assert counts.get("out_of_sample", 0) == N_OUT, (
        f"Expected {N_OUT} out-of-sample profiles, got {counts.get('out_of_sample', 0)}."
    )


def save_outputs(df: pd.DataFrame) -> None:
    """Save master and convenience CSV files."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(MASTER_FILE, index=False)
    df[df["sample_type"] == "in_sample"].to_csv(IN_SAMPLE_FILE, index=False)
    df[df["sample_type"] == "out_of_sample"].to_csv(OUT_SAMPLE_FILE, index=False)


def save_flexibility_output(df: pd.DataFrame) -> None:
    """
    Save minute-level upward flexibility: F_up = load - LOAD_MIN.

    Rationale: LOAD_MIN is the technical lower load bound, so any load above it
    can, in principle, be reduced to create upward reserve activation headroom.
    """
    minute_cols = [f"m{i}" for i in range(1, N_MINUTES + 1)]
    flex_cols = [f"F_up_{i}" for i in range(1, N_MINUTES + 1)]

    flex_values = df[minute_cols].to_numpy() - LOAD_MIN
    assert not np.isnan(flex_values).any(), "Upward flexibility contains NaN values."
    assert np.all(flex_values >= -1e-10), "Upward flexibility must be non-negative."
    flex_df = pd.DataFrame(flex_values, columns=flex_cols)
    flex_df.insert(0, "sample_type", df["sample_type"].values)
    flex_df.insert(0, "scenario_id", df["scenario_id"].values)
    flex_df.to_csv(FLEX_FILE, index=False)


def plot_example_profiles(df: pd.DataFrame, n_examples: int = N_EXAMPLE_PLOTS) -> None:
    """Plot a handful of generated profiles for quick visual inspection."""
    minute_cols = [f"m{i}" for i in range(1, N_MINUTES + 1)]
    x = np.arange(1, N_MINUTES + 1)
    subset = df.head(min(n_examples, len(df)))

    plt.figure(figsize=(10, 6))
    for _, row in subset.iterrows():
        plt.plot(x, row[minute_cols].to_numpy(), alpha=0.75, linewidth=1.5)

    plt.title("Example Synthetic Load Profiles (Assignment 2, Step 2)")
    plt.xlabel("Minute")
    plt.ylabel("Load demand [kW]")
    plt.ylim(LOAD_MIN - 5, LOAD_MAX + 5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150)
    plt.show()


def print_summary(df: pd.DataFrame) -> None:
    """Print a short generation summary."""
    minute_cols = [f"m{i}" for i in range(1, N_MINUTES + 1)]
    values = df[minute_cols].to_numpy()
    diffs = np.diff(values, axis=1)

    print("Synthetic load scenario generation complete.")
    print(f"Total profiles: {len(df)}")
    print(f"In-sample: {(df['sample_type'] == 'in_sample').sum()}")
    print(f"Out-of-sample: {(df['sample_type'] == 'out_of_sample').sum()}")
    print(f"Minutes per profile: {N_MINUTES}")
    print(f"Observed load range: [{values.min():.2f}, {values.max():.2f}] kW")
    print(f"Max absolute minute-to-minute change: {np.abs(diffs).max():.2f} kW")
    print(f"Saved: {MASTER_FILE}")
    print(f"Saved: {IN_SAMPLE_FILE}")
    print(f"Saved: {OUT_SAMPLE_FILE}")
    print(f"Saved: {FLEX_FILE}")
    print(f"Saved plot: {PLOT_FILE}")


def main() -> None:
    """Generate, validate, export, and visualize synthetic load scenarios."""
    rng = np.random.default_rng(RANDOM_SEED)

    profiles = generate_profiles(n_profiles=N_TOTAL, rng=rng)
    validate_profiles(profiles)

    df = build_master_dataframe(profiles, rng)
    validate_split(df)
    save_outputs(df)
    save_flexibility_output(df)
    plot_example_profiles(df)
    print_summary(df)


if __name__ == "__main__":
    main()
