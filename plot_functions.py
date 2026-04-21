
import matplotlib.pyplot as plt
import numpy as np


def plot_three_panel_scenarios(df_wind_scenarios,
                               df_price_scenarios,
                               df_imb_scenarios,
                               figsize=(10, 2),
                               dpi=300):

    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi)

    ax1, ax2, ax3 = axes

    # -------------------------
    # (a) WIND SCENARIOS
    # -------------------------
    for col in df_wind_scenarios.columns:
        ax1.plot(df_wind_scenarios.index,
                 df_wind_scenarios[col],
                 color='skyblue',
                 alpha=0.7,
                 linewidth=1)

    ax1.set_title("(a) Wind Scenarios")
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Power (MW)")
    ax1.grid(alpha=0.3)

    # -------------------------
    # (b) PRICE SCENARIOS
    # -------------------------
    T = df_price_scenarios.index
    T_plot = list(T) + [T[-1] + 1]
    for col in df_price_scenarios.columns:
        y = df_price_scenarios[col].values
        y_plot = list(y) + [y[-1]]  # repeat last value

        ax2.step(
            T_plot,
            y_plot,
            where="post",
            color='coral',
            alpha=0.6,
            linewidth=1
        )

    ax2.set_title("(b) Price Scenarios")
    ax2.set_xlabel("Hour")
    ax2.set_ylabel("EUR/MWh")
    ax2.grid(alpha=0.3)

    # -------------------------
    # (c) IMBALANCE HEATMAP
    # -------------------------
    im = ax3.imshow(
        df_imb_scenarios.values.T,
        aspect='auto',
        cmap='BuPu',
        interpolation='nearest',
        alpha = 0.4
    )

    ax3.set_title("(c) Imbalance Scenarios")
    ax3.set_xlabel("Hour")
    ax3.set_ylabel("Scenario")

    ax3.set_yticks(np.arange(df_imb_scenarios.shape[1]))
    ax3.set_yticklabels([f"ω{i+1}" for i in range(df_imb_scenarios.shape[1])])

    # Colorbar
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#f0f0f0", label="(0)"),
        Patch(facecolor="#c26bcea1", label="(1)")
    ]

    ax3.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        ncol=1
    )

    # -------------------------
    # FINAL FORMATTING
    # -------------------------
    for ax in [ax1, ax2, ax3]:
        ax.set_xticks(range(0,25,1))
        ax.set_xticklabels(
            [str(i) if i % 4 == 0 else "" for i in range(25)]
        )

    plt.tight_layout()
    plt.show()


def plot_optimal_offering(T, p_DA_optimal, lambda_DA_avg, lambda_imb_avg):
    
    '''Plots the optimal day-ahead offering strategy.
    Parameters:
    -----------
    T (list): List of hours in the day.
    p_DA_optimal (dict): Dictionary containing the optimal day-ahead offering for each hour.
    lambda_DA_avg (float): Average day-ahead price.
    lambda_imb_avg (float): Average imbalance price.

    Returns:
    -----------
    None: Displays a step plot of the optimal day-ahead offering strategy.
    '''
    T_steps = list(T) + [T[-1] + 1]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Left axis: power
    ax1.step(T_steps, [p_DA_optimal[t] for t in T] + [p_DA_optimal[T[-1]]], 
             where='pre', label='Optimal Offering (MW)', color="#118ee7")
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('Power (MW)')
    ax1.set_xticks(T_steps)

    # Right axis: prices
    ax2 = ax1.twinx()
    ax2.step(T_steps, [lambda_DA_avg[t] for t in T] + [lambda_DA_avg[T[-1]]], 
             '--', label='DA Price', color="#48ad8c")
    ax2.step(T_steps, [lambda_imb_avg[t] for t in T] + [lambda_imb_avg[T[-1]]], 
             '--', label='Imbalance Price', color="#045536")
    ax2.set_ylabel('Price (EUR/MWh)')

    # Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.title('Optimal Offering vs Prices')
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_optimal_offering_prob(T, Omega_in, p_DA_optimal, y_imb):
    '''Plots the optimal day-ahead offering strategy.
    Parameters:
    -----------
    T (list): List of hours in the day.
    Omega_in (list): List of scenarios.
    p_DA_optimal (dict): Dictionary containing the optimal day-ahead offering for each hour.
    y_imb (dict): Dictionary containing imbalance decisions for each (t, w) pair.

    Returns:
    -----------
    None: Displays a step plot of the optimal day-ahead offering strategy.
    '''
    T_steps = list(T) + [T[-1] + 1]

    prob = {}
    for t in T:
        prob[t] = np.mean([y_imb[(t, w)] for w in Omega_in])

    fig, ax1 = plt.subplots(figsize=(10, 3))

    # Left axis: power 
    # (we add a starting and end position)
    T_plot = [T[0]] + list(T) + [T[-1] + 1]
    p_plot = [0] + [p_DA_optimal[t] for t in T] + [0]

    ax1.step(
        T_plot,
        p_plot,
        where='post',
        label='Optimal Offering (MW)',
        color="#118ee7"
    )

    # Right axis: probability
    ax2 = ax1.twinx()
    ax2.fill_between(
        T_steps,
        [prob[t] for t in T] + [prob[T[-1]]],
        step='post',
        color="#e7a011",
        alpha=0.15,
        label="P(deficit) per hour"
    )

    ax1.set_ylabel('Power (MW)')
    ax2.set_ylabel('Probability of Deficit')
    ax2.set_ylim(0, 1)
    ax1.set_ylim(0,510)
    ax1.set_xticks(range(0,25,1))
    ax1.set_yticks(range(0,550,100))
    ax1.set_xlabel("Hour of the Day")
    ax2.axhline(y=0.375, color="#ca7900", linestyle='-', linewidth=1.2, label='Threshold (p=0.375)')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.35),
        ncol=3,
        frameon=False
    )

    print('Optimal Offering vs Probability of Deficit')
    plt.grid()
    plt.tight_layout()
    plt.show()






def plot_profit_distribution(T, Omega, p_DA_optimal, P_real, lambda_DA, lambda_imb, y_imb, price_scheme):
    """
    Computes and plots the profit distribution across scenarios.

    Parameters:
    -----------
    T : iterable
        Set of time periods (e.g. hours)
    Omega : iterable
        Set of scenarios
    p_DA_optimal : dict
        Optimal day-ahead bids {t: value}
    P_real : dict
        Realized production {(t,w): value}
    lambda_DA : dict
        Day-ahead prices {(t,w): value}
    lambda_imb : dict
        Imbalance prices {(t,w): value}

    Returns:
    --------
    None: Displays a histogram of the profit distribution across scenarios.
    """

    # Compute profit per scenario
    profit_scenarios = {}

    if price_scheme == "one_price":
        for w in Omega:
            profit = 0
            for t in T:
                profit += (
                    lambda_DA[(t, w)] * p_DA_optimal[t]
                    + lambda_imb[(t, w)] * (P_real[(t, w)] - p_DA_optimal[t])
                )
            profit_scenarios[w] = profit

    elif price_scheme == "two_price":

        for w in Omega:
            profit = 0
            for t in T:
                imbalance = P_real[(t, w)] - p_DA_optimal[t]
                Delta_up = max(imbalance, 0)
                Delta_down = max(-imbalance, 0)

                profit += (
                    lambda_DA[(t, w)]   * p_DA_optimal[t]
                    + y_imb[(t,w)]      * (lambda_DA[(t, w)] * Delta_up - lambda_imb[(t,w)] * Delta_down)
                    + (1-y_imb[(t,w)])  * (lambda_imb[(t,w)] * Delta_up - lambda_DA[(t,w)] * Delta_down))
            profit_scenarios[w] = profit
    else:
        "Error: Price Scheme must be either 'one-price' or 'two-price'."
        return None

    # Convert to list
    profits = list(profit_scenarios.values())
    mean_profit = np.mean(profits)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 2.5), dpi=300)
    ax.hist(profits, bins=50, color="#2eabb4", alpha=0.7, edgecolor="white", linewidth=0.3)
    ax.axvline(mean_profit, linestyle='--', linewidth=1.5,
               label=f"Mean = {mean_profit:,.0f} EUR", color="#e05c2a")

    ax.set_xlabel("Expected Profit (EUR)")
    ax.set_ylabel("Frequency")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.legend(framealpha=0.9)
    ax.grid(alpha=0.3, axis='y', linestyle='--')
    plt.tight_layout()
    plt.show()


