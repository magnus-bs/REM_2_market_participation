
import matplotlib.pyplot as plt
import numpy as np


def plot_scenarios(df_wind_scenarios, df_price_scenarios, figsize=(10, 3.2), dpi=300):
    
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
    ax1.legend(handles=legend_elements, loc='upper left', ncol=2)

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
    #ax1.step(T_steps, [p_DA_optimal[t] for t in T] + [p_DA_optimal[T[-1]]],
     #        where='pre', label='Optimal Offering (MW)', color="#118ee7")

    ax1.step(T_steps, [p_DA_optimal[t] for t in T] + [p_DA_optimal[T[-1]]],
             where='pre', label='Optimal Offering (MW)', color= "#118ee7")
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('Power (MW)')
    ax1.set_xticks(T_steps)

    # Right axis: probability
    ax2 = ax1.twinx()
    ax2.fill_between(
        T_steps,
        [prob[t] for t in T] + [prob[T[-1]]],
        step='pre',
        color="#e7a011",
        alpha=0.15,
        label="P(deficit) per hour"
    )

    ax2.set_ylabel('Probability of Deficit')
    ax2.set_ylim(0, 1)
    ax1.set_ylim(0,500)
    ax1.set_yticks(range(0,550,100))
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


