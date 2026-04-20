
import matplotlib.pyplot as plt
import numpy as np





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
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Left axis: power
    ax1.step(T, [p_DA_optimal[t] for t in T], where='post', label='Optimal Offering (MW)',color="#118ee7")
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('Power (MW)')
    ax1.set_xticks(T)

    # Right axis: prices
    ax2 = ax1.twinx()
    ax2.step(T, [lambda_DA_avg[t] for t in T], '--', label='DA Price', color="#48ad8c")
    ax2.step(T, [lambda_imb_avg[t] for t in T], '--', label='Imbalance Price', color="#045536")
    ax2.set_ylabel('Price (DKK/MWh)')

    # Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.title('Optimal Offering vs Prices')
    plt.grid()
    plt.show()



def plot_profit_distribution(T, Omega, p_DA_optimal, P_real, lambda_DA, lambda_imb):
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

    for w in Omega:
        profit = 0
        for t in T:
            profit += (
                lambda_DA[(t, w)] * p_DA_optimal[t]
                + lambda_imb[(t, w)] * (P_real[(t, w)] - p_DA_optimal[t])
            )
        profit_scenarios[w] = profit

    # Convert to list
    profits = list(profit_scenarios.values())
    mean_profit = np.mean(profits)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.hist(profits, bins=20)
    plt.axvline(mean_profit, linestyle='--', label=f"Mean = {mean_profit:.2f}", color="#ee9a79")

    plt.xlabel("Profit (DKK)")
    plt.ylabel("Frequency")
    plt.title("Profit Distribution Across Scenarios")
    plt.legend()
    plt.grid()
    plt.show()

    return None