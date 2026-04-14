import matplotlib.pyplot as plt





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
