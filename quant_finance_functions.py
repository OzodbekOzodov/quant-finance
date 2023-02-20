
import statsmodels.api as sm

def check_stationarity(data):
    """
    Checks for stationarity of a time series using the Augmented Dickey-Fuller (ADF) test.
    Returns a boolean value indicating whether the data is stationary.
    """
    adf_result = sm.tsa.stattools.adfuller(data)
    p_value = adf_result[1]
    is_stationary = p_value < 0.05
    return is_stationary


import matplotlib.pyplot as plt



def plot_time_series(time, value, title=''):
    """
    Plots a time series given a list of x-values and y-values.

    Args:
    x (list): A list of x-values.
    y (list): A list of y-values.
    xlabel (str): The label for the x-axis (optional).
    ylabel (str): The label for the y-axis (optional).
    title (str): The title of the plot (optional).

    Returns:
    None
    """
    plt.plot(time, value)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(title)
    plt.show()


import numpy as np
import pandas as pd

def time_series_summary(time_series):
    """
    Calculates summary statistics for a time series.

    Args:
    time_series (list or np.array): A time series as a list or np.array.

    Returns:
    dict: A dictionary with summary statistics.
    """
    stats = pd.DataFrame({
        'count': len(time_series),
        'mean': np.mean(time_series),
        'std': np.std(time_series),
        'min': np.min(time_series),
        '25%': np.percentile(time_series, 25),
        '50%': np.percentile(time_series, 50),
        '75%': np.percentile(time_series, 75),
        'max': np.max(time_series)
    }, index=[0])
    
    return stats
