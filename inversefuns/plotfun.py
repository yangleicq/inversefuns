"""
This module contains functions for plotting
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_cdf(data):
    sorted_data = np.sort(data)
    plt.plot(sorted_data, np.linspace(0,1,sorted_data.size),lw=2)