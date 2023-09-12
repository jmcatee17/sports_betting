import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
This file is a repository of functions that help calculate expected value of a wager.
"""

# Define expected_under and expected_over functions
def expected_under(l, wu, p):
    if l < 0:
        Eu = -wu + p * wu - (100 * p * wu) / l
    else:
        Eu = -wu + p * wu + (p * l * wu) / 100
    return Eu
    
def expected_over(h, wo, p):
    if h < 0:
        Eo = -p * wo - (100 * wo) / h + (p * wo) / h
    else:
        Eo = -p * wo + h * wo / 100 - p * h * wo / 100
    return Eo

# Define your linear function for profitability
def profitability_function(wu, wo, p, l, h):
    Eu = expected_under(l, wu, p)
    Eo = expected_over(h, wo, p)
    
    return Eu + Eo

def arg_max_profitability(wu_bounds, wo_bounds, p_bounds, l, h):
    # Create a meshgrid of wu, wo, and p values
    num_points = 100
    wu_values = np.linspace(wu_bounds[0], wu_bounds[1], num_points)
    wo_values = np.linspace(wo_bounds[0], wo_bounds[1], num_points)
    p_values = np.linspace(p_bounds[0], p_bounds[1], num_points)
    wu_mesh, wo_mesh, p_mesh = np.meshgrid(wu_values, wo_values, p_values, indexing='ij')

    # Calculate profitability for each combination of wu, wo, and p
    profitability_values = profitability_function(wu_mesh, wo_mesh, p_mesh, l = l_input, h = h_input)

    # Find the maximum point of profitability
    max_index = np.unravel_index(np.argmax(profitability_values), profitability_values.shape)
    max_wu = wu_mesh[max_index]
    max_wo = wo_mesh[max_index]
    max_p = p_mesh[max_index]
    max_profitability = profitability_values[max_index]

    print("Maximum Point of Profitability:")
    print("wu:", max_wu)
    print("wo:", max_wo)
    print("p:", max_p)
    print("Maximum Profitability:", max_profitability)

    return max_wu, max_wo, max_p, max_profitability

def profitability_graph(wu_bounds, wo_bounds, p_bounds, l, h):
    # Create a meshgrid of wu, wo, and p values
    num_points = 100
    wu_values = np.linspace(wu_bounds[0], wu_bounds[1], num_points)
    wo_values = np.linspace(wo_bounds[0], wo_bounds[1], num_points)
    p_values = np.linspace(p_bounds[0], p_bounds[1], num_points)
    wu_mesh, wo_mesh, p_mesh = np.meshgrid(wu_values, wo_values, p_values, indexing='ij')

    # Define profitability values
    profitability_values = profitability_function(wu_mesh, wo_mesh, p_mesh, l = l_input, h = h_input)

    # Identify the indices of profitable outcomes
    profitable_indices = profitability_values > 0

    # Extract only the profitable combinations
    profitable_wu = wu_mesh[profitable_indices]
    profitable_wo = wo_mesh[profitable_indices]
    profitable_p = p_mesh[profitable_indices]
    profitable_profitability = profitability_values[profitable_indices]

    # Create a 3D scatter plot of profitable combinations
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(profitable_wu, profitable_wo, profitable_p, c=profitable_profitability, cmap='viridis', marker='o')
    plt.colorbar(scatter, label='Profitability')

    # Set labels and title
    ax.set_xlabel('wu')
    ax.set_ylabel('wo')
    ax.set_zlabel('p')
    plt.title('Profitable Combinations 3D Scatter Plot')

    # Show the plot
    plt.show()


# Set the bounds for wu, wo, and p
wu_bounds = (0, 100)  # Example bounds for wu
wo_bounds = (0, 100)  # Example bounds for wo
p_bounds = (0.4, 0.8)  # Example bounds for p
l_input = -150
h_input = 180

# arg_max_profitability(wu_bounds = wu_bounds, wo_bounds = wo_bounds, p_bounds = p_bounds, l = l_input, h = h_input)
profitability_graph(wu_bounds = wu_bounds, wo_bounds = wo_bounds, p_bounds = p_bounds, l = l_input, h = h_input)