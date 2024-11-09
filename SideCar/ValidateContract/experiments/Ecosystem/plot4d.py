import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

def create_enhanced_4d_visualization(results_df, save_path='4d_analysis_enhanced.png'):
    """
    Create an enhanced 4D visualization using surface plots with improved aesthetics
    - X axis: Number of Data Products
    - Y axis: Execution Time
    - Z axis: Number of Operations
    - Color: Number of Policies
    """
    # Set style
    plt.style.use('seaborn')

    # Create figure with higher resolution
    fig = plt.figure(figsize=(15, 12), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    # Create meshgrid with more points for smoother surface
    x = np.unique(results_df['data_products'])
    y = np.unique(results_df['execution_time'])
    X, Y = np.meshgrid(x, y)

    # Create Z (operations) and C (policies) matrices
    Z = np.zeros_like(X)
    C = np.zeros_like(X)

    # Fill Z and C matrices with interpolation for smoother surface
    for i, x_val in enumerate(x):
        for j, y_val in enumerate(y):
            mask = (results_df['data_products'] == x_val) & (
                np.isclose(results_df['execution_time'], y_val, rtol=1e-10))
            if mask.any():
                Z[j, i] = results_df.loc[mask, 'operations'].iloc[0]
                C[j, i] = results_df.loc[mask, 'policies'].iloc[0]

    # Create surface plot with enhanced aesthetics
    norm = plt.Normalize(C.min(), C.max())
    surf = ax.plot_surface(X, Y, Z,
                           facecolors=plt.cm.viridis(norm(C)),
                           alpha=0.9,
                           antialiased=True,
                           rstride=1,
                           cstride=1)

    # Enhance the grid
    ax.grid(True, linestyle='--', alpha=0.6)

    # Customize axes
    ax.set_xlabel('Number of Data Products', fontsize=12, labelpad=15)
    ax.set_ylabel('Execution Time (s)', fontsize=12, labelpad=15)
    ax.set_zlabel('Number of Operations', fontsize=12, labelpad=15)

    # Rotate the view for better perspective
    ax.view_init(elev=20, azim=45)

    # Add title with enhanced styling
    plt.title('Data Ecosystem Simulation\nRelationship between Products, Time, Operations, and Policies',
              fontsize=14, pad=20, fontweight='bold')

    # Add color bar with better formatting
    m = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    cbar = plt.colorbar(m, ax=ax, pad=0.1, aspect=30)
    cbar.set_label('Number of Policies', fontsize=12, fontweight='bold', labelpad=15)
    cbar.ax.tick_params(labelsize=10)

    # Add grid lines for better depth perception
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Customize tick parameters
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='z', labelsize=10)

    # Adjust layout and save with high DPI
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)
    plt.close()

    return fig

# Example usage:
results_df = pd.read_csv('results.csv')
create_enhanced_4d_visualization(results_df, save_path='4d_analysis_enhanced.png')