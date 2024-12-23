import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


def load_and_clean_results(rdf_file: str, framework_file: str):
    rdf_df = pd.read_csv(rdf_file)
    framework_df = pd.read_csv(framework_file)

    rdf_df['total_validation_time'] = rdf_df['rdf_conversion_time'] + rdf_df['shacl_validation_time']

    rdf_break = rdf_df.iloc[-1]
    framework_break = framework_df.iloc[-1]

    return rdf_df, framework_df, rdf_break, framework_break


def interpolate_data(rdf_df, framework_df, column):
    log_x = np.log10(rdf_df['size'])
    log_y = np.log10(rdf_df[column])
    interpolator = interp1d(log_x, log_y, fill_value='extrapolate')
    log_x_new = np.linspace(
        np.log10(rdf_df['size'].min()),
        np.log10(framework_df['size'].max()),
        100
    )

    log_y_new = interpolator(log_x_new)
    x_new = 10 ** log_x_new
    y_new = 10 ** log_y_new

    return x_new, y_new, x_new[-1], y_new[-1]


def create_time_comparison_plot(rdf_df, framework_df, rdf_break, framework_break,
                                output_file='time_benchmark_analysis.png'):
    plt.rcParams.update({
        'font.size': 16,
        'font.family': 'serif',
        'axes.labelsize': 20,
        'axes.titlesize': 20,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 16,
        'lines.linewidth': 2,
        'lines.markersize': 8
    })

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(framework_df['size'], framework_df['execution_time'],
            label='Framework', color='green', marker='o',
            markersize=6, linewidth=2, markeredgewidth=1)

    x_interp_total, y_interp_total, final_x_total, final_y_total = interpolate_data(
        rdf_df, framework_df, 'total_validation_time')

    ax.plot(rdf_df['size'], rdf_df['total_validation_time'],
            label='RDF Validator', color='blue', marker='o',
            markersize=6, linewidth=2, markeredgewidth=1)

    ax.plot(x_interp_total, y_interp_total, '--', color='blue',
            label='RDF (Interpolated)', linewidth=2, alpha=0.7)
    ax.plot(rdf_break['size'], rdf_break['total_validation_time'],
            'rx', markersize=12, markeredgewidth=2, label='Break Point')
    ax.plot(framework_break['size'], framework_break['execution_time'],
            'rx', markersize=12, markeredgewidth=2)
    ax.set_xlabel('Number of Rows', fontsize=18, labelpad=10)
    ax.set_ylabel('Time (s)', fontsize=18, labelpad=10)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, which="both", ls="-", alpha=0.2)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)

    ax.annotate(f'RDFValidator (Interpolated):\n'
                f'Number of rows: {final_x_total:,.0f}\n'
                f'Total Time: {final_y_total:.2f}s',
                xy=(final_x_total, final_y_total),
                xytext=(25, 0.9), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.9),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                fontsize=12)

    ax.annotate(f'RDFValidator Break: \n'
                f'Number of rows: {rdf_break["size"]:,}\n'
                f'Total Time: {rdf_break["total_validation_time"]:.2f}s\n'
                f'SHACL Time: {rdf_break["shacl_validation_time"]:.2f}s\n'
                f'RDF Conversion Time: {rdf_break["rdf_conversion_time"]:.2f}s',
                xy=(rdf_break['size'], rdf_break['total_validation_time']),
                xytext=(120, 0.7), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.9),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                fontsize=12)

    ax.annotate(f'Framework Break:\n'
                f'Number of rows: {framework_break["size"]:,}\n'
                f'Total Time: {framework_break["execution_time"]:.2f}s',
                xy=(framework_break['size'], framework_break['execution_time']),
                xytext=(25, 0.4), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.9),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                fontsize=12)

    # Add legend with clean styling
    ax.legend(bbox_to_anchor=(1.03, 0.2),
              loc='center left',
              borderaxespad=0,
              frameon=True,
              fancybox=False,
              edgecolor='black',
              fontsize=14)

    plt.tight_layout()
    plt.savefig(output_file,
                bbox_inches='tight',
                dpi=300,
                facecolor='white',
                edgecolor='none')
    plt.close()

if __name__ == "__main__":
    try:
        rdf_df, framework_df, rdf_break, framework_break = load_and_clean_results(
            'shacl_benchmark_results.csv',
            'framework_benchmark_results.csv'
        )

        create_time_comparison_plot(rdf_df, framework_df, rdf_break, framework_break)

        print("\nVisualization saved as 'time_benchmark_analysis.png'")

    except Exception as e:
        print(f"Error creating visualization: {str(e)}")
        raise