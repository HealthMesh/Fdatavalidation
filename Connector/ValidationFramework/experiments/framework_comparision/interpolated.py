import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


def load_and_clean_results(shacl_file: str, framework_file: str):
    shacl_df = pd.read_csv(shacl_file)
    framework_df = pd.read_csv(framework_file)

    shacl_break = shacl_df.iloc[-1]
    framework_break = framework_df.iloc[-1]

    return shacl_df, framework_df, shacl_break, framework_break


def interpolate_shacl_data(shacl_df, framework_df):

    log_x = np.log10(shacl_df['size'])
    log_y = np.log10(shacl_df['execution_time'])
    interpolator = interp1d(log_x, log_y, fill_value='extrapolate')
    log_x_new = np.linspace(
        np.log10(shacl_df['size'].min()),
        np.log10(framework_df['size'].max()),
        100
    )

    log_y_new = interpolator(log_x_new)
    x_new = 10 ** log_x_new
    y_new = 10 ** log_y_new

    return x_new, y_new, x_new[-1], y_new[-1]


def create_time_comparison_plot(shacl_df, framework_df, shacl_break, framework_break,
                                output_file='time_benchmark_analysis.png'):
    plt.figure(figsize=(12, 8))
    plt.plot(framework_df['size'], framework_df['execution_time'],
             label='Validation Framework', color='green', marker='o', markersize=4, linewidth=2)
    x_interp, y_interp, final_x, final_y = interpolate_shacl_data(shacl_df, framework_df)
    plt.plot(shacl_df['size'], shacl_df['execution_time'],
             label='RDF Validation', color='blue', marker='o', markersize=4, linewidth=2)
    plt.plot(x_interp, y_interp, '--', color='blue', label='(Interpolated)',
             linewidth=2, alpha=0.7)
    plt.plot(shacl_break['size'], shacl_break['execution_time'],
             'rx', markersize=15, markeredgewidth=3)
    plt.plot(framework_break['size'], framework_break['execution_time'],
             'rx', markersize=15, markeredgewidth=3)
    plt.xlabel('Number of Rows', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(fontsize=10)
    plt.annotate(f'RDF Validation Break:\n{shacl_break["size"]:,} rows\n{shacl_break["execution_time"]:.2f}s',
                 xy=(shacl_break['size'], shacl_break['execution_time']),
                 xytext=(30, -30), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    plt.annotate(f'Framework Break:\n{framework_break["size"]:,} rows\n{framework_break["execution_time"]:.2f}s',
                 xy=(framework_break['size'], framework_break['execution_time']),
                 xytext=(-100, 30), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    plt.annotate(f'Interpolated:\n{final_x:,.0f} rows\n{final_y:.2f}s',
                 xy=(final_x, final_y),
                 xytext=(-30, 30), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    print("\nRDF Validator Details:")
    print("-" * 50)
    print(f"Initial rows: {shacl_df['size'].iloc[0]:,.0f}")
    print(f"Breaking point: {shacl_break['size']:,.0f} rows")
    print(f"Breaking time: {shacl_break['execution_time']:.2f}s")
    print(f"Interpolated end: {final_x:,.0f} rows")
    print(f"Interpolated time: {final_y:.2f}s")
    print(f"Growth rate: {(final_y / shacl_break['execution_time']):.2f}x")

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()


if __name__ == "__main__":
    try:
        shacl_df, framework_df, shacl_break, framework_break = load_and_clean_results(
            'shacl_benchmark_results.csv',
            'framework_benchmark_results.csv'
        )

        create_time_comparison_plot(shacl_df, framework_df, shacl_break, framework_break)

        print("\nVisualization saved as 'time_benchmark_analysis.png'")

    except Exception as e:
        print(f"Error creating visualization: {str(e)}")
        raise