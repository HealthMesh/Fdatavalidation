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
    plt.figure(figsize=(12, 8))

    plt.plot(framework_df['size'], framework_df['execution_time'],
             label='Validation Framework', color='green', marker='o', markersize=4, linewidth=2)

    # Interpolate all three time metrics
    x_interp_total, y_interp_total, final_x_total, final_y_total = interpolate_data(
        rdf_df, framework_df, 'total_validation_time')
    _, _, _, final_y_rdf = interpolate_data(
        rdf_df, framework_df, 'rdf_conversion_time')
    _, _, _, final_y_shacl = interpolate_data(
        rdf_df, framework_df, 'shacl_validation_time')

    plt.plot(rdf_df['size'], rdf_df['total_validation_time'],
             label='RDF Validator', color='blue', marker='o', markersize=4, linewidth=2)
    plt.plot(x_interp_total, y_interp_total, '--', color='blue',
             label='RDF Validator (Interpolated)', linewidth=2, alpha=0.7)

    plt.plot(rdf_break['size'], rdf_break['total_validation_time'],
             'rx', markersize=15, markeredgewidth=3)
    plt.plot(framework_break['size'], framework_break['execution_time'],
             'rx', markersize=15, markeredgewidth=3)

    plt.xlabel('Number of Rows', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(fontsize=10)

    plt.annotate(f'RDF Validator Break:\n{rdf_break["size"]:,} rows\n'
                 f'Total: {rdf_break["total_validation_time"]:.2f}s\n'
                 f'RDF: {rdf_break["rdf_conversion_time"]:.2f}s\n'
                 f'SHACL: {rdf_break["shacl_validation_time"]:.2f}s',
                 xy=(rdf_break['size'], rdf_break['total_validation_time']),
                 xytext=(30, -50), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.annotate(f'Framework Break:\n{framework_break["size"]:,} rows\n'
                 f'Time: {framework_break["execution_time"]:.2f}s',
                 xy=(framework_break['size'], framework_break['execution_time']),
                 xytext=(-100, 30), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.annotate(f'Interpolated RDF Validator:\n{final_x_total:,.0f} rows\n'
                 f'Total: {final_y_total:.2f}s\n'
                 f'RDF: {final_y_rdf:.2f}s\n'
                 f'SHACL: {final_y_shacl:.2f}s',
                 xy=(final_x_total, final_y_total),
                 xytext=(-250, -40), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
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