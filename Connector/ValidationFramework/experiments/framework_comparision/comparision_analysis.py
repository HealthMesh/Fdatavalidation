import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_and_clean_results(shacl_file: str, framework_file: str):
    shacl_df = pd.read_csv(shacl_file)
    framework_df = pd.read_csv(framework_file)

    shacl_df['total_validation_time'] = shacl_df['rdf_conversion_time'] + shacl_df['shacl_validation_time']

    shacl_break = shacl_df.iloc[-1]
    framework_break = framework_df.iloc[-1]

    return shacl_df, framework_df, shacl_break, framework_break


def create_time_comparison_plot(shacl_df, framework_df, shacl_break, framework_break,
                                output_file='time_benchmark_analysis.png'):
    plt.figure(figsize=(12, 8))

    plt.plot(shacl_df['size'], shacl_df['shacl_validation_time'],
             label='SHACL Validation Only', color='blue', marker='o', markersize=4, linewidth=2)

    plt.plot(shacl_df['size'], shacl_df['total_validation_time'],
             label='Total (RDF Conversion + SHACL)', color='red', marker='o', markersize=4, linewidth=2)


    plt.plot(framework_df['size'], framework_df['execution_time'],
             label='Framework', color='green', marker='o', markersize=4, linewidth=2)

    plt.plot(shacl_break['size'], shacl_break['shacl_validation_time'],
             'rx', markersize=15, markeredgewidth=3)
    plt.plot(framework_break['size'], framework_break['execution_time'],
             'rx', markersize=15, markeredgewidth=3)

    plt.xlabel('Number of Rows', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(fontsize=10)

    plt.annotate(f'RDF Validation Break:\n{shacl_break["size"]:,} rows\n'
                 f'SHACL: {shacl_break["shacl_validation_time"]:.2f}s\n'
                 f'Total: {shacl_break["total_validation_time"]:.2f}s',
                 xy=(shacl_break['size'], shacl_break['execution_time']),
                 xytext=(30, -30), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.annotate(f'Framework Break:\n{framework_break["size"]:,} rows\n{framework_break["execution_time"]:.2f}s',
                 xy=(framework_break['size'], framework_break['execution_time']),
                 xytext=(-100, 30), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    shacl_final_time = shacl_break['shacl_validation_time']
    total_final_time = shacl_break['total_validation_time']
    framework_final_time = framework_break['execution_time']

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