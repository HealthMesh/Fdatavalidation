import os
import sys
import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np
import psutil
import gc
import logging
from typing import Dict, List
from rdflib import *

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='framework_benchmark_new.log'
)

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(base_dir)

from Connector.ValidationFramework.parser.parser import DCParser
from Connector.ValidationFramework.translator.translator import PCTranslator


def get_process_memory() -> float:

    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def save_metrics(metrics: List[Dict], filename: str = 'framework_benchmark_results_new.csv'):

    pd.DataFrame(metrics).to_csv(filename, index=False)


def plot_results(metrics: List[Dict], output_file: str = 'framework_benchmark_results_new.png'):

    results = pd.DataFrame(metrics)
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 1, 1)
    plt.plot(results['size'], results['execution_time'], marker='o', color='green')
    plt.title('Framework Execution Time vs Data Size')
    plt.xlabel('Number of Rows')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    plt.yscale('log')
    plt.xscale('log')


    plt.subplot(2, 1, 2)
    plt.plot(results['size'], results['memory_used'], marker='o', color='green')
    plt.title('Framework Memory Usage vs Data Size')
    plt.xlabel('Number of Rows')
    plt.ylabel('Memory Usage (MB)')
    plt.grid(True)
    plt.yscale('log')
    plt.xscale('log')

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def queryPC(sdm, dp):
    """Query to get policy checker from semantic data model"""
    query = f'''
    PREFIX tb: <http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/tbox#>
    PREFIX ab: <http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/abox#>
    PREFIX odrl: <http://www.w3.org/ns/odrl/2/>

    SELECT ?pc ?ds
    WHERE {{
        ?pc a tb:PolicyChecker .
        ?pc tb:validates ?dp .
        FILTER (?dp = ab:{dp})        
    }}
    '''
    q_res = sdm.query(query)
    return [row.pc for row in q_res]


def run_framework_benchmark(initial_df: pd.DataFrame, sdm: Graph) -> List[Dict]:

    metrics = []
    tbox = Namespace('http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/tbox#')
    abox = Namespace('http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/abox#')

    current_df = initial_df.copy()
    iteration = 0

    # Prepare framework components
    dp = "UPENN-GBM_clinical_info_v21csv"

    #parsing
    parser = DCParser(dp, sdm).parse_contracts()
    sdm_local = sdm + parser
    pc = queryPC(sdm_local, dp)[0]

    #translation
    udf = PCTranslator(pc.split("#")[1], sdm_local).translate()


    initOP = sdm_local.value(subject=abox[pc.split("#")[1]], predicate=tbox.nextStep)
    path = sdm_local.value(subject=initOP, predicate=tbox.hasInput)

    while True:
        current_size = len(current_df)
        logging.info(f"\nStarting Framework benchmark iteration {iteration + 1}, size: {current_size}")
        gc.collect()

        temp_file = f'temp_{current_size}.csv'
        try:
            current_df.to_csv(temp_file, index=False, na_rep='NA')

            start_memory = get_process_memory()
            start_time = time.time()

            udf(temp_file) # UDF EXECUTION

            end_time = time.time()
            end_memory = get_process_memory()

            execution_time = end_time - start_time
            memory_used = end_memory - start_memory

            metrics.append({
                'iteration': iteration,
                'size': current_size,
                'execution_time': execution_time,
                'memory_used': memory_used,
                'success': True
            })

            # Save and plot intermediate results
            save_metrics(metrics)
            plot_results(metrics)

            print(
                f"Framework Iteration {iteration} - Size: {current_size}, Time: {execution_time:.2f}s, Memory: {memory_used:.2f}MB")

            current_df = pd.concat([current_df, current_df], ignore_index=True)
            iteration += 1

        except Exception as e:
            logging.error(f"Framework benchmark failed at size {current_size}: {str(e)}")
            break

        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    return metrics


if __name__ == "__main__":

    file_path = os.path.join(base_dir,
                             '../../DataProductLayer/DataProduct/Data/Explotation/UPENN-GBM_clinical_info_v2.1.csv')
    sdm = Graph().parse(
        os.path.join(base_dir, '../../FederatedComputationalGovernance/SemanticDataModel/sdm.ttl'),
        format='turtle'
    )

    initial_df = pd.read_csv(file_path)


    framework_metrics = run_framework_benchmark(initial_df.copy(), sdm)


