import os
import sys
import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np
import pyshacl
import psutil
import gc
import logging
from typing import Dict, List, Tuple
from rdflib import *


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='shacl_benchmark.log'
)


def get_process_memory() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def save_metrics(metrics: List[Dict], filename: str = 'shacl_benchmark_results.csv'):
    pd.DataFrame(metrics).to_csv(filename, index=False)


def plot_results(metrics: List[Dict], output_file: str = 'shacl_benchmark_results.png'):
    """Create visualization of SHACL benchmark results"""
    results = pd.DataFrame(metrics)
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 1, 1)
    plt.plot(results['size'], results['execution_time'], marker='o')
    plt.title('SHACL Execution Time vs Data Size')
    plt.xlabel('Number of Rows')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    plt.yscale('log')
    plt.xscale('log')

    plt.subplot(2, 1, 2)     # THIS IS NOT WOKRING WELL
    plt.plot(results['size'], results['memory_used'], marker='o', color='green')
    plt.title('SHACL Memory Usage vs Data Size')
    plt.xlabel('Number of Rows')
    plt.ylabel('Memory Usage (MB)')
    plt.grid(True)
    plt.yscale('log')
    plt.xscale('log')

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def extract_tbox_from_dataframe(df: pd.DataFrame, tbox_namespace: Namespace) -> Graph:

    g = Graph()
    g.bind('tb', tbox_namespace)

    tb = Namespace(tbox_namespace)
    g.add((tb.Patient, RDF.type, RDFS.Class))

    dtype_mapping = {
        'int64': XSD.integer,
        'float64': XSD.double,
        'object': XSD.string,
        'bool': XSD.boolean,
        'datetime64[ns]': XSD.dateTime
    }

    for col in df.columns:
        col_type = str(df[col].dtype)
        rdf_type = dtype_mapping.get(col_type, XSD.string)
        g.add((tb[col], RDF.type, RDF.Property))
        g.add((tb[col], RDFS.domain, tb.Patient))
        g.add((tb[col], RDFS.range, rdf_type))

    return g


def convert_to_rdf(df: pd.DataFrame, abox_namespace: Namespace) -> Graph:

    g = Graph()
    g.bind('ab', abox_namespace)
    ab = Namespace(abox_namespace)

    try:
        for index, row in df.iterrows():
            subject = URIRef(ab[f'row{index}'])
            for col in df.columns:
                if pd.notna(row[col]):
                    predicate = URIRef(ab[col])
                    obj = Literal(row[col])
                    g.add((subject, predicate, obj))
    except Exception as e:
        logging.error(f"Error converting DataFrame to RDF: {str(e)}")
        raise

    return g


def validate_graph(g: Graph, shape_path: str) -> Tuple[bool, str]:

    try:
        conforms, results_graph, results_text = pyshacl.validate(
            g,
            shacl_graph=shape_path,
            inference='rdfs',
            abort_on_first=False,
            allow_infos=False,
            allow_warnings=False
        )
        return conforms, results_text
    except Exception as e:
        logging.error(f"Error during SHACL validation: {str(e)}")
        raise


def run_shacl_benchmark(initial_df: pd.DataFrame, shape_path: str) -> List[Dict]:

    metrics = []
    tbox = Namespace('http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/tbox#')
    abox = Namespace('http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/abox#')

    current_df = initial_df.copy()
    iteration = 0

    while True:
        current_size = len(current_df)
        logging.info(f"\nStarting SHACL benchmark iteration {iteration + 1}, size: {current_size}")
        gc.collect()

        try:
            start_memory = get_process_memory()
            start_time = time.time()


            tbox_graph = extract_tbox_from_dataframe(current_df, tbox)
            abox_graph = convert_to_rdf(current_df, abox)
            combined_graph = tbox_graph + abox_graph
            conforms, results = validate_graph(combined_graph, shape_path)

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

            save_metrics(metrics)
            plot_results(metrics)

            print(
                f"SHACL Iteration {iteration} - Size: {current_size}, Time: {execution_time:.2f}s, Memory: {memory_used:.2f}MB")

            current_df = pd.concat([current_df, current_df], ignore_index=True)
            iteration += 1

        except Exception as e:
            logging.error(f"SHACL benchmark failed at size {current_size}: {str(e)}")
            break

    return metrics


if __name__ == "__main__":

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    file_path = os.path.join(base_dir,
                             '../../DataProductLayer/DataProduct/Data/Explotation/UPENN-GBM_clinical_info_v2.1.csv')
    shape_path = 'shacl_shape.ttl'

    initial_df = pd.read_csv(file_path)

    shacl_metrics = run_shacl_benchmark(initial_df.copy(), shape_path)
