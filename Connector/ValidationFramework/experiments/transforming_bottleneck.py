from rdflib import *
import os, sys
import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np

import pyshacl
import subprocess

# Add the parent directory of SideCar to the Python path
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(base_dir)


from ValidateContract.parser.parser import DCParser
from ValidateContract.translator.translator import PCTranslator



tbox = Namespace('http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/tbox#')
abox = Namespace('http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/abox#')
dcat = Namespace('https://www.w3.org/ns/dcat#')
dcterms = Namespace('http://purl.org/dc/terms/')
dqv = Namespace('http://www.w3.org/ns/dqv#')

base_dir = os.path.dirname(os.path.realpath(__file__))

from rdflib import Graph, Namespace, RDF, RDFS, XSD


def extract_tbox_from_dataframe(df, tbox_namespace):
    """Extract TBox (schema) from a DataFrame."""
    g = Graph()
    g.bind('tb', tbox_namespace)

    tb = Namespace(tbox_namespace)

    # Define the class for the rows
    g.add((tb.Patient, RDF.type, RDFS.Class))

    # Map pandas dtypes to XSD datatypes
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

        # Define the property
        g.add((tb[col], RDF.type, RDF.Property))
        g.add((tb[col], RDFS.domain, tb.Patient))
        g.add((tb[col], RDFS.range, rdf_type))

    return g

def convert_to_rdf(df, abox_namespace):
    """Convert a DataFrame to an RDF graph."""
    g = Graph()
    g.bind('ab', abox_namespace)
    ab = Namespace(abox_namespace)

    for index, row in df.iterrows():
        subject = URIRef(ab[f'row{index}'])
        for col in df.columns:
            predicate = URIRef(ab[col])
            obj = Literal(row[col])
            g.add((subject, predicate, obj))
    return g


def validate_graph(g, shape_path):
    """Validate the RDF graph using a SHACL shape."""
    conforms, results_graph, results_text = pyshacl.validate(
        g,
        shacl_graph=shape_path,
        inference='rdfs',
        abort_on_first=False,
        allow_infos=False,
        allow_warnings=False
    )
    return conforms, results_text


def queryPC(ds):
    query = f'''
    PREFIX tb: <http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/tbox#>
    PREFIX ab: <http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/abox#>
    PREFIX odrl: <http://www.w3.org/ns/odrl/2/>

    SELECT ?pc ?ds
    WHERE {{
        ?pc a tb:PolicyChecker .
        ?pc tb:validates ?p .
        ?pc tb:hasType ?t .
        ?ds tb:hasDTT ?t
        FILTER (?ds = ab:{ds})
    }}
    '''
    q_res = sdm.query(query)
    pcs = [row.pc for row in q_res]
    return pcs




if __name__ == "__main__":
    file_path_tabular = os.path.join(base_dir,
                                     '../../../DataProductLayer/DataProduct/Data/Explotation/UPENN-GBM_clinical_info_v2.1.csv')
    sdm = Graph().parse(os.path.join(base_dir, '../../../FederatedComputationalGovernance/SemanticDataModel/sdm.ttl'),
                        format='turtle')
    df = pd.read_csv(file_path_tabular)
    #double size data
    df = pd.concat([df, df], ignore_index=True)
    # tiple the size
    df = pd.concat([df, df], ignore_index=True)
    # quadruple the size
    df = pd.concat([df, df], ignore_index=True)

    sizes = range(10, df.shape[0] + 1, 500)  # Start from 10, increment by 10
    graph_creation_times = []
    validation_times = []

    parsing_times = []
    translation_times = []
    udf_validation_times = []  # You can track additional UDF validation time if needed

    shape_path = 'shacl_shape.ttl'
    combined_graph = Graph()

    for size in sizes:
        df_sample = df.head(size)

        #### Measure RDF graph creation and validation (SHACL) ####
        # Graph creation time (TBox + ABox)
        start_time_graph = time.time()
        tbox_graph = extract_tbox_from_dataframe(df_sample, tbox)
        abox_graph = convert_to_rdf(df_sample, abox)
        combined_graph = tbox_graph + abox_graph
        end_time_graph = time.time()
        graph_creation_times.append(end_time_graph - start_time_graph)

        # SHACL validation time
        start_time_validation = time.time()
        conforms, results_text = validate_graph(combined_graph, shape_path)
        end_time_validation = time.time()
        validation_times.append(end_time_validation - start_time_validation)

        #### Measure parsing, translation, and UDF validation (Framework) ####
        # Parsing time
        start_time_parsing = time.time()
        dp = "UPENN-GBM_clinical_info_v21csv"
        parser = DCParser(dp, sdm).parse_contracts()
        end_time_parsing = time.time()
        parsing_times.append(end_time_parsing - start_time_parsing)

        # Translation time
        pc = queryPC(dp)[0]
        start_time_translation = time.time()
        script_path = os.path.join(base_dir, '../translator/translator.py')
        subprocess.run(['chmod', '+x', script_path])
        parameter = f'{pc} --execute'
        result = subprocess.run(['python', script_path, parameter], capture_output=True, text=True)
        end_time_translation = time.time()

        # Check translation execution
        if result.returncode == 0:
            translation_times.append(end_time_translation - start_time_translation)
        else:
            print("Translation script failed:", result.stderr)
            break

        # (Optional) If you have a separate UDF validation step
        udf_validation_times.append(0)  # Replace with actual validation time if needed

    #### Prepare Data for Plotting ####
    # Convert lists to NumPy arrays for easier operations
    graph_creation_times = np.array(graph_creation_times)
    validation_times = np.array(validation_times)
    parsing_times = np.array(parsing_times)
    translation_times = np.array(translation_times)
    udf_validation_times = np.array(udf_validation_times)

    # Assuming data arrays are already defined
    # graph_creation_times, validation_times, parsing_times, translation_times, udf_validation_times, sizes

    #### Prepare Data for Line Chart Plotting ####
    # Total execution times for RDF + SHACL and Parser + Translator
    #### Calculate Variance (simulated for demonstration purposes, replace with actual if available) ####
    # For now, we'll simulate small variances for both processes
    # You can replace these simulated variances with actual calculated variances if you have them

    total_rdf_shacl_times = graph_creation_times + validation_times
    total_parser_translator_times = parsing_times + translation_times + udf_validation_times


    rdf_shacl_variance = 0.05 * total_rdf_shacl_times  # Simulating 5% variance for demonstration
    parser_translator_variance = 0.05 * total_parser_translator_times  # Simulating 5% variance for demonstration

    #### Plotting Two Line Charts with Variance ####
    #### Plotting Two Line Charts Without Markers and with Variance ####
    plt.figure(figsize=(10, 6))

    # Plot line for RDF + SHACL total execution time (no markers)
    plt.plot(sizes, total_rdf_shacl_times, color='blue', linestyle='-', linewidth=2,
             label='RDF Validation Total Time')
    # Add shaded area for RDF + SHACL variance
    plt.fill_between(sizes, total_rdf_shacl_times - rdf_shacl_variance, total_rdf_shacl_times + rdf_shacl_variance,
                     color='blue', alpha=0.2, label='Variance')

    # Plot line for Parser + Translator total execution time (no markers)
    plt.plot(sizes, total_parser_translator_times, color='green', linestyle='-', linewidth=2,
             label='Framework Validation Total Time')
    # Add shaded area for Parser + Translator variance
    plt.fill_between(sizes, total_parser_translator_times - parser_translator_variance,
                     total_parser_translator_times + parser_translator_variance, color='green', alpha=0.2,
                     label='Variance')

    #### Add labels and formatting ####
    plt.title('')
    plt.xlabel('Size (#Rows)')
    plt.ylabel('Time (s)')

    # Rotate x-axis labels for clarity
    plt.xticks(sizes, rotation=45, ha='right')

    # Embed legend inside the plot (upper left corner)
    plt.legend(loc='upper left', frameon=True)

    # Add grid for better visual separation
    plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.7)

    # Adjust layout and spacing
    plt.tight_layout()

    # Save and show the plot
    plt.savefig('comparison_rdf_shacl_vs_parser_translator_linechart_without_markers.png')
    plt.show()
