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


    sizes = range(10, df.shape[0] + 1, 50)  # Start from 10, increment by 10
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

    # Total times for both processes
    total_rdf_shacl_times = graph_creation_times + validation_times
    total_parser_translator_times = parsing_times + translation_times + udf_validation_times

    #### Plotting Comparative Stacked Bar Charts ####
    plt.figure(figsize=(12, 8))

    # Increase the bar width
    bar_width = 8  # Make bars thicker
    bar_positions = np.array(list(sizes)) - bar_width / 2  # Shift positions for side-by-side bars

    plt.bar(bar_positions, graph_creation_times, width=bar_width, color='skyblue', label='RDF Graph Creation Time')
    plt.bar(bar_positions, validation_times, width=bar_width, bottom=graph_creation_times, color='orange',
            label='SHACL Validation Time')

    # Plot Parser + Translator times (side-by-side)
    bar_positions_translator = np.array(list(sizes)) + bar_width / 2  # Shift second bar positions
    plt.bar(bar_positions_translator, parsing_times, width=bar_width, color='lightgreen', label='Parser Time')
    plt.bar(bar_positions_translator, translation_times, width=bar_width, bottom=parsing_times, color='purple',
            label='Translator Time')

    #### Add labels and formatting ####
    plt.title('Comparison of RDF+SHACL vs. Parser+Translator Execution Time by Data Size')
    plt.xlabel('Data Size (Number of Rows)')
    plt.ylabel('Execution Time (seconds)')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xticks(sizes)  # Set x-ticks to be the sizes
    plt.grid(True)

    # Save and show the plot
    plt.tight_layout()  # Adjust layout to prevent clipping of legend
    plt.savefig('comparison_rdf_shacl_vs_parser_translator.png')
    plt.show()