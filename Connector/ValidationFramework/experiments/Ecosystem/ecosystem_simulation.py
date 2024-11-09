from rdflib import *
from hashlib import sha256

import os, sys
import pandas as pd
import pydicom
import uuid
import random
import json
import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
sys.path.append(base_dir)

from ValidateContract.parser.parser import DCParser


tbox = Namespace('http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/tbox#')
abox = Namespace('http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/abox#')
dcat = Namespace('https://www.w3.org/ns/dcat#')
dcterms = Namespace('http://purl.org/dc/terms/')
dqv = Namespace('http://www.w3.org/ns/dqv#')

# %%ç

base_dir = os.path.dirname(os.path.realpath(__file__))

def generate_unique_uri():

    unique_identifier = str(uuid.uuid4())
    return unique_identifier


class Profiler:

    def __init__(self, file_path, owner="Unknown"):
        self.file_path = file_path
        self.source_graph = self.create_graph()
        self.datasetname = self.self_get_dataset_name()
        self.set_owner = self.set_owner(owner)
        self.file_extension = self.get_file_extension()
        self.technology = self.add_technology()
        self.source_graph = self.extract_metadata()

    def create_graph(self):
        """
        This function returns a graph object with the necessary prefixes
        :return: RDF Graph
        """
        g = Graph()
        g.bind('tb', 'http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/tbox#')
        g.bind('ab', 'http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/abox#')
        return g

    def set_owner(self, owner):
        # Owner Metadata
        self.source_graph.add((abox[self.datasetname], tbox.owner, Literal(owner)))
        return owner

    def self_get_dataset_name(self):

        #name = os.path.basename(self.file_path).replace('.', '')
        name = generate_unique_uri()
        # Name Metadata
        self.source_graph.add((abox[name], RDF.type, dcat.Dataset))
        identifier = sha256(name.encode('utf-8')).hexdigest()
        self.source_graph.add((abox[name], dcterms.identifier, Literal(identifier)))
        return name

    def get_dataname(self):
        return self.datasetname

    def get_file_extension(self):
        file_name, file_extension = os.path.splitext(self.file_path)
        format = abox.Tabular
        if file_extension.lower() == '.csv':
            format = abox.Tabular
        elif file_extension.lower() == '.dcm':
            format = abox.Image

        # DataSetTypeTemplate Metadata
        self.source_graph.add((format, RDF.type, tbox.DatasetTypeTemplate))
        self.source_graph.add((format, dcterms['format'], Literal(file_extension)))  # Correct usage of the namespace
        self.source_graph.add((abox[self.datasetname], tbox.hasDTT, format))

        return file_extension.lower()

    def generate_unique_uri(self, base_uri):
        import uuid
        unique_identifier = str(uuid.uuid4())
        return URIRef(f"{base_uri}{unique_identifier}")

    def add_technology(self):
        # triple
        self.source_graph.add((abox[self.datasetname + "_TA"], RDF.type, tbox.TechnologyAspects))
        self.source_graph.add((abox[self.datasetname], tbox.hasTA, abox[self.datasetname + "_TA"]))

        acces_uri = self.generate_unique_uri(abox)
        self.source_graph.add((abox[self.datasetname + "_TA"], tbox.typeAcces, acces_uri))
        self.source_graph.add((acces_uri, RDF.type, tbox.Acces))
        self.source_graph.add((acces_uri, RDFS.label, abox.Static))
        # PATH
        self.source_graph.add((acces_uri, tbox.path, Literal(self.file_path)))

    def extract_metadata(self):
        if self.file_extension.lower() == '.csv':
            return self.extract_csv_metadata()
        elif self.file_extension.lower() == '.dcm':
            return self.extract_dicom_metadata()
        else:
            raise ValueError(f"Unsupported file extension: {self.file_extension}")

    def extract_csv_metadata(self):
        df = pd.read_csv(self.file_path)
        for column in df.columns:
            self.source_graph.add((abox[column], RDF.type, tbox.Attribute))
            self.source_graph.add((abox[self.datasetname], tbox.hasAttribute, abox[column]))
            self.source_graph.add((abox[column], tbox.attribute, Literal(column)))
        return self.source_graph

    def extract_dicom_metadata(self, n_attributes=50):
        ds = pydicom.dcmread(self.file_path)
        # Iterate over all attributes
        for attribute in dir(ds)[:n_attributes]:
            if attribute[0].isalpha():
                if hasattr(ds, attribute):
                    self.source_graph.add((abox[attribute], RDF.type, tbox.Attribute))
                    self.source_graph.add((abox[self.datasetname], tbox.hasAttribute, abox[attribute]))
                    self.source_graph.add((abox[attribute], tbox.attribute, Literal(attribute)))
        return self.source_graph

    def get_source_graph(self):
        return self.source_graph


class Federator:
    """
        Federator class is responsible for federating the data from different sources.
    """

    def __init__(self, ds, SDM):
        self.ds = ds
        self.sdm = SDM

    def check_dp_existance(self):
        if self.sdm.value(predicate=RDF.type, subject=abox[self.ds]):
            return True
        else:
            return False

    def generate_uri_id(self):
        return str(uuid.uuid4())

    def add_mappings(self, mappings):

        # create contract
        self.sdm.add((abox[f'dc_{self.ds}'], RDF.type, tbox.DataContract))
        self.sdm.add((abox[self.ds], tbox.hasDC, abox[f'dc_{self.ds}']))

        # add mappings
        for key, value in mappings.items():  # as key value pair dictionary

            # Generate mapping UUID
            mapping_uuid = self.generate_uri_id()
            self.sdm.add((abox[mapping_uuid], RDF.type, tbox.SchemaMapping))
            self.sdm.add((abox[f'dc_{self.ds}'], tbox.hasMapping, abox[mapping_uuid]))

            # Add Mapping
            self.sdm.add((abox[mapping_uuid], tbox.mfrom, abox[key]))
            self.sdm.add((abox[mapping_uuid], tbox.mto, abox[value]))

        return self.sdm

    def add_policies(self, policies):
        # Add agreed policies
        for policy in policies:
            self.sdm.add((abox[f'dc_{self.ds}'], tbox.hasPolicy, abox[policy]))

        return self.sdm


def get_CMD(sdm):
    query = '''
    PREFIX tb: <http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/tbox#>
    PREFIX ab: <http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/abox#>
    PREFIX odrl: <http://www.w3.org/ns/odrl/2/>

    SELECT ?dp
    WHERE {
        ?dp a tb:DataProduct .
    }
    '''
    q_res = sdm.query(query)
    CMD = [row.dp.split("#")[1] for row in q_res]
    return CMD

def get_datasets_associated(dp, sdm):
    query = f'''
    PREFIX tb: <http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/tbox#>
    PREFIX ab: <http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/abox#>
    PREFIX odrl: <http://www.w3.org/ns/odrl/2/>

    SELECT ?dataset
    WHERE {{
        ?dataset tb:hasDC ?dcc .
        ?dcc tb:hasPolicy ?p .
        ?p odrl:duty ?d .
        ?d odrl:target ?t .
        ?dp tb:hasFeature ?t .
        FILTER (?dp = ab:{dp})
    }}
    '''
    qres = sdm.query(query)
    datasets = [row.dataset.split("#")[1] for row in qres]
    datasets = set(datasets)
    return datasets

def queryPC(ds, sdm):
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

def queryOPS(eco):
    query = '''
    PREFIX tb: <http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/tbox#>
    PREFIX ab: <http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/abox#>

    SELECT (COUNT(?node) AS ?totalCount)
    WHERE {
        ?node a ?type .
        FILTER (?type IN (tb:Operation))
    }
    '''
    q_res = eco.query(query)
    total_count = q_res.bindings[0]['totalCount'] if q_res.bindings else 0
    return total_count


def update_policy_uris(graph):
    query = '''
    PREFIX odrl: <http://www.w3.org/ns/odrl/2/>
    PREFIX ab: <http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/abox#>

    SELECT ?policy
    WHERE {
        ?policy a odrl:Privacy .
    }
    '''
    qres = graph.query(query)
    for row in qres:
        old_uri = row.policy
        new_uri = generate_unique_uri(abox)

        # Add new triples with the new URI
        for s, p, o in graph.triples((old_uri, None, None)):
            graph.add((new_uri, p, o))

        # Remove old triples
        graph.remove((old_uri, None, None))

    return graph


def generate_unique_uri():
    return URIRef(f"{uuid.uuid4()}")

def duplicate_policies(graph, original_policy_uri, data_contract_uri, n):
    for i in range(n):
        new_policy_uri = abox[generate_unique_uri()]
        for s, p, o in graph.triples((original_policy_uri, None, None)):
            graph.add((new_policy_uri, p, o))
        graph.add((data_contract_uri, tbox.hasPolicy, new_policy_uri))

    return graph



if __name__ == "__main__":

    # Tabular data asset
    file_path_tabular = os.path.join(base_dir, '../../../../DataProductLayer/DataProduct/Data/Explotation/UPENN-GBM_clinical_info_v2.1.csv')

    # Numero de Data Products
    NDataProducts = 1
    NDPolicies = 100


    execution_times = []
    pcs_counts = []
    ndp_list = list(range(1, NDPolicies + 1))  # List of NDP counts


    ecosystem = Graph().parse(os.path.join(base_dir, 'sdm_copy.ttl'), format='turtle')


    # Generate N Data Product
    for iteration in range(1,NDataProducts+1):
        SDM = get_CMD(ecosystem)
        chosen_file_path = file_path_tabular
        profiler = Profiler(chosen_file_path)
        graph = profiler.get_source_graph()
        dataset_name = profiler.get_dataname()
        ecosystem += graph
        federate = Federator(dataset_name, ecosystem)
        dp_meta_path = os.path.join(base_dir, '../../../../DataPlatformLayer/Integration/dp1.json')
        dp_meta = json.load(open(dp_meta_path))
        mappings = dp_meta['mappings']
        policies = dp_meta['policies'][:1]
        federate.add_mappings(mappings)
        integration_graph = federate.add_policies(policies)
        ecosystem += integration_graph


    ecosystem.serialize(destination='ecosystem_policies.ttl', format='turtle')


    # Genearate n Policies in each iteration
    for i in range(NDPolicies):

        ecosystem = Graph().parse(os.path.join(base_dir, 'ecosystem_policies.ttl'), format='turtle')

        start_time = time.time()  # Start timer
        SDM = get_CMD(ecosystem)

        # Get a random Data Product
        dps = get_datasets_associated(SDM[0], ecosystem)

        for dp in dps:

            dc = ecosystem.value(predicate=tbox.hasDC, subject=abox[dp])
            ecosystem = duplicate_policies(ecosystem, abox["p1"], dc, i)
            ecosystem.remove((dc, tbox.hasPolicy, abox["p1"]))
            pc = DCParser(dp, ecosystem).parse_contracts()
            ecosystem += pc
        end_time = time.time()  # End timer
        pcs_counts.append(int(queryOPS(ecosystem)))
        execution_times.append(end_time - start_time)
        print("count", int(queryOPS(ecosystem)))


        ecosystem.serialize(destination=f'ecosystem_policies_after.ttl', format='turtle')







    # Data for plotting
    x = ndp_list  # Number of NDPs
    y = execution_times  # Number of PCs
    z = pcs_counts  # Execution time for each NDP

    for ndp in ndp_list:
        print(f"NDP: {ndp} - Execution Time: {execution_times[ndp-1]} - OPS: {pcs_counts[ndp-1]}")

    # Create 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the data
    ax.plot_trisurf(x, y, z, cmap='viridis')

    # Labels
    ax.set_xlabel('#Policies')
    ax.set_ylabel('Time (seconds)')
    ax.set_zlabel('#OPS')

    plt.savefig("ecosystem_policies.png")

    # Data for plotting
    x = ndp_list  # Number of NDPs
    y = execution_times  # Execution time for each NDP

    # Create 2D figure
    plt.figure()
    plt.plot(x, y, color='r', label='Execution Time')

    # Labels and title
    plt.xlabel('Policies')
    plt.ylabel('Time (seconds)')
    plt.title('Execution Time vs Number of Data Products')
    plt.legend()

    # Save the plot
    plt.savefig("ecosystem_policies_2d.png")
    plt.show()