from rdflib import *
import os, sys
import time
import matplotlib.pyplot as plt
import numpy as np

# Add the parent directory of SideCar to the Python path
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(base_dir)

from ValidateContract.parser.parser import DCParser
from ValidateContract.translator.translator import *

# Namespaces
tbox = Namespace('http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/tbox#')
abox = Namespace('http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/abox#')
dcat = Namespace('https://www.w3.org/ns/dcat#')
dcterms = Namespace('http://purl.org/dc/terms/')
tb = Namespace("http://www.semanticweb.org/acraf/ontologies/2021/0/SDM#")
odrl = Namespace("http://www.w3.org/ns/odrl/2/")

#%
base_dir = os.path.dirname(os.path.realpath(__file__))
sdm = Graph().parse(os.path.join(base_dir, '../../../FederatedComputationalGovernance/SemanticDataModel/sdm.ttl'), format='turtle')
print(base_dir)

def get_CMD():
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

def get_datasets_associated(dp):
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


def get_policy(policies_map, pc):
    query = f'''
    PREFIX tb: <http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/tbox#>
    PREFIX ab: <http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/abox#>
    PREFIX odrl: <http://www.w3.org/ns/odrl/2/>

    SELECT ?p 
    WHERE {{
        ab:{pc} tb:validates ?p .
    }}
    LIMIT 1
    '''
    q_res = sdm.query(query)
    p = [row.p for row in q_res]

    policy = p[0].split("#")[1]
    if policy not in policies_map:
        policies_map[policy] = []
    policies_map[policy].append(pc)
    return policy



def plot_overhead(policy_times, policies_map):
    avg_parser_times = []
    avg_translation_times = []
    policies = list(policies_map.keys())

    for policy in policies:
        parser_times = []
        translation_times = []
        for pc in policies_map[policy]:
            if pc in policy_times:
                parser_times.extend(policy_times[pc]['parser_times'])
                translation_times.extend(policy_times[pc]['translation_times'])

        if parser_times:
            avg_parser_times.append(np.mean(parser_times) * 1000)  # Convert to milliseconds
        else:
            avg_parser_times.append(0)  # Replace NaN with 0

        if translation_times:
            avg_translation_times.append(np.mean(translation_times) * 1000)  # Convert to milliseconds
        else:
            avg_translation_times.append(0)  # Replace NaN with 0

    x = np.arange(len(policies))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot parser times
    ax1.bar(x - width / 2, avg_parser_times, width, label='Parser Time (ms)', color='orange')
    ax1.set_xlabel('Policies')
    ax1.set_ylabel('Parser Time (ms)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(policies, rotation=45)

    # Create a second y-axis for translation times
    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, avg_translation_times, width, label='Translation Time (ms)', color='green')
    ax2.set_ylabel('Translation Time (ms)')

    fig.tight_layout()
    fig.legend(loc="upper right")
    plt.title('Average Parsing and Translation Time for Each Policy')
    plt.savefig('policy_times.png')



if __name__ == "__main__":
    CMD = get_CMD()
    policy_times = {}
    policies_map = {}

    for dp in CMD:
        print(f".Working on CDM {dp}")
        datasets = get_datasets_associated(dp)
        for dataset in datasets:
            print(f"..Parsing data product {dataset}")
            start_time_parser = time.time()
            DCParser(dataset, sdm).parse_contracts()
            end_time_parser = time.time()
            parser_time = end_time_parser - start_time_parser
            print(f"Parser time for dataset {dataset}: {parser_time} seconds")

            pcs = queryPC(dataset)
            for pc in pcs:
                if pc not in policy_times:
                    policy_times[pc.split("#")[1]] = {'parser_times': [], 'translation_times': []}

                policy_times[pc.split("#")[1]]['parser_times'].append(parser_time)

                print(f"...Translating Policy Checker: {pc}")
                start_time_translation = time.time()
                PCTranslator(pc, sdm).translate()
                end_time_translation = time.time()
                translation_time = end_time_translation - start_time_translation
                policy_times[pc.split("#")[1]]['translation_times'].append(translation_time)
                print(f"Translation time for policy checker {pc}: {translation_time} seconds")

                get_policy(policies_map, pc.split("#")[1])

    plot_overhead(policy_times, policies_map)