from rdflib import *
import os, sys
import time
import matplotlib.pyplot as plt
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(base_dir)

from Connector.ValidationFramework.parser.parser import DCParser
from Connector.ValidationFramework.translator.translator import *

tbox = Namespace('http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/tbox#')
abox = Namespace('http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/abox#')
dcat = Namespace('https://www.w3.org/ns/dcat#')
dcterms = Namespace('http://purl.org/dc/terms/')
tb = Namespace("http://www.semanticweb.org/acraf/ontologies/2021/0/SDM#")
odrl = Namespace("http://www.w3.org/ns/odrl/2/")

#%
base_dir = os.path.dirname(os.path.realpath(__file__))
sdm = Graph().parse(os.path.join(base_dir, '../../../../FederatedComputationalGovernance/SemanticDataModel/sdm.ttl'), format='turtle')
print(base_dir)

def get_CMD():
    query = '''
    PREFIX tb: <http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/tbox#>
    PREFIX ab: <http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/abox#>
    PREFIX odrl: <http://www.w3.org/ns/odrl/2/>

    SELECT ?dp
    WHERE {
        ?dp a tb:CommonDataModel .
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

def queryPC(dp):
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

    pcs = [row.pc for row in q_res]
    return pcs



def get_policy(policies_map, pc):
    query = f'''
    PREFIX tb: <http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/tbox#>
    PREFIX ab: <http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/abox#>
    PREFIX odrl: <http://www.w3.org/ns/odrl/2/>

    SELECT ?p 
    WHERE {{
        ab:{pc} tb:accordingTo ?p .
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


import matplotlib.pyplot as plt
import numpy as np


def plot_overhead(policy_times, policies_map):
    avg_parser_times = []
    avg_translation_times = []
    policies = list(policies_map.keys())

    policies.remove('p1')
    policies.sort()
    for policy in policies:
        parser_times = []
        translation_times = []
        for pc in policies_map[policy]:
            if pc in policy_times:
                parser_times.extend(policy_times[pc]['parser_times'])
                translation_times.extend(policy_times[pc]['translation_times'])

        if parser_times:
            avg_parser_times.append(np.mean(parser_times) * 1000)
        else:
            avg_parser_times.append(0)

        if translation_times:
            avg_translation_times.append(np.mean(translation_times) * 1000)
        else:
            avg_translation_times.append(0)

    x = np.arange(len(policies))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    parser_bars = ax.bar(x - width / 2, avg_parser_times, width, label='Parser Time (ms)', color='orange', hatch='//')
    translation_bars = ax.bar(x + width / 2, avg_translation_times, width, label='Translation Time (ms)', color='green',
                              hatch='xx')

    policy_label_mapping = {
        'p1v2': 'Privacy',
        'p2': 'Statistical Properties',
        'p3': 'Schema Validation',
        'p4': 'Fairness',
        'p5': 'Data Consistency'
    }

    policies = [policy_label_mapping.get(policy, policy) for policy in policies]

    ax.set_xlabel('Policies')
    ax.set_ylabel('Time (ms)')
    ax.set_xticks(x)

    ax.set_xticklabels(policies, rotation=45)
    ax.set_yscale('log')

    for bar in parser_bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    for bar in translation_bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.title('')
    plt.subplots_adjust(top=0.9)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    fig.tight_layout(rect=[0, 0, 0.85, 1])

    plt.savefig('policy_times_fixed_final.png', bbox_inches='tight')
    plt.show()


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