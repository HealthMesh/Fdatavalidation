from rdflib import *
import os, sys
import time
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(base_dir)

from Connector.ValidationFramework.planner.planner import DCParser
from Connector.ValidationFramework.translator.executor import *

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



def get_pc(p):
    query = f'''
    PREFIX tb: <http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/tbox#>
    PREFIX ab: <http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/abox#>
    PREFIX odrl: <http://www.w3.org/ns/odrl/2/>

    SELECT ?pc 
    WHERE {{
        ?pc tb:accordingTo ab:{p} .
    }}
    LIMIT 1
    '''
    q_res = sdm.query(query)

    pcs = [row.pc for row in q_res]
    return pcs[0] if pcs else None


def plot_overhead(policy_times):
    plt.rcParams.update({
        'font.size': 20,
        'font.family': 'serif',
        'axes.labelsize': 20,
        'axes.titlesize': 20,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20
    })

    policy_label_mapping = {
        'p1v2': 'Privacy',
        'p2': 'Statistical Properties',
        'p3': 'Schema Validation',
        'p4': 'Fairness',
        'p5': 'Data Consistency'
    }

    policies = [p for p in policy_times.keys() if p != 'p1']
    policies.sort()

    parser_times = [policy_times[policy]['parser_times'][0] * 1000 for policy in policies]
    translation_times = [policy_times[policy]['translation_times'][0] * 1000 for policy in policies]

    fig, ax = plt.subplots(figsize=(10, 7))

    x = np.arange(len(policies))
    width = 0.35

    parser_bars = ax.bar(x - width / 2, parser_times, width,
                         label='Parser',
                         color='orange',
                         hatch='//',
                         edgecolor='black',
                         linewidth=1)

    translation_bars = ax.bar(x + width / 2, translation_times, width,
                              label='Translator',
                              color='green',
                              hatch='xx',
                              edgecolor='black',
                              linewidth=1)

    ax.set_ylabel('Time (ms)', fontsize=14)
    ax.set_yscale('log')

    ymin = 1e-3  # min
    ymax = 1e5  # max
    ax.set_ylim(ymin, ymax)

    ax.set_yticks([1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)

    ax.grid(True, which='major', linestyle='--', alpha=0.3)

    policies_labels = [policy_label_mapping.get(policy, policy) for policy in policies]
    ax.set_xticks(x)
    ax.set_xticklabels(policies_labels, rotation=45, ha='right')


    ax.legend(loc='upper right',
              bbox_to_anchor=(1.0, 0.95),
              frameon=True,
              fancybox=False,
              edgecolor='black',
              fontsize=12)

    plt.tight_layout()

    plt.savefig('policy_times.png',
                dpi=300,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none')

    plt.show()

if __name__ == "__main__":
    CMD = get_CMD()
    policy_times = {}
    policies_map = {}

    for dp in CMD:
        print(f".Working on CDM {dp}")
        datasets = get_datasets_associated(dp)
        for dataset in datasets:
            print(f"..Processing dataset: {dataset}")
            parser = DCParser(dataset, sdm)
            policies, mappings = parser._read_contracts()
            for policy in policies:
                print(f"...Parsing Policy: {policy}")
                policy_times[policy.split("#")[1]] = {'parser_times': [], 'translation_times': []}
                start_time_parser = time.time()
                pc = parser._parse_policy(policy)
                policy_time = time.time() - start_time_parser
                policy_times[policy.split("#")[1]]['parser_times'].append(policy_time)
                sdm = sdm + pc

                # get policy checker

                pc = get_pc(policy.split("#")[1])
                print(f"...Translating Policy Checker: {pc}")
                start_time_translation = time.time()
                PCTranslator(pc, sdm).translate()
                end_time_translation = time.time()
                translation_time = end_time_translation - start_time_translation
                policy_times[policy.split("#")[1]]['translation_times'].append(translation_time)
                print(f"Translation time for policy checker {pc}: {translation_time} seconds")


    plot_overhead(policy_times)