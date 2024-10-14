from rdflib import *
from owlrl import *
import os
import json
import uuid
import argparse
import time
import matplotlib.pyplot as plt
from pyshacl import validate


# basedir
base_dir = os.path.dirname(os.path.realpath(__file__))

# RDF Namespaces
tbox = Namespace('http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/tbox#')
abox = Namespace('http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/abox#')
dcat = Namespace('https://www.w3.org/ns/dcat#')
dcterms = Namespace('http://purl.org/dc/terms/')
tb = Namespace("http://www.semanticweb.org/acraf/ontologies/2021/0/SDM#")
odrl = Namespace("http://www.w3.org/ns/odrl/2/")
prov = Namespace("http://www.w3.org/ns/prov#")
dqv = Namespace("http://www.w3.org/ns/dqv#")


# Utility functions
def generate_unique_uri(base_uri):
    """Generate a unique URI based on the base URI."""
    unique_identifier = str(uuid.uuid4())
    return URIRef(f"{base_uri}{unique_identifier}")

def load_jsonld_data(file_path):
    """Load and parse JSON-LD data from a file."""
    with open(file_path, 'r') as f:
        return json.loads(f.read())


# Decorator to measure the time taken by a function and store policy names
def timeit(func):
    """Decorator to measure time taken by a function."""
    def wrapper(self, policy, *args, **kwargs):
        start_time = time.time()
        result = func(self, policy, *args, **kwargs)
        elapsed_time = time.time() - start_time
        wrapper.times.append(elapsed_time)
        wrapper.policy_names.append(str(policy))
        return result
    wrapper.times = []
    wrapper.policy_names = []
    return wrapper




# Policy Checker Class

class PolicyChecker(Graph):
    """Class representing the policy checker that validates policies."""

    def __init__(self, p, p_type, data_format, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        self.p_type = p_type.split("/")[-1]
        self.bind("ab", "http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/abox#")
        self.bind("tb", "http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/tbox#")
        self.URI = generate_unique_uri(abox)
        self.add((self.URI, RDF.type, tbox.PolicyChecker))
        self.add((self.URI, tbox.validates, p))
        self.add((self.URI, tbox.hasType, data_format))

    def get_URI(self):
        return self.URI

    def get_policy_type(self):
        return self.p_type

    def get_policy(self):
        return self.p


# Policy Parsing Class

class DCParser:
    """
    Parse Policies of Data Contracts to Policy Checkers
    """

    def __init__(self, dp, graph):
        self.dp = dp
        self.g = graph
        self.attr_mappings = None

    def validate_graph(self) -> bool:
        """
        Validate the policies grammar is compliant with the grammar defined
        :return: conformance/non-conformance
        """
        shapes = Graph().parse(os.path.join(base_dir, 'policy_grammar.json'), format="turtle")
        conforms, report_graph, report_text = validate(self.g, shacl_graph=shapes)
        return conforms

    def read_contracts(self):
        """
        Get the policies associated with a data product
        :return: list of policies and mappings
        """
        contracts = self.g.objects(subject=abox[self.dp], predicate=tbox.hasDC)
        policies_list, mappings_dict = [], {}

        for contract in contracts:
            # Handle policies
            policies = self.g.objects(subject=contract, predicate=tbox.hasPolicy)
            policies_list.extend(policies)

            # Handle attribute mappings
            mappings = self.g.objects(subject=contract, predicate=tbox.hasMapping)
            for mapping in mappings:
                mfrom = self.g.value(subject=mapping, predicate=tbox.mfrom)
                mto = self.g.value(subject=mapping, predicate=tbox.mto)
                mappings_dict[mto] = mfrom

        self.attr_mappings = mappings_dict
        return policies_list, mappings_dict

    def execute_rule(self, rule_path, pc, mappings, nrules=1):
        """Execute SPARQL rules for policy validation, simulating a larger number of rules."""
        rule_files = os.listdir(rule_path)[:nrules]  # Limit the number of rules to simulate scalability

        for sparqlrule in rule_files:
            with open(os.path.join(rule_path, sparqlrule), 'r') as file:
                rule = file.read()
                for key, value in mappings.items():
                    rule = rule.replace(f"<{{{key}}}>", f"<{value}>")
                try:
                    results = self.g.query(rule)
                    result_graph = Graph()
                    for triple in results:
                        result_graph.add(triple)
                    pc += result_graph
                except Exception as e:
                    print("Parsing Error: ", e)
        return pc

    def get_last_op(self, pc):
        """Retrieve the last operation of a policy checker."""
        last_op = pc.value(subject=pc.get_URI(), predicate=tbox.nextStep)
        while pc.value(subject=last_op, predicate=tbox.nextStep):
            last_op = pc.value(subject=last_op, predicate=tbox.nextStep)
        return last_op

    def init_op(self, policy, pc):
        """Initialize operations for a policy."""
        initOPrules = os.path.join(base_dir, "../parser/rules/initOP")
        mappings = {
            "dp": abox[self.dp],
            "pc": pc.get_URI(),
            "op_uri": generate_unique_uri(abox),
        }
        pc = self.execute_rule(initOPrules, pc, mappings)
        return self.get_last_op(pc), pc

    def handle_attributes(self, pc):
        """Update attributes in the policy checker according to mappings."""
        operation = pc.get_URI()
        while operation:
            attribute = pc.value(subject=operation, predicate=tbox.hasAttribute)
            if attribute and attribute in self.attr_mappings:
                pc.remove((operation, tbox.hasAttribute, None))
                pc.add((operation, tbox.hasAttribute, self.attr_mappings[attribute]))
            operation = pc.value(subject=operation, predicate=tbox.nextStep)
        return pc

    def handle_duties(self, pc, init_op):
        """Handle duties in a policy checker."""
        ops_rules_path = os.path.join(base_dir, "../parser/rules/OPS")
        mappings = {
            "dp": abox[self.dp],
            "pc": pc.get_URI(),
            "op_uri": generate_unique_uri(abox),
            "last_op": init_op,
            "policy_uri": pc.get_policy(),
        }
        pc = self.execute_rule(ops_rules_path, pc, mappings)
        return self.get_last_op(pc), pc

    def plot_times(self, filename='policy_times.png'):
        """Plot the time taken to parse each policy."""
        plt.bar(self.parse_policy.policy_names, self.parse_policy.times)
        plt.title('Time taken to parse each policy')
        plt.xlabel('Policy')
        plt.ylabel('Time (seconds)')
        plt.xticks(rotation=90)  # Rotate policy names for better readability
        plt.tight_layout()  # Adjust layout to make room for labels
        plt.savefig(filename)
        plt.close()

    @timeit
    def parse_policy(self, policy):
        """Parse an individual policy into an intermediate representation."""
        p_type = self.g.value(subject=policy, predicate=RDF.type)
        data_format = self.g.value(subject=abox[self.dp], predicate=tbox.hasDTT)
        pc = PolicyChecker(policy, p_type, data_format)

        # Add initial operations and handle duties
        last_op, pc = self.init_op(policy, pc)
        last_op, pc = self.handle_duties(pc, last_op)
        pc = self.handle_attributes(pc)

        # Add final report
        report_uid = generate_unique_uri(abox)
        pc.add((last_op, tbox.nextStep, report_uid))
        pc.add((report_uid, RDF.type, tbox.Report))

        return pc

    def parse_contracts(self):
        """Parse all policies associated with a data product."""
        policies, mappings = self.read_contracts()

        for policy in policies:
            print("Parsing Policy... ", policy)
            pc = self.parse_policy(policy)
            self.g += pc

        output_path = os.path.join(base_dir, "../../../FederatedComputationalGovernance/SemanticDataModel/sdm.ttl")
        self.g.serialize(destination=output_path, format="turtle")

        return self.g


def simulate_scalability(dc_parser, rule_path, start=1, end=50, step=1):
    """Simulate scalability by averaging execution time over multiple iterations."""
    times = []
    rules_range = list(range(start, end + 1, step))
    num_iterations = 0
    for nrules in rules_range:
        print(f"Executing with {nrules} rules...")

        cumulative_time = 0
        for _ in range(num_iterations):
            start_time = time.time()  # Start timer for current number of rules
            for policy in dc_parser.read_contracts()[0]:  # Retrieve policies
                mappings = dc_parser.read_contracts()[1]  # Retrieve attribute mappings
                pc = dc_parser.parse_policy(policy)  # Parse the policy
                dc_parser.execute_rule(rule_path, pc, mappings, nrules=1)  # Execute rules
            elapsed_time = time.time() - start_time  # Calculate elapsed time for this run
            cumulative_time += elapsed_time  # Add time to cumulative total

        times.append(cumulative_time)  # Store the averaged time for plotting
        num_iterations+=1

    # Plot the results
    plt.plot(rules_range, times, marker='o')
    plt.title('Average Time Taken for Execution vs Number of Rules')
    plt.xlabel('Number of Rules Executed')
    plt.ylabel('Average Time (seconds)')
    plt.grid(True)
    plt.savefig('scalability.png')  # Save the plot


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parse Data Contracts for Data Product")
    parser.add_argument("dp", type=str, help="Data Product identifier")
    parser.add_argument("--plot", action="store_true", help="Flag to plot the times for each policy")
    args = parser.parse_args()

    print("Parsing Data Contracts for Data Product: ", args.dp)
    contract_graph = Graph().parse(os.path.join(base_dir, "../../../FederatedComputationalGovernance/SemanticDataModel/sdm.ttl"))
    dc_parser = DCParser(args.dp, contract_graph)

    rule_path = os.path.join(base_dir, "../parser/rules/OPS")
    simulate_scalability(dc_parser, rule_path, start=1, end=50, step=1)


    if args.plot:
        dc_parser.plot_times('policy_times.png')

    print("Done")
