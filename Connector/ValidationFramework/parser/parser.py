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


def generate_unique_uri(base_uri):
    unique_identifier = str(uuid.uuid4())
    return URIRef(f"{base_uri}{unique_identifier}")

def load_jsonld_data(file_path):
    with open(file_path, 'r') as f:
        return json.loads(f.read())


def timeit(func):
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
    """ Create Policy Checker """

    def __init__(self, p, dp, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        self.bind("ab", "http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/abox#")
        self.bind("tb", "http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/tbox#")
        self.URI = generate_unique_uri(abox)

        self.add((self.URI, RDF.type, tbox.PolicyChecker))
        self.add((self.URI, tbox.accordingTo, p))
        self.add((self.URI, tbox.validates, abox[dp]))
        # self.p_type = p_type.split("/")[-1]

    def get_URI(self):
        return self.URI

    def get_policy_type(self):
        return self.p_type

    def get_policy(self):
        return self.p


# Policy Parsing Class


class DCParser:

    def __init__(self, dp, graph):
        self.dp = dp
        self.g = graph
        self.attr_mappings = None

    def _validate_graph(self) -> bool:

        from pyshacl import validate
        shapes = Graph().parse(os.path.join(base_dir, 'policy_grammar.json'), format="turtle")
        conforms, report_graph, report_text = validate(self.g, shacl_graph=shapes)
        # return boolean
        return conforms

    def _read_contracts(self):

        contracts = self.g.objects(subject=abox[self.dp], predicate=tbox.hasDC)
        policies_list = []
        mappings_dict = {}
        for contract in contracts:
            # handle policies
            policies = self.g.objects(subject=contract, predicate=tbox.hasPolicy)
            for policy in policies:
                policies_list.append(policy)
            # handle mappings
            mappings = self.g.objects(subject=contract, predicate=tbox.hasMapping)
            for mapping in mappings:
                mfrom = self.g.value(subject=mapping, predicate=tbox.mfrom)
                mto = self.g.value(subject=mapping, predicate=tbox.mto)
                mappings_dict[mto] = mfrom

        self.attr_mappings = mappings_dict
        return policies_list, mappings_dict

    def executRule(self, rule_path, pc, mappings):

        for sparqlrule in os.listdir(rule_path):
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

        last_op = pc.value(subject=pc.get_URI(), predicate=tbox.nextStep)
        while last_op:
            if not pc.value(subject=last_op, predicate=tbox.nextStep):
                break
            last_op = pc.value(subject=last_op, predicate=tbox.nextStep)
        return last_op

    def _initOP(self, policy, pc):
        """
        :param IR:
        :param policy:
        :return:
        """

        initOPrules = os.path.join(base_dir, "rules/initOP")
        mappings = {
            "dp": abox[self.dp],
            "pc": pc.get_URI(),
            "op_uri": generate_unique_uri(abox),
        }

        pc = self.executRule(initOPrules, pc, mappings)

        return self.get_last_op(pc), pc

    def _handle_attributes(self, pc):
        operation = pc.get_URI()
        while operation:
            if pc.value(subject=operation, predicate=tbox.hasInput):
                attributes = pc.objects(subject=operation, predicate=tbox.hasInput)
                for attribute in attributes:
                    if attribute in self.attr_mappings.keys():
                        pc.remove((operation, tbox.hasInput, attribute))
                        pc.add((operation, tbox.hasInput, self.attr_mappings[attribute]))
            operation = pc.value(subject=operation, predicate=tbox.nextStep)
        return pc

    def _handle_policy_patterns(self, pc, initOP):

        initOPrules = os.path.join(base_dir, "rules/OPS")
        mappings = {
            "dp": abox[self.dp],
            "pc": pc.get_URI(),
            "op_uri": generate_unique_uri(abox),
            "last_op": initOP,
            "policy_uri": pc.get_policy(),
        }
        pc = self.executRule(initOPrules, pc, mappings)

        return self.get_last_op(pc), pc

    def _parse_policy(self, policy):

        pc = PolicyChecker(policy, self.dp)

        last_op, pc = self._initOP(policy, pc)

        last_op, pc = self._handle_policy_patterns(pc, last_op)
        pc = self._handle_attributes(pc)

        # Report
        report_uid = generate_unique_uri(abox)
        pc.add((last_op, tbox.nextStep, report_uid))
        pc.add((report_uid, RDF.type, tbox.Report))
        # DUTY
        return pc

    def parse_contracts(self):

        # validate policies
        # if self._validate_graph() == True:
        # get policies
        policies, mappings = self._read_contracts()

        for policy in policies:
            pc = self._parse_policy(policy)
            self.g = self.g + pc

        self.g.serialize(
            destination=os.path.join(base_dir, "../../../FederatedComputationalGovernance/SemanticDataModel/sdm.ttl"),
            format="turtle")

        return self.g


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parse Data Contracts for Data Product")
    parser.add_argument("dp", type=str, help="Data Product identifier")
    parser.add_argument("--plot", action="store_true", help="Flag to plot the times for each policy")
    args = parser.parse_args()

    print("Parsing Data Contracts for Data Product: ", args.dp)
    contract_graph = Graph().parse(os.path.join(base_dir, "../../../FederatedComputationalGovernance/SemanticDataModel/sdm.ttl"))
    dc_parser = DCParser(args.dp, contract_graph)
    dc_parser.parse_contracts()

    if args.plot:
        dc_parser.plot_times('policy_times.png')

    print("Done")
