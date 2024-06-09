import uuid
from rdflib import *
from owlrl import *
import json
import os

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


class PolicyChecker(Graph):
    """ Create Policy Checker """

    def __init__(self, p, p_type, format, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        self.bind("ab", "http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/abox#")
        self.bind("tb", "http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/tbox#")
        self.URI = generate_unique_uri(abox)
        self.add((self.URI, RDF.type, tbox.PolicyChecker))
        self.add((self.URI, tbox.validates, p))
        self.add((self.URI, tbox.hasType, format))

        self.p_type = p_type.split("/")[-1]

    def get_URI(self):
        return self.URI

    def get_policy_type(self):
        return self.p_type

    def get_policy(self):
        return self.p

class DCParser:
    """
    Parse Policies of Data Contracts to Policy Checkers
    """

    def __init__(self, dp, graph):
        self.dp = dp
        self.g = graph
        self.attr_mappings = None

    def _validate_graph(self) -> bool:
        """
        Validate the policies grammar is compliant with the grammar defined
        :return: conformance/non-conformance
        """
        from pyshacl import validate
        shapes = Graph().parse(os.path.join(base_dir, 'policy_grammar.json'), format="turtle")
        conforms, report_graph, report_text = validate(self.g, shacl_graph=shapes)
        # return boolean
        return conforms

    def _read_contracts(self):
        """
        Get the policies associated with a data product
        :return: list of policies
        """
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

                results = self.g.query(rule)

                result_graph = Graph()

                for triple in results:
                    result_graph.add(triple)

                pc += result_graph

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

        initOPrules = "/home/acraf/psr/tfm/Fdatavalidation/SideCar/ValidateContract/parser/rules/initOP"
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
            if pc.value(subject=operation, predicate=tbox.hasAttribute):
                attribute = pc.value(subject=operation, predicate=tbox.hasAttribute)
                if attribute in self.attr_mappings.keys():
                    pc.remove((operation, tbox.hasAttribute, None))
                    pc.add((operation, tbox.hasAttribute, self.attr_mappings[attribute]))
            operation = pc.value(subject=operation, predicate=tbox.nextStep)
        return pc

    def _handle_duties(self, pc, initOP):
        """
        :param pc:
        :param policy:
        :return:
        """

        initOPrules = "/home/acraf/psr/tfm/Fdatavalidation/SideCar/ValidateContract/parser/rules/OPS"
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
        """
        Parse the policy to intermediate representation
        :param policy: policy to parse
        :return: None
        """

        # get policy type
        p_type = self.g.value(subject=policy, predicate=RDF.type)
        # data format
        format = self.g.value(subject=abox[self.dp], predicate=tbox.hasDTT)
        # create policy checker graph
        pc = PolicyChecker(policy, p_type, format)

        # add initOperation
        last_op, pc = self._initOP(policy, pc)

        # handle Duties
        last_op, pc = self._handle_duties(pc, last_op)
        pc = self._handle_attributes(pc)
        # Report
        report_uid = generate_unique_uri(abox)
        pc.add((last_op, tbox.nextStep, report_uid))
        pc.add((report_uid, RDF.type, tbox.Report))
        # DUTY
        return pc

    def parse_contracts(self):
        """
        Get the policies associated with a data product
        :return: list of policies
        """

        # validate policies
        # if self._validate_graph() == True:
        # get policies
        policies, mappings = self._read_contracts()

        for policy in policies:
            pc = self._parse_policy(policy)
            self.g = self.g + pc

        # self.g.serialize(destination=os.path.join(base_dir, "generated.ttl"), format="turtle")

        #self.g.serialize(destination=os.path.join(base_dir, "../../../FederatedComputationalGovernance/SemanticDataModel/sdm.ttl"),
         #   format="turtle")

        return self.g
        # print(pc.serialize(format="turtle"))
        # else :
        # <   print("The policies do not comply with the grammar")


