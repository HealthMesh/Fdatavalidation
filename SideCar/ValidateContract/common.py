from rdflib import *

class GraphPC:

    def __init__(self):

        self.g = Graph()
        tbox = Namespace('http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/tbox#')
        abox = Namespace('http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/abox#')
        dcat = Namespace('https://www.w3.org/ns/dcat#')
        dcterms = Namespace('http://purl.org/dc/terms/')
        tb = Namespace("http://www.semanticweb.org/acraf/ontologies/2021/0/SDM#")
        odrl = Namespace("http://www.w3.org/ns/odrl/2/")
        prov = Namespace("http://www.w3.org/ns/prov#")
        dqv = Namespace("http://www.w3.org/ns/dqv#")

        self.dp = "http://www.semanticweb.org/ontologies/2018/9/untitled-ontology-3#"
        self.attr_mappings = {}