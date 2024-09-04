from rdflib import *


def create_graph():
    """
    This function returns a graph object with the necessary prefixes
    :return: RDF Graph
    """
    g = Graph()
    g.bind('tb', 'http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/tbox#')
    g.bind('ab', 'http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/abox#')
    return g