from flask import Flask, render_template, request, redirect, url_for, jsonify
from rdflib import *
from pyvis.network import Network

from Connector.ValidationFramework.translator.translator import base_dir

app = Flask(__name__)


SDM = Graph().parse('../SemanticDataModel/sdm.ttl', format='turtle')

@app.route('/')
def index():

    query = f'''
    PREFIX tb: <http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/tbox#>
    PREFIX ab: <http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/abox#>
    PREFIX odrl: <http://www.w3.org/ns/odrl/2/>

    SELECT ?dp
    WHERE {{
        ?dp a tb:DataProduct .
    }}
    '''

    q_res = SDM.query(query)

    data_products = [row.dp.split("#")[1] for row in q_res]
    data_products

    return render_template('index.html', data_products=data_products)

@app.route('/visualize', methods=['POST'])
def visualize():
    uri = request.form.get('uri')
    if not uri:
        return "URI is required", 400

    # Extract and visualize the subgraph
    abox = Namespace('http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/abox#')
    net = extract_subgraph(SDM, abox[uri])
    path = 'templates/graph.html'
    net.save_graph(path)
    return render_template('graph.html', path=url_for('static', filename='graph.html'))

# Define your namespaces
TB = Namespace("http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/tbox#")
ODRL = Namespace("http://www.w3.org/ns/odrl/2/")
DCAT = Namespace("https://www.w3.org/ns/dcat#")


def extract_subgraph(graph, uri):
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")
    subject = URIRef(uri)

    # Initialize set to track visited nodes to prevent infinite loops
    visited = set()

    # Start recursive traversal from the policy checker
    traverse_policy_checker(graph, subject, net, visited)

    return net


def traverse_policy_checker(graph, node, net, visited):
    """Recursive function to process policy checkers and their related steps."""
    if node in visited:
        return
    visited.add(node)

    # Add the node to the network
    net.add_node(str(node), label=str(node).split('#')[-1], title=str(node), color="blue")

    # Define the properties we are interested in
    relevant_properties = {TB.hasType, TB.nextStep, TB.validates, TB.hasAbstract, TB.hasOutput, TB.hasParameter, TB.hasAttribute, TB.hasInput, TB.result}

    # Traverse all related triples for the current node
    for s, p, o in graph.triples((node, None, None)):
        if p in relevant_properties:
            net.add_node(str(o), label=str(o).split('#')[-1], title=str(o), color="red")
            net.add_edge(str(s), str(o), title=str(p).split('#')[-1])

            # Recurse only if the object is an operation, report, or another policy checker
            if graph.value(o, RDF.type) in [TB.Operation, TB.Report, TB.PolicyChecker, TB.initOperation]:
                traverse_policy_checker(graph, o, net, visited)


@app.route('/create_policy', methods=['POST'])
def create_policy():
    policy_id = request.form['policy_id']
    target = request.form['target']
    action = request.form['action']

    policy = {
        "@context": {
            "@base": "http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/",
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
            "odrl": "http://www.w3.org/ns/odrl/2/",
            "dcat": "http://www.w3.org/ns/dcat#",
            "tb": "http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/tbox#",
            "ab": "http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/abox#"
        },
        "@id": f"ab:{policy_id}",
        "@type": "odrl:Privacy",
        "odrl:duty": [
            {
                "@id": "ab:AnonDuty",
                "@type": "odrl:Duty",
                "odrl:target": {
                    "@id": f"ab:{target}"
                },
                "odrl:action": [
                    {
                        "@id": f"odrl:{action}",
                        "@type": "odrl:Action"
                    }
                ]
            }
        ]
    }

    return jsonify(policy)

if __name__ == '__main__':
    app.run(debug=True)
