import streamlit as st
import rdflib
from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS
import json
import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import tempfile
from streamlit_agraph import agraph, Node, Edge, Config


class SDMManager:
    def __init__(self):
        self.base_dir = os.getcwd()
        self.tb = Namespace('http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/tbox#')
        self.ab = Namespace('http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/abox#')
        self.load_graph()

    def load_graph(self):
        """Load the semantic data model graph"""
        self.g = Graph()
        try:
            self.g.parse(os.path.join(self.base_dir, 'SemanticDataModel/sdm.ttl'), format='ttl')
        except Exception as e:
            st.error(f"Error loading graph: {str(e)}")
            self.g = Graph()

    def save_graph(self):
        """Save the current graph state"""
        try:
            self.g.serialize(destination=os.path.join(self.base_dir, 'SemanticDataModel/sdm.ttl'), format='ttl')
            st.success("Graph saved successfully!")
        except Exception as e:
            st.error(f"Error saving graph: {str(e)}")

    def add_common_data_model(self, name, description, domain):
        """Add a new common data model to the graph"""
        cmd_uri = self.ab[name]
        self.g.add((cmd_uri, RDF.type, self.tb.CommonDataModel))
        self.g.add((cmd_uri, RDFS.label, Literal(name)))
        self.g.add((cmd_uri, RDFS.comment, Literal(description)))
        self.g.add((cmd_uri, self.tb.domain, Literal(domain)))
        self.save_graph()

    def add_policy(self, name, policy_type, constraint):
        """Add a new policy to the graph"""
        policy_uri = self.ab[name]
        self.g.add((policy_uri, RDF.type, self.tb.Policy))
        self.g.add((policy_uri, RDFS.label, Literal(name)))
        self.g.add((policy_uri, self.tb.policyType, Literal(policy_type)))
        self.g.add((policy_uri, self.tb.constraint, Literal(constraint)))
        self.save_graph()

    def visualize_graph(self):
        """Convert RDF graph to visualization format"""
        nodes = []
        edges = []

        # Create nodes for each subject
        for s, p, o in self.g:
            if isinstance(s, URIRef):
                node_id = s.split('#')[-1]
                node_type = self.g.value(s, RDF.type)
                node_type = node_type.split('#')[-1] if node_type else "Unknown"
                nodes.append(Node(id=node_id, label=node_id, size=25, color=self.get_node_color(node_type)))

            if isinstance(o, URIRef):
                node_id = o.split('#')[-1]
                node_type = self.g.value(o, RDF.type)
                node_type = node_type.split('#')[-1] if node_type else "Unknown"
                nodes.append(Node(id=node_id, label=node_id, size=25, color=self.get_node_color(node_type)))

        # Create edges
        for s, p, o in self.g:
            if isinstance(s, URIRef) and isinstance(o, URIRef):
                source = s.split('#')[-1]
                target = o.split('#')[-1]
                predicate = p.split('#')[-1]
                edges.append(Edge(source=source, target=target, label=predicate))

        # Remove duplicates
        nodes = list({node.id: node for node in nodes}.values())

        return nodes, edges

    @staticmethod
    def get_node_color(node_type):
        """Return color based on node type"""
        color_map = {
            'CommonDataModel': '#1f77b4',
            'Policy': '#2ca02c',
            'DataProduct': '#ff7f0e',
            'Unknown': '#7f7f7f'
        }
        return color_map.get(node_type, '#7f7f7f')


class SDMMetadataVisualizer:
    def __init__(self):
        self.base_dir = os.getcwd()
        self.tb = Namespace('http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/tbox#')
        self.ab = Namespace('http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/abox#')
        self.load_graph()


        self.color_scheme = {
            'Policy': '#90EE90',
            'CommonDataModel': '#FFEFD5',
            'DataProduct': '#FFB6C1',
            'Attribute': '#FFB6C1',
            'DTT': '#FFB6C1',
            'DC': '#FFB6C1',
            'TA': '#FFB6C1'
        }

    def load_graph(self):

        self.g = Graph()
        try:
            self.g.parse(os.path.join(self.base_dir, 'SemanticDataModel/sdm.ttl'), format='ttl')
        except Exception as e:
            st.error(f"Error loading graph: {str(e)}")
            self.g = Graph()

    def get_node_type(self, uri):

        node_type = self.g.value(uri, RDF.type)
        if node_type:
            return node_type.split('#')[-1]
        return "Unknown"

    def visualize_metadata(self):

        nodes = []
        edges = []
        added_nodes = set()

        query = """
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX tb: <http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/tbox#>

        SELECT DISTINCT ?s ?p ?o
        WHERE {
            {
                ?s rdf:type tb:DataProduct .
                ?s ?p ?o .
            } UNION {
                ?s rdf:type odrl:Policy .
                ?s ?p ?o .
            } UNION {
                ?s rdf:type tb:CommonDataModel .
                ?s ?p ?o .
            }
        }
        """

        results = self.g.query(query)

        for s, p, o in results:
            source_id = s.split('#')[-1]
            predicate = p.split('#')[-1]

            if source_id not in added_nodes:
                source_type = self.get_node_type(s)
                nodes.append(Node(
                    id=source_id,
                    label=source_id,
                    size=25,
                    color=self.color_scheme.get(source_type, '#808080')
                ))
                added_nodes.add(source_id)

            if isinstance(o, URIRef):
                target_id = o.split('#')[-1]
                if target_id not in added_nodes:
                    target_type = self.get_node_type(o)
                    if target_type in self.color_scheme:  # Only add relevant node types
                        nodes.append(Node(
                            id=target_id,
                            label=target_id,
                            size=25,
                            color=self.color_scheme.get(target_type, '#808080')
                        ))
                        added_nodes.add(target_id)

                        edges.append(Edge(source=source_id, target=target_id, label=predicate))

        return nodes, edges



def add_code_metadata(self, implementation_id, code, parameters, dependencies, operation, dataset_type, returns=None):

    imp_uri = self.ab[implementation_id]
    code_uri = self.ab[f"{implementation_id}Code"]

    self.g.add((imp_uri, RDF.type, self.tb.Implementation))


    self.g.add((code_uri, RDF.type, self.tb.Code))
    self.g.add((code_uri, self.tb.code, Literal(code)))
    self.g.add((imp_uri, self.tb.hasCode, code_uri))


    for i, param in enumerate(parameters):
        param_uri = self.ab[f"{implementation_id}CodeParam{i + 1}"]
        self.g.add((param_uri, RDF.type, self.tb.Parameter))
        self.g.add((param_uri, self.tb.name, Literal(param['name'])))
        self.g.add((param_uri, self.tb.type, Literal(param['type'])))
        self.g.add((imp_uri, self.tb.hasParameters, param_uri))

    for i, dep in enumerate(dependencies):
        dep_uri = self.ab[f"{implementation_id}CodeDep{i + 1}"]
        self.g.add((dep_uri, RDF.type, self.tb.Library))
        self.g.add((dep_uri, self.tb.name, Literal(dep)))
        self.g.add((imp_uri, self.tb.dependsOn, dep_uri))

    op_uri = self.ab[operation]
    self.g.add((op_uri, RDF.type, self.tb.Operation))
    self.g.add((imp_uri, self.tb.forOp, op_uri))

    type_uri = self.ab[dataset_type]
    self.g.add((type_uri, RDF.type, self.tb.DatasetTypeTemplate))
    self.g.add((imp_uri, self.tb.forType, type_uri))

    if returns:
        self.g.add((imp_uri, self.tb.returns, Literal(returns)))

    self.save_graph()



def main():
    st.set_page_config(page_title="Federated Team", layout="wide")
    st.title("Federated Team Admin Console")

    sdm_manager = SDMManager()

    tabs = st.tabs(["SDM", "Common Data Models", "Policies", "Query", "Rewrite Rules", "Code Metadata"])

    with tabs[0]:
        st.header("Graph Visualization")
        visualizer = SDMMetadataVisualizer()


        st.sidebar.header("Legend")
        for node_type, color in visualizer.color_scheme.items():
            st.sidebar.markdown(
                f'<div style="background-color: {color}; padding: 10px; margin: 5px; border-radius: 5px;">{node_type}</div>',
                unsafe_allow_html=True
            )

        nodes, edges = visualizer.visualize_metadata()

        config = Config(
            width=1000,
            height=800,
            directed=True,
            physics=False,
            hierarchical=True,
            hierarchical_sort_method="directed",
            link_length=100,
            nodeHighlightBehavior=True,
            highlightColor="#F7A7A6",
            staticGraphWithDragAndDrop=True
        )

        agraph(nodes=nodes, edges=edges, config=config)

    with tabs[1]:
        st.header("Add Common Data Model")

        name = st.text_input("Name")
        description = st.text_area("Description")
        domain = st.text_input("Domain")

        if st.button("Add Common Data Model"):
            if name and description and domain:
                sdm_manager.add_common_data_model(name, description, domain)
                st.success(f"Added Common Data Model: {name}")
            else:
                st.error("Please fill in all fields")

    with tabs[2]:
        st.header("Add Policy")

        policy_name = st.text_input("Policy Name")
        policy_type = st.selectbox("Policy Type", ["Privacy", "Access", "Usage"])
        constraint = st.text_area("Constraint")

        if st.button("Add Policy"):
            if policy_name and constraint:
                sdm_manager.add_policy(policy_name, policy_type, constraint)
                st.success(f"Added Policy: {policy_name}")
            else:
                st.error("Please fill in all fields")


    with tabs[3]:
        st.header("SPARQL Query")

        query = st.text_area("Enter SPARQL Query", height=150)
        if st.button("Execute Query"):
            try:
                results = sdm_manager.g.query(query)
                results_df = pd.DataFrame(results)
                st.dataframe(results_df)
            except Exception as e:
                st.error(f"Query error: {str(e)}")


    with tabs[4]:
        st.header("Add Code Metadata")

        col1, col2 = st.columns(2)

        with col1:
            implementation_id = st.text_input("Implementation ID (e.g., Imp1)")
            code = st.text_area("Code", height=100)
            returns = st.text_input("Return Type (optional)")
            operation = st.selectbox("Operation", [
                "LoadData",
                "anonymize",
                "qM",
                "Constraint",
                "FairnessDegree"
            ])
            dataset_type = st.selectbox("Dataset Type", ["Tabular", "Image"])

        with col2:
            st.subheader("Parameters")
            num_params = st.number_input("Number of Parameters", min_value=0, max_value=5, value=1)
            parameters = []
            for i in range(num_params):
                st.markdown(f"**Parameter {i + 1}**")
                param_name = st.text_input(f"Name {i + 1}", key=f"param_name_{i}")
                param_type = st.text_input(f"Type {i + 1}", key=f"param_type_{i}")
                if param_name and param_type:
                    parameters.append({"name": param_name, "type": param_type})

            st.subheader("Dependencies")
            num_deps = st.number_input("Number of Dependencies", min_value=0, max_value=5, value=1)
            dependencies = []
            for i in range(num_deps):
                dep = st.text_input(f"Library {i + 1}", key=f"dep_{i}")
                if dep:
                    dependencies.append(dep)

        if st.button("Add Code Metadata"):
            if implementation_id and code and parameters and dependencies and operation and dataset_type:
                try:
                    sdm_manager.add_code_metadata(
                        implementation_id=implementation_id,
                        code=code,
                        parameters=parameters,
                        dependencies=dependencies,
                        operation=operation,
                        dataset_type=dataset_type,
                        returns=returns if returns else None
                    )
                    st.success(f"Added Code Metadata: {implementation_id}")
                except Exception as e:
                    st.error(f"Error adding code metadata: {str(e)}")
            else:
                st.error("Please fill in all required fields")


        st.subheader("Existing Code Metadata")
        query = """
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX tb: <http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/tbox#>
        PREFIX ab: <http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/abox#>

        SELECT ?imp ?code ?op ?type ?returns
        WHERE {
            ?imp rdf:type tb:Implementation .
            ?imp tb:hasCode ?codeUri .
            ?codeUri tb:code ?code .
            ?imp tb:forOp ?opUri .
            ?opUri rdf:type tb:Operation .
            ?imp tb:forType ?typeUri .
            OPTIONAL { ?imp tb:returns ?returns }
        }
        """

        try:
            results = sdm_manager.g.query(query)
            df = pd.DataFrame(results, columns=['Implementation', 'Code', 'Operation', 'Type', 'Returns'])
            st.dataframe(df)
        except Exception as e:
            st.error(f"Error querying code metadata: {str(e)}")


if __name__ == "__main__":
    main()