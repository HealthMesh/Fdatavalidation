import pandas as pd
from rdflib import Graph, Namespace, Literal, RDF, URIRef
from rdflib.namespace import XSD
import json

# JSON-LD structure
json_ld = {
    "@context": {
        "dqv": "http://www.w3.org/ns/dqv#",
        "odrl": "http://www.w3.org/ns/odrl/2/",
        "xsd": "http://www.w3.org/2001/XMLSchema#",
        "ab": "http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/abox#"
    },
    "@id": "ab:p2",
    "@type": "dqv:QualityPolicy",
    "odrl:duty": [
        {
            "@id": "ab:CompleteDuty",
            "@type": "odrl:Duty",
            "odrl:target": "ab:Age_at_scan_years",
            "odrl:constraint": [
                {
                    "@id": "odrl:c1",
                    "@type": "odrl:Constraint",
                    "odrl:operator": "odrl:gteq",
                    "odrl:rightOperand": {
                        "@value": "99",
                        "@type": "xsd:decimal"
                    },
                    "odrl:leftOperand": {
                        "@id": "ab:NonNullValuesCount",
                        "@type": "dqv:QualityMeasurement"
                    }
                }
            ]
        }
    ]
}

# Create a Graph
g = Graph()
g.parse(data=json.dumps(json_ld), format='json-ld')

# Define namespaces
DQV = Namespace("http://www.w3.org/ns/dqv#")
ODRL = Namespace("http://www.w3.org/ns/odrl/2/")
XSD = Namespace("http://www.w3.org/2001/XMLSchema#")
AB = Namespace("http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/abox#")

def extract_info(g):
    # Query for target column, operator, rightOperand, and leftOperand
    query = """
    PREFIX dqv: <http://www.w3.org/ns/dqv#>
    PREFIX odrl: <http://www.w3.org/ns/odrl/2/>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    PREFIX ab: <http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/abox#>

    SELECT ?target ?operator ?rightOperand ?leftOperand
    WHERE {
        ?policy a dqv:QualityPolicy .
        ?policy odrl:duty ?duty .
        ?duty odrl:target ?target .
        ?duty odrl:constraint ?constraint .
        ?constraint odrl:operator ?operator .
        ?constraint odrl:rightOperand ?rightOperand .
        ?constraint odrl:leftOperand ?leftOperand .
    }
    """
    results = g.query(query)

    for row in results:
        target = row.target.split(":")[1]
        operator = row.operator.split(":")[1]
        right_operand = float(row.rightOperand)
        left_operand = row.leftOperand.split(":")[1]
        return target, operator, right_operand, left_operand

target_column, operator, constraint_value, measurement_type = extract_info(g)

df = pd.read_csv('/home/acraf/psr/tfm/Fdatavalidation/DataProductLayer/DataProduct/Data/Explotation/UPENN-GBM_clinical_info_v2.1.csv')

measurement_type = str(measurement_type).split("#")[1]
if measurement_type == "NonNullValuesCount":
    non_null_count = (df[target_column].count().sum() / len(df)) * 100
    if operator == "gteq":
        is_satisfied = non_null_count >= constraint_value

    print(f"Constraint is satisfied: {is_satisfied}")

