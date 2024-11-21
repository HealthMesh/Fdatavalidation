import types
from datetime import datetime

from flask import Flask, request, jsonify
from rdflib import Graph, Namespace, URIRef
import os, sys
import logging
from typing import Dict, List, Optional
import json


tbox = Namespace('http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/tbox#')
abox = Namespace('http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/abox#')

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
sys.path.append(base_dir)


from Connector.ValidationFramework.parser.parser import DCParser
from Connector.ValidationFramework.translator.translator import PCTranslator


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SDM_PATH = os.path.join(BASE_DIR, '../../../FederatedComputationalGovernance/SemanticDataModel/sdm.ttl')

# Namespaces
TB = Namespace("http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/tbox#")
AB = Namespace("http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/abox#")
ODRL = Namespace("http://www.w3.org/ns/odrl/2/")


class ValidationFramework:
    def __init__(self):
        self.sdm = Graph()
        self.load_semantic_model()

    def load_semantic_model(self):
        """Load the semantic data model from TTL file"""
        try:
            self.sdm.parse(SDM_PATH, format='turtle')
            logging.info("Semantic model loaded successfully")
        except Exception as e:
            logging.error(f"Error loading semantic model: {str(e)}")
            raise

    def parse_contracts(self, dp: str) -> Graph:
        """Parse contracts for a given data product"""
        C = DCParser(dp, self.sdm).parse_contracts()
        return C

    def translate_policy_checker(self, pc: str) -> types.FunctionType:
        """Translate a policy checker to a different format"""
        try:
            udf = PCTranslator(pc, self.sdm).translate()
            initOP = self.sdm.value(subject=abox[pc.split("#")[1]], predicate=tbox.nextStep)
            path = self.sdm.value(subject=initOP, predicate=tbox.hasInput)
            try:
                print("entrou")
                udf(path)
            except Exception as e:
                logging.error(f"Error executing translated policy checker: {str(e)}")
                return
            return udf

        except Exception as e:
            logging.error(f"Error translating policy checker: {str(e)}")
            return

    def query_policy_checkers(self, dp: str) -> List[str]:
        """Query policy checkers for a given data product"""
        query = """
        PREFIX tb: <http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/tbox#>
        PREFIX ab: <http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/abox#>
        PREFIX odrl: <http://www.w3.org/ns/odrl/2/>

        SELECT ?pc ?ds
        WHERE {
            ?pc a tb:PolicyChecker .
            ?pc tb:validates ?dp .
            FILTER (?dp = ab:%s)        
        }
        """

        results = self.sdm.query(query % dp)
        return [str(row.pc).split("#")[1] for row in results]

    def get_validation_reports(self) -> List[Dict]:
        """Get all validation reports"""
        query = """
        PREFIX odrl: <http://www.w3.org/ns/odrl/2/>
        PREFIX tb: <http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/tbox#>
        PREFIX ab: <http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/abox#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?dp ?p ?pc ?result
        WHERE {
            ?pc a tb:PolicyChecker .
            ?pc tb:validates ?dp .
            ?pc tb:accordingTo ?p .
            ?pc tb:nextStep+ ?rep .
            ?rep a tb:Report .
            ?rep tb:result ?result .
        }
        """

        results = self.sdm.query(query)
        return [{
            'data_product': str(row['dp']).split('#')[1],
            'policy': str(row['p']).split('#')[1],
            'policy_checker': str(row['pc']).split('#')[1],
            'result': row['result'].value
        } for row in results]

    def get_data_products(self) -> List[str]:
        """Get all data products with data contracts"""
        query = """
        PREFIX tb: <http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/tbox#>
        PREFIX ab: <http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/abox#>
        PREFIX odrl: <http://www.w3.org/ns/odrl/2/>

        SELECT DISTINCT ?dp
        WHERE {
            ?dp tb:hasDC ?dc .
        }
        """

        results = self.sdm.query(query)
        return [str(row['dp']).split('#')[1] for row in results]


    def add_policy(self, policy) -> Graph:
        instances = Graph().parse(data=policy, format='json-ld')
        self.sdm += instances
        self.sdm.serialize(destination=SDM_PATH, format='turtle')
        return instances.identifier


validation_framework = ValidationFramework()




@app.route('/test', methods=['GET'])
def test():


    validation_result = True

    if validation_result:
        return jsonify({'status': 'true'})
    else:
        return jsonify({'status': 'false'})


@app.route('/data-products', methods=['GET'])
def get_data_products():
    try:
        products = validation_framework.get_data_products()
        return jsonify({
            'data_products': products,
            'count': len(products)
        })
    except Exception as e:
        logging.error(f"Error getting data products: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/policies', methods=['PUT'])
def add_policy():
    try:
        policy_data = request.get_json()

        # Maybe add SHACL shapes for validation
        policy_id = validation_framework.add_policy(policy_data)

        return jsonify({
            'message': 'Policy added successfully',
            'policy_id': policy_id,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logging.error(f"Error adding policy: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/policy-checkers/<dp_id>', methods=['GET'])
def get_policy_checkers(dp_id: str):
    try:
        checkers = validation_framework.query_policy_checkers(dp_id)
        return jsonify({
            'data_product': dp_id,
            'policy_checkers': checkers,
            'count': len(checkers)
        })
    except Exception as e:
        logging.error(f"Error getting policy checkers: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/validation-reports', methods=['GET'])
def get_validation_reports():
    try:
        reports = validation_framework.get_validation_reports()
        return jsonify({
            'reports': reports,
            'count': len(reports)
        })
    except Exception as e:
        logging.error(f"Error getting validation reports: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/parse-contracts/<dp_id>', methods=['GET'])
def parse_contracts(dp_id: str):
    try:
        Checkers = ValidationFramework().parse_contracts(dp_id)
        return jsonify({
            'message': f'Contracts parsed successfully for data product {dp_id}',
            'status': 'success'
        })
    except Exception as e:
        logging.error(f"Error parsing contracts: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/translate/<pc_id>', methods=['GET'])
def translate_policy_checker(pc_id: str):
    try:
        udf = ValidationFramework().translate_policy_checker(pc_id)
        return jsonify({
            'message': f'Policy checker {pc_id} translated successfully',
            'status': 'success'
        })
    except Exception as e:
        logging.error(f"Error translating policy checker: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)