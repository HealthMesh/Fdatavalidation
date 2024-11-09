# Data Validation Framework


A data governance framework for automating data validation in federated data ecosystems or data spaces.

## Overview

This framework provides a prototype implementation for validating data across federated ecosystems. The framework is based on the building blocks
of our previous federated architectural framework and encompasses  the following layers:
- **Data Product Layer**: Contains data products and their infrastructure extensions (e.g., connectors).
- **Data Platform Layer**}: Functions as a central gateway for data management processes, including data asset registration and analytical studies.
- **Federated Computational Governance Layer**: Establishes guidelines and artifacts for data product management and governance.


![demo/demo_images/Prototype-Approach.drawio.png](demo/demo_images/Prototype-Approach.drawio.png)

## Prototype Folders and Files

- **Federated Computational Governance**
  - Computational Catalogues
    - pX.json: Policies in JSON-LD
  - Global Definitions
    - common_data_models.json: Common Data Models in JSON-LD
  - FederatedTeam
    - tbox.ttl: Terminology Box of the framework
  - SemanticDataModel
    - sdm.ttl: Semantic Data Model with Federation Metadata

- **Data Product Layer**
  - Data Product (CSV with Patient Demographics)
  - Data Product 2 (DICOM Image)

- **Data Platform Layer**
  - Registration
    - profiler.ipynb: Script to extract metadata from data products
  - Integrator
    - integrator.ipynb: Script to integrate data products
    - dpX.json: Data Product mappings and agreed Policies

- **Connector**
  - ValidationFramework
    - parser
      - parser.ipynb: Script to generate Policy Checkers
      - rules: SPARQL Construct rules for Graph Tranformation
    - translator
      - translator.ipynb: Script to translate Policy Checkers to UDF
    - experiments
      - ecoystem: Ecoystem simulation
      - transforming_bottleneck: Framework comparison vs transforming to RDF
      - policy_times: Measuring policy processing times

## Getting Started

1. Clone the repository
2. Install required dependencies using `pip install -r requirements.txt`
3. run demo.ipynb in the demo folder to see the prototype components


## Key Features

- Policy-based data validation
- Support for multi-modal data
- Tracability and Transparency


## Future Work

- Integration within Data Space Connectors
- Add more policy patterns

## Contributing
Guidelines for contributing to this project will be added soon.

## License

[License information to be added]

## Contact

emaiL: achraf.hmimou@upc.edu