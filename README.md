# Data Validation Worfklow


This repository contains a prototype wokrflow for automating data validation in data spaces. The workflow is derived from an architectural framework which uses knowledge graphs as basis for automation. 

## Architectural Framework Overview

![](\\wsl$\Ubuntu-24.04\home\acraf\psr\Fdatavalidation\demo\demo_images\framework-workflow.png)

The approach is build upon a federated architectural framework and encompasses the following layers:
- **Data Product Layer**: Contains data products and their infrastructure extensions (e.g., connectors).
- **Data Platform Layer**}: Functions as a central gateway for data management processes, including data asset registration and analytical studies.
- **Federated Computational Governance Layer**: Establishes guidelines and artifacts for data product management and governance.


## Prototype Folders and Files

| Folder                           | File                                            | Description                                                         | Defined by     |
|----------------------------------|-------------------------------------------------|---------------------------------------------------------------------|:---------------|
| FederatedComputationalGovernance | ComputationalCatalogues/p*.json                 | Federation defined Policies in JSON-LD                              | Federated Team |
| FederatedComputationalGovernance | ComputationalCatalogues/common_data_models.json | Common Data Models in JSON-LD                                       | Federated Team |
| FederatedComputationalGovernance | FederatedTeam/tbox.ttl                          | Terminology Box for Semantic Data Model                             | Federated Team |
| FederatedComputationalGovernance | SemanticDataModel/sdm.ttl                       | Semantic Data Model with all annotations                            | Federated Team |
| DataProductLayer                 | DataProduct1                                    | CSV with Patient Demographics                                       |                |
| DataProductLayer                 | DataProduct2                                    | DICOM Image                                                         |                |
| DataProductLayer                 | DataProduct3                                    | ML Model                                                            |                |
| DataPlatformLayer                | Registration/profiler.ipynb                     | Notebook to automate boostraping of data sources                    |                |
| DataPlatformLayer                | Registration/integrator.ipynb                   | Notebook to generate mappings from DataProducts to CommonDataModels |                |
| DataPlatformLayer                | Registration/dpX.json                           | Data Products integrated with Data Contracts                        |                |
| Connector                        | parser/parser.ipynb                             | Parser implementation p                                             |                |
| Connector                        | parser/rules                                    | Graph Transformation Rules as SPARQL CONSTRUCT queries              |                |
| Connector                        | translator/translator.ipynb                     | Translator implementation                                           |                |
| Connector                        | translator/code_metadata.json                   | Code metadata in JSON-LD                                            |                |
| Connector                        | experiments/policy_times                        | Measure processing times                                            |                |
| Connector                        | experiments/transforming_bottleneck             | Workflow vs RDFValidation                                           |                |
| Connector                        | experiments/ecosystem                           | Measure parsing scalability                                         |                |


## Reproduce workflow

1. Clone the repository
2. Install required dependencies using `pip install -r requirements.txt`
3. run **demo.ipynb** in the demo folder to see the prototype workflow in action


## Key Features
- Policy-based data validation~~
- Support for multi-modal data
- Tracability and Transparency


## Future Work

- Support for more expressive policy patterns and use cases
- Integration and deployement within existing Data Space components (e.g., Eclipse DataSpace Components)
- Optimizations and user-framework interaction interfaces


