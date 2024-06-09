import sys
import time
sys.path.insert(0, '/home/acraf/psr/tfm/Fdatavalidation/SideCar/ValidateContract/parser')
import policy_parser
from rdflib import *

dp = "UPENN-GBM_clinical_info_v2"

graph = Graph().parse("/home/acraf/psr/tfm/Fdatavalidation/FederatedComputationalGovernance/SemanticDataModel/sdm.ttl")

# Create an instance of DCParser
parser = policy_parser.DCParser(dp, graph)

# Record the start time
start_time = time.time()

# Execute the method
parser.parse_contracts()

# Record the end time
end_time = time.time()

# Calculate the execution time
execution_time = end_time - start_time

print(f"The execution time of the parse_contracts method is {execution_time} seconds.")