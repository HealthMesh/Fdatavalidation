from rdflib import *
from owlrl import *
import os
import json
import uuid
import argparse
import time
import matplotlib.pyplot as plt
from owlrl import *
from jinja2 import Template
import types
import pyshacl
import functools
import inspect
import autopep8
import json
import types
import autopep8
from jinja2 import Template
from rdflib import Graph
import time



# Basedir
base_dir = os.path.dirname(os.path.realpath(__file__))

# Namespaces
tbox = Namespace('http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/tbox#')
abox = Namespace('http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/abox#')
dcat = Namespace('https://www.w3.org/ns/dcat#')
dcterms = Namespace('http://purl.org/dc/terms/')
tb = Namespace("http://www.semanticweb.org/acraf/ontologies/2021/0/SDM#")
odrl = Namespace("http://www.w3.org/ns/odrl/2/")


# Utility Functions
def add_jsonld_instances(graph, path):

    with open(path, 'r') as f:
        json_ld_data = json.loads(f.read())
        instances = Graph().parse(data=json_ld_data, format='json-ld')
        graph += instances
    return graph


# URI functions
def get_urii(uri):

    return uri.split("#")[-1]



class Implementation(type):
    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        cls.tg = cls._initialize_graph()
        cls.ex = Graph()
        cls.translation = None
        cls.implementation = Graph()

    @staticmethod
    def _initialize_graph():
        tg = Graph()
        tg.bind("tb", "http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/tbox#")
        tg.bind("ab", "http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/abox#")
        print("CODE METADAT PATH,", os.path.join(base_dir, 'code_metadata.json'))
        tg = add_jsonld_instances(tg, os.path.join(base_dir, 'code_metadata.json'))
        return tg

    def find_implementation(cls, operation, type):


        for s, p, o in cls.tg.triples((None, tbox.forOp, operation)):
            for s, p, o in cls.tg.triples((s, tbox.forType, type)):
                return s


        qres = cls.tg.query(
            """
            SELECT ?impl
            WHERE {
                ?impl tb:forType ?type .
                ?impl tb:forOp ?operation .
            }
            LIMIT 1
            """,
            initBindings={'type': type, 'operation': operation}
        )
        for row in qres:
            return row[0]
        return None

    def _add_triples_recursively(cls, subject):
        for s, p, o in cls.tg.triples((subject, None, None)):
            cls.implementation.add((s, p, o))
            if isinstance(o, rdflib.URIRef):
                cls._add_triples_recursively(o)

    def _get_implementation_subgraph(cls, imp):
        cls.translation = imp
        cls._add_triples_recursively(imp)

    def annotate_execution_times_and_results(cls):
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                EX = Namespace("http://example.org/ns#")
                execution_time = end_time - start_time
                execution_node = URIRef(f"http://example.org/execution/{cls.translation.split('#')[-1]}")

                cls.ex.add((execution_node, RDF.type, EX.Execution))
                cls.ex.add((execution_node, EX.executionTime, Literal(execution_time, datatype=XSD.decimal)))
                cls.ex.add((execution_node, EX.intermediateResult, Literal(result)))
                return result

            return wrapper

        return decorator

    def _translate_and_compile(cls, operation, template_str, context):
        translation = cls.tg.value(subject=operation, predicate=tbox.hasTranslation)
        cls.translation = translation
        try:
            template = Template(template_str)
            rendered_code = template.render(context)
            fixed_code = autopep8.fix_code(rendered_code)
            print(fixed_code)
        except Exception as e:
            print("Code Generation Error: ", e)
            return None

        try:
            compiled_code = compile(fixed_code, '', 'exec')
            new_func = types.FunctionType(compiled_code.co_consts[0], globals(), translation)
            # new_func = cls.annotate_execution_times_and_results()(new_func)  # Apply the decorator
        except Exception as e:
            print("Code Compilation Error: ", e)
            return None

        return new_func
    # additional methods here


# %%
class Operation(metaclass=Implementation):


    def __init__(self, operation, format="Tabular", inputs={}):
        self.tg = self.__class__.tg
        self.format = format
        self.implementation = Graph()
        self.translation = None
        self.f = self._init_func(operation)

    def _init_func(self, operation):
        translation = self.__class__.find_implementation(operation, self.format)
        if translation:
            self.__class__._get_implementation_subgraph(translation)

        self.translation = translation


        print("Translation", translation)

        code = self.tg.value(subject=translation, predicate=tbox.hasCode)
        params = self.tg.objects(subject=translation, predicate=tbox.hasParameters)
        dependencies = self.tg.objects(subject=translation, predicate=tbox.dependsOn)
        params = [self.tg.value(subject=p, predicate=tbox.name) for p in params]
        dependencies = [self.tg.value(subject=d, predicate=tbox.name) for d in dependencies]

        c_path = self.tg.value(subject=code, predicate=tbox.code)

        codelines = str(c_path).split('\n')

        context = {
            "name": translation.split("#")[1],
            "codelines": codelines,
            "params": params,
            "libraries": dependencies
        }

        template = """
        def {{ name }}({{",".join(params)}}, *args, **kwargs):
        {% for library in libraries %}
            import {{ library }}
        {% endfor %}

        {% for line in codelines %}
            data = {{  line }}
        {% endfor %}
            return data
        """
        return self.__class__._translate_and_compile(operation, template, context)

    def get_func(self):
        return self.f

    def get_implementaiton(self):
        return self.translation, self.__class__.implementation


# %%
def Operation_Constraint(data, lo=None, op=None, ro=None):

    import pandas as pd

    if str(op) == "odrl:gteq":
        return data >= ro

    if str(op) == "odrl:isA":
        if str(ro) == "xsd:string":
            return pd.api.types.is_string_dtype(data[lo])


# %%
class PCTranslator:

    def __init__(self, pc, graph, format="Tabular"):
        # Graph
        self.g = graph
        self.pc = abox[pc]
        self.dp = self.g.value(subject=self.pc, predicate=tbox.validates)
        self.format = self.g.value(subject=self.dp, predicate=tbox.hasDTT)

    def _validate_graph(self) -> bool:
        from pyshacl import validate
        shapes = Graph().parse(os.path.join(base_dir, 'pc_grammar.ttl'), format="turtle")
        conforms, report_graph, report_text = validate(self.g, shacl_graph=shapes)
        return conforms

    def _create_function_as_decorator(self, before_func=None, after_func=None, before_args={},
                                      after_args={}) -> types.FunctionType:
        def decorator(f):
            def wrapper(*args, **kwargs):
                if before_func is not None:
                    before_func(*before_args)
                data = f(*args, **kwargs)
                if after_func is not None:
                    result = after_func(data=data, **after_args)
                    return result
                return data

            return wrapper

        return decorator

    def _create_report_decorator(self, pc) -> types.FunctionType:
        def decorator(f):
            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                data = f(*args, **kwargs)
                self.g.add((pc, tbox.result, Literal(data)))
                self.g.serialize(
                    os.path.join(base_dir, "../../../FederatedComputationalGovernance/SemanticDataModel/sdm.ttl"),
                    format="turtle")
                return data

            return wrapper

        return decorator

    def _add_implementation_to_graph(self, imp, operation):
        implementation, subimp = imp.get_implementaiton()
        self.g += subimp
        self.g.add((operation, tbox.hasTranslation, implementation))

        return implementation

    def _handle_inputs(self, operation, abstract_op):

        """
        Handle the inputs of the operation
        :param operation:
        :return:
        """

        inputs = self.g.objects(subject=operation, predicate=tbox.hasInput)
        inputs_dict = {}

        for input in inputs:
            print("input", input, "type", type(input))
            if abstract_op == odrl["Constraint"]:
                if isinstance(input, Literal) and input.datatype in [XSD.float, XSD.double, XSD.decimal, XSD.integer,
                                                                     XSD.int]:
                    inputs_dict["ro"] = float(input)
                if "odrl" in input:
                    inputs_dict["op"] = input
                if "string" in str(input):
                    inputs_dict["ro"] = input
                if abox in input and input != abox["data"]:
                    inputs_dict["lo"] = input.split("#")[-1]
            else:
                if input != abox["data"]:
                    inputs_dict["attr"] = input.split("#")[-1]

        return inputs_dict

    def _operation_to_code(self, operation) -> types.FunctionType:

        # Operation type
        abstract_op = self.g.value(subject=operation, predicate=tbox.hasAbstract)
        kwargs = self._handle_inputs(operation, abstract_op)

        print("AbstractOP", abstract_op)
        print("kwargs", kwargs)

        imp = Operation(abstract_op, self.format)
        self._add_implementation_to_graph(imp, operation)
        # Case for initial Operation
        if abstract_op == odrl["Constraint"]:

            decorated_imp = self._create_function_as_decorator(after_func=Operation_Constraint, after_args=kwargs)
            return decorated_imp
        elif "attr" in kwargs.keys() and abstract_op != abox["LoadData"]:
            decorated_imp = self._create_function_as_decorator(after_func=imp.get_func(), after_args=kwargs)
            return decorated_imp

        return imp.get_func()

    def traverse_and_generate(self) -> types.FunctionType:
        """
        Traverse the policy checker and generate the code
        :param language:
        :return:
        """

        # Get first operation
        operation = self.g.value(subject=self.pc, predicate=tbox.nextStep)
        function = None

        # Traverse the policy checker (path)
        while operation:
            operation_type = get_urii(str(self.g.value(subject=operation, predicate=RDF.type)))
            if operation_type == "Operation":
                if function is None:
                    function = self._operation_to_code(operation)
                else:
                    decorator = self._operation_to_code(operation)
                    function = decorator(function)
            elif operation_type == "Report":
                decorator = self._create_report_decorator(operation)
                function = decorator(function)

            operation = self.g.value(subject=operation, predicate=tbox.nextStep)

        return function

    def translate(self) -> types.FunctionType:
        """
        Get the policies associated with a data product
        :return: list of policies
        """

        # if self._validate_graph() == True:
        # get policies
        udf = self.traverse_and_generate()
        return udf
        # else:
        #    raise Exception("Policy Checker is not compliant with the grammar")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate Policy Checkers to UDs")
    parser.add_argument("pc", type=str, help="Policy Checker URI")
    parser.add_argument("--execute", action="store_true", help="Execute the resulting UDF")
    parser.add_argument("--plot", action="store_true", help="Flag to plot the times for each policy")
    args = parser.parse_args()

    print(f'Translating Policy Checker {args.pc} to UDF')
    sdm = Graph().parse(os.path.join(base_dir, "../../../FederatedComputationalGovernance/SemanticDataModel/sdm.ttl"),
                        format="turtle")

    function = PCTranslator(args.pc, sdm).translate()

    if args.execute:
        initOP = sdm.value(subject=abox[args.pc], predicate=tbox.nextStep)
        path = sdm.value(subject=initOP, predicate=tbox.hasInput)
        function(path)

    if args.plot:
        PCTranslator(args.pc, sdm).plot_execution_times()

    print("Done")