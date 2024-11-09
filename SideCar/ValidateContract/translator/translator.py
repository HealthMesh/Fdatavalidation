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
    """
    Adds JSON-LD instances to the graph.

    Args:
        graph (rdflib.Graph): The RDF graph to which instances will be added.
        path (str): The path to the JSON-LD file.

    Returns:
        rdflib.Graph: The updated RDF graph.
    """
    with open(path, 'r') as f:
        json_ld_data = json.loads(f.read())
        instances = Graph().parse(data=json_ld_data, format='json-ld')
        graph += instances
    return graph


# URI functions
def get_urii(uri):
    """
    Extracts the fragment part of a URI.

    Args:
        uri (str): The URI string.

    Returns:
        str: The fragment part of the URI.
    """
    return uri.split("#")[-1]


class Implementation(type):
    """
    Metaclass for initializing and managing RDF graphs for implementations.
    """

    def __init__(cls, name, bases, attrs):
        """
        Initializes the class with a translation graph and an execution graph.

        Args:
            name (str): The name of the class.
            bases (tuple): The base classes.
            attrs (dict): The class attributes.
        """
        super().__init__(name, bases, attrs)
        cls.tg = cls._initialize_graph()
        cls.ex = Graph()
        cls.translation = None
        cls.implementation = Graph()

    @staticmethod
    def _initialize_graph():
        """
        Initializes the translation graph with predefined namespaces and JSON-LD instances.

        Returns:
            rdflib.Graph: The initialized translation graph.
        """
        tg = Graph()
        tg.bind("tb", "http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/tbox#")
        tg.bind("ab", "http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/abox#")
        tg = add_jsonld_instances(tg, os.path.join(base_dir, 'code_metadata.json'))
        return tg

    def find_implementation(cls, operation, type):
        """
        Finds the implementation for a given operation and type.

        Args:
            operation (rdflib.URIRef): The operation URI.
            type (rdflib.URIRef): The type URI.

        Returns:
            rdflib.URIRef: The implementation URI, if found.
        """
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
        """
        Recursively adds triples to the implementation graph.

        Args:
            subject (rdflib.URIRef): The subject URI to start adding triples from.
        """
        for s, p, o in cls.tg.triples((subject, None, None)):
            cls.implementation.add((s, p, o))
            if isinstance(o, rdflib.URIRef):
                cls._add_triples_recursively(o)

    def _get_implementation_subgraph(cls, imp):
        """
        Retrieves the subgraph for a given implementation.

        Args:
            imp (rdflib.URIRef): The implementation URI.
        """
        cls.translation = imp
        cls._add_triples_recursively(imp)



    def _translate_and_compile(cls, operation, template_str, context):
        """
        Translates and compiles a template string into a Python function.

        Args:
            operation (rdflib.URIRef): The operation URI.
            template_str (str): The template string.
            context (dict): The context for rendering the template.

        Returns:
            function: The compiled Python function, or None if an error occurs.
        """
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
        except Exception as e:
            print("Code Compilation Error: ", e)
            return None

        return new_func


class Operator(metaclass=Implementation):
    """
    Class representing an operator with a specific implementation.
    """

    def __init__(self, operation, format="Tabular"):
        """
        Initializes the operator with the given operation and format.

        Args:
            operation (rdflib.URIRef): The operation URI.
            format (str): The format of the operator (default is "Tabular").
        """
        self.tg = self.__class__.tg
        self.format = format
        self.implementation = Graph()
        self.translation = None
        self.f = self._init_func(operation)

    def _init_func(self, operation):
        """
        Initializes the function associated with the operation.

        Args:
            operation (rdflib.URIRef): The operation URI.

        Returns:
            function: The initialized function.
        """
        translation = self.__class__.find_implementation(operation, self.format)
        if translation:
            self.__class__._get_implementation_subgraph(translation)

        self.translation = translation

        code = self.tg.value(subject=translation, predicate=tbox.hasCode)
        params = self.tg.objects(subject=translation, predicate=tbox.hasParameters)
        dependencies = self.tg.objects(subject=translation, predicate=tbox.dependsOn)
        params = [self.tg.value(subject=p, predicate=tbox.name) for p in params]
        dependencies = [self.tg.value(subject=d, predicate=tbox.name) for d in dependencies]

        c_path = self.tg.value(subject=code, predicate=tbox.code)
        codelines = str(c_path).split('\n')

        context = {
            "name": translation.split("#")[1],
            "codelines": codelines[0],
            "params": params,
            "libraries": dependencies
        }

        template = """
         def {{ name }}({{",".join(params)}}, *args, **kwargs):
             data = {{codelines}}
             return data
         """
        return self.__class__._translate_and_compile(operation, template, context)

    def get_func(self):
        """
        Returns the function associated with the operator.

        Returns:
            function: The function associated with the operator.
        """
        return self.f

    def get_implementaiton(self):
        """
        Returns the implementation and its subgraph.

        Returns:
            tuple: The implementation URI and its subgraph.
        """
        return self.translation, self.__class__.implementation


class Operation(metaclass=Implementation):
    """
    Class representing an operation with a specific implementation.
    """

    def __init__(self, operation, format="Tabular"):
        """
        Initializes the operation with the given operation and format.

        Args:
            operation (rdflib.URIRef): The operation URI.
            format (str): The format of the operation (default is "Tabular").
        """
        self.tg = self.__class__.tg
        self.format = format
        self.implementation = Graph()
        self.translation = None
        self.f = self._init_func(operation)

    def _init_func(self, operation):
        """
        Initializes the function associated with the operation.

        Args:
            operation (rdflib.URIRef): The operation URI.

        Returns:
            function: The initialized function.
        """
        translation = self.__class__.find_implementation(operation, self.format)
        if translation:
            self.__class__._get_implementation_subgraph(translation)

        self.translation = translation

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
        """
        Returns the function associated with the operation.

        Returns:
            function: The function associated with the operation.
        """
        return self.f

    def get_implementaiton(self):
        """
        Returns the implementation and its subgraph.

        Returns:
            tuple: The implementation URI and its subgraph.
        """
        return self.translation, self.__class__.implementation


class PCTranslator:
    """
    Class for translating PolicyChecker operations to an executable script.
    """

    def __init__(self, pc, graph, format="Tabular"):
        """
        Initializes the PCTranslator with the given PolicyChecker and graph.

        Args:
            pc (str): The PolicyChecker URI.
            graph (rdflib.Graph): The RDF graph.
            format (str): The format of the PolicyChecker (default is "Tabular").
        """
        self.g = graph
        self.pc = abox[pc]
        self.format = self.g.value(subject=self.pc, predicate=tbox.hasType)
        self.execution_times = []
        self._operation_to_code = self._time_execution(self._operation_to_code)
        self.traverse_and_generate = self._time_execution(self.traverse_and_generate)

    def _time_execution(self, func):
        """
        Decorator to measure the execution time of a function.

        Args:
            func (function): The function to be measured.

        Returns:
            function: The wrapped function with execution time measurement.
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            self.execution_times.append((func.__name__, execution_time))
            return result

        return wrapper

    def plot_execution_times(self):
        """
        Plots the execution times for each operation.
        """
        operations = [name for name, _ in self.execution_times]
        times = [time for _, time in self.execution_times]

        plt.figure(figsize=(10, 5))
        plt.bar(operations, times, color='blue')
        plt.xlabel('Operations')
        plt.ylabel('Execution Time (seconds)')
        plt.title('Execution Time for Each Operation')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig("execution_times.png")



    def _validate_graph(self) -> bool:
        """
        Validates the policies grammar is compliant with the defined grammar.

        Returns:
            bool: True if the graph is compliant, False otherwise.
        """
        from pyshacl import validate
        shapes = Graph().parse(os.path.join(base_dir, 'pc_grammar.ttl'), format="turtle")
        conforms, report_graph, report_text = validate(self.g, shacl_graph=shapes)
        return conforms

    def _create_function_as_decorator(self, before_func=None, after_func=None, before_args={},
                                      after_args={}) -> types.FunctionType:
        """
        Creates a decorator function that executes before and after functions.

        Args:
            before_func (function, optional): The function to execute before the main function.
            after_func (function, optional): The function to execute after the main function.
            before_args (dict, optional): The arguments for the before function.
            after_args (dict, optional): The arguments for the after function.

        Returns:
            function: The decorator function.
        """

        def decorator(f):
            @functools.wraps(f)
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
        """
        Creates a decorator function that adds the result to the graph and serializes it.

        Args:
            pc (rdflib.URIRef): The PolicyChecker URI.

        Returns:
            function: The decorator function.
        """

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

    def _get_func_parameters(self, f):
        """
        Retrieves the parameters of a function.

        Args:
            f (function): The function.

        Returns:
            inspect.Signature: The function parameters.
        """
        import inspect
        return inspect.signature(f).parameters

    def _add_implementation_to_graph(self, imp, operation):
        """
        Adds the implementation and its subgraph to the graph.

        Args:
            imp (Implementation): The implementation instance.
            operation (rdflib.URIRef): The operation URI.

        Returns:
            rdflib.URIRef: The implementation URI.
        """
        implementation, subimp = imp.get_implementaiton()
        self.g += subimp
        self.g.add((operation, tbox.hasTranslation, implementation))
        return implementation


    def _operation_to_code(self, operation) -> types.FunctionType:
        """
        Translates an operation to a Python function.

        Args:
            operation (rdflib.URIRef): The operation URI.

        Returns:
            function: The translated Python function.
        """
        operation_type = get_urii(str(self.g.value(subject=operation, predicate=RDF.type)))
        abstract_op = self.g.value(subject=operation, predicate=tbox.hasAbstract)
        kwargs = {}

        if operation_type == "initOperation":
            imp = Operation(abstract_op, self.format)
            self._add_implementation_to_graph(imp, operation)
            return imp.get_func()

        elif operation_type == "Operation":
            imp = Operation(abstract_op, self.format)
            self._add_implementation_to_graph(imp, operation)
            attrs = self.g.objects(subject=operation, predicate=tbox.hasAttribute)
            attrs = [str(a).split("#")[1] for a in attrs]
            kwargs["attr"] = attrs[0]
            decorated_imp = self._create_function_as_decorator(after_func=imp.get_func(), after_args=kwargs)
            return decorated_imp

        elif operation_type == "Operator":
            imp = Operator(abstract_op, self.format)
            self._add_implementation_to_graph(imp, operation)
            kwargs = {
                "lo": self.g.value(subject=operation, predicate=tbox.lo),
                "ro": float(str(self.g.value(subject=operation, predicate=tbox.ro)))
            }
            decorated_imp = self._create_function_as_decorator(after_func=imp.get_func(), after_args=kwargs)
            return decorated_imp


    def traverse_and_generate(self) -> types.FunctionType:
        """
        Traverses the policy checker and generates the code.

        Returns:
            function: The generated function.
        """
        operation = self.g.value(subject=self.pc, predicate=tbox.nextStep)
        function = None

        while operation:
            operation_type = get_urii(str(self.g.value(subject=operation, predicate=RDF.type)))
            if operation_type == "initOperation":
                function = self._operation_to_code(operation)
            elif operation_type == "Operation" or operation_type == "Operator":
                decorator = self._operation_to_code(operation)
                function = decorator(function)
            elif operation_type == "Report":
                decorator = self._create_report_decorator(operation)
                function = decorator(function)

            operation = self.g.value(subject=operation, predicate=tbox.nextStep)

        return function

    def translate(self) -> types.FunctionType:
        """
        Translates the PolicyChecker to a user-defined function (UDF).

        Returns:
            function: The translated UDF.
        """
        udf = self.traverse_and_generate()
        return udf


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
        print("Executing UDF")
        initOP = sdm.value(subject=abox[args.pc], predicate=tbox.nextStep)
        path = sdm.value(subject=initOP, predicate=tbox.hasParameter)
        function(path)

    if args.plot:
        print("Plotting Execution Times")
        PCTranslator(args.pc, sdm).plot_execution_times()

    print("Done")