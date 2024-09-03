import os
import json
import time
import types
import autopep8
import functools
import inspect
from jinja2 import Template
from rdflib import Graph, Namespace, URIRef, RDF, Literal
from rdflib.namespace import XSD
from owlrl import OWLRL
from pyshacl import validate


# Set up namespaces
tbox = Namespace('http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/tbox#')
abox = Namespace('http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/abox#')
dcat = Namespace('https://www.w3.org/ns/dcat#')
dcterms = Namespace('http://purl.org/dc/terms/')
tb = Namespace("http://www.semanticweb.org/acraf/ontologies/2021/0/SDM#")
odrl = Namespace("http://www.w3.org/ns/odrl/2/")

# Set base directory
base_dir = os.getcwd()
folder = ''  # Add the specific folder if needed
if folder:
    base_dir = os.path.join(base_dir, folder)

def add_jsonld_instances(graph, path):
    with open(path, 'r') as f:
        json_ld_data = json.loads(f.read())
        instances = Graph().parse(data=json_ld_data, format='json-ld')
        graph += instances
    return graph

class Implementation(type):
    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        cls.tg = cls._initialize_graph()
        cls.ex = Graph()
        cls.translation = None

    @staticmethod
    def _initialize_graph():
        tg = Graph()
        tg.bind("tb", "http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/tbox#")
        tg.bind("ab", "http://www.semanticweb.org/acraf/ontologies/2024/healthmesh/abox#")
        tg = add_jsonld_instances(tg, os.path.join(base_dir, 'translations_mappings.json'))
        return tg

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
        except Exception as e:
            print("Code Compilation Error: ", e)
            return None

        return new_func

class Operator(metaclass=Implementation):
    def __init__(self, operation, format="Tabular"):
        self.tg = self.__class__.tg
        self.format = format
        self.implementation = Graph()
        self.translation = None
        self.f = self._init_func(operation)

    def find_implementation(self, operation, type):
        qres = self.tg.query(
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

    def _get_implementation_subgraph(self, imp):
        self.translation = imp
        for s, p, o in self.tg.triples((imp, None, None)):
            self.implementation.add((s, p, o))
            if isinstance(o, URIRef):
                for s2, p2, o2 in self.tg.triples((o, None, None)):
                    self.implementation.add((s2, p2, o2))
                    if isinstance(o2, URIRef):
                        for s3, p3, o3 in self.tg.triples((o2, None, None)):
                            self.implementation.add((s3, p3, o3))

    def _init_func(self, operation):
        translation = self.find_implementation(operation, self.format)
        if translation:
            self._get_implementation_subgraph(translation)

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
        def {{ name }}({{",".join(params)}}):
        {% for library in libraries %}
            import {{ library }}
        {% endfor %}
        {% for line in codelines %}
            data = {{ line }}
        {% endfor %}
            return data
        """
        return self.__class__._translate_and_compile(operation, template, context)

    def get_func(self):
        return self.f

    def get_implementation(self):
        return self.translation, self.implementation

class Operation(Operator):
    def _init_func(self, operation):
        return super()._init_func(operation)

class PCTranslator:
    def __init__(self, pc, graph, format="Tabular"):
        self.g = graph
        self.pc = abox[pc]
        self.format = self.g.value(subject=self.pc, predicate=tbox.hasType)

    def _validate_graph(self) -> bool:
        shapes = Graph().parse(os.path.join(base_dir, 'pc_grammar.ttl'), format="turtle")
        conforms, report_graph, report_text = validate(self.g, shacl_graph=shapes)
        return conforms

    def _create_function_as_decorator(self, before_func=None, after_func=None, before_args={}, after_args={}) -> types.FunctionType:
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
            def wrapper(*args, **kwargs):
                data = f(*args, **kwargs)
                self.g.add((pc, tbox.result, Literal(data)))
                self.g.serialize(os.path.join(base_dir, "../../../FederatedComputationalGovernance/SemanticDataModel/sdm.ttl"), format="turtle")
                return data
            return wrapper
        return decorator

    def _get_func_parameters(self, f):
        return inspect.signature(f).parameters

    def _operation_to_code(self, operation) -> types.FunctionType:
        operation_type = str(self.g.value(subject=operation, predicate=RDF.type)).split("#")[-1]
        abstract_op = self.g.value(subject=operation, predicate=tbox.hasAbstract)
        kwargs = {}

        if operation_type == "initOperation":
            imp = Operation(abstract_op, self.format)
            implementation, subimp = imp.get_implementation()
            self.g += subimp
            self.g.add((operation, tbox.hasTranslation, implementation))
            return imp.get_func()

        elif operation_type == "Operation":
            imp = Operation(abstract_op, self.format)
            implementation, subimp = imp.get_implementation()
            self.g += subimp
            self.g.add((operation, tbox.hasTranslation, implementation))

            params = self._get_func_parameters(imp.get_func())
            parameters = self.g.objects(subject=operation, predicate=tbox.hasParameter)
            parameters = [str(p).split("#")[1] for p in parameters]
            attrs = self.g.objects(subject=operation, predicate=tbox.hasAttribute)
            attrs = [str(a).split("#")[1] for a in attrs]

            kwargs["attr"] = attrs[0]
            decorated_imp = self._create_function_as_decorator(after_func=imp.get_func(), after_args=kwargs)
            return decorated_imp

        elif operation_type == "Operator":
            imp = Operator(abstract_op, self.format)
            implementation, subimp = imp.get_implementation()
            self.g += subimp
            self.g.add((operation, tbox.hasTranslation, implementation))

            kwargs = {
                "lo": self.g.value(subject=operation, predicate=tbox.lo),
                "ro": float(str(self.g.value(subject=operation, predicate=tbox.ro)))
            }
            decorated_imp = self._create_function_as_decorator(after_func=imp.get_func(), after_args=kwargs)
            return decorated_imp

    def traverse_and_generate(self) -> types.FunctionType:
        operation = self.g.value(subject=self.pc, predicate=tbox.nextStep)
        function = None
        while operation:
            operation_type = str(self.g.value(subject=operation, predicate=RDF.type)).split("#")[-1]
            if operation_type == "initOperation":
                function = self._operation_to_code(operation)
            elif operation_type in ["Operation", "Operator"]:
                decorator = self._operation_to_code(operation)
                function = decorator(function)
            elif operation_type == "Report":
                decorator = self._create_report_decorator(operation)
                function = decorator(function)
            operation = self.g.value(subject=operation, predicate=tbox.nextStep)
        return function

    def translate(self) -> types.FunctionType:
        return self.traverse_and_generate()
