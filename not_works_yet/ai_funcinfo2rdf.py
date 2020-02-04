from rdflib import Graph, BNode, Namespace, URIRef

g = Graph()

# todo: create definition of objects:

prog = URIRef("http://example.org/people/Bob")
prog_input = BNode()
prog_output = BNode()
csvfile = BNode()
csvfilepath = BNode()
n_csvfile = Namespace("http://example.org/file/csv")

g.add((prog, prog_input, csvfilepath))
g.add((prog, prog_output, n_csvfile.header))
