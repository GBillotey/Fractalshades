# -*- coding: utf-8 -*-



class DAgraph():
    def __init__(self):
        self._nodes = {}
        self._vertex= {}
        
    def add_node(self, kind, x, y, **props):
        new_key = list(dict.keys())[-1] +1
        self._nodes[new_key] = Node(self, new_key, kind, x, y, **props)
        
    def add_vertex(self, , , y, **props):
        new_key = list(dict.keys())[-1] +1
        self._nodes[new_key] = Node(self, new_key, kind, x, y, **props)
#    def to_DAG(self):
#        """ Returns a representation of the graph as a Direct Acyclic Graph
#        by ordering the class method nodes 
#        OR SHOULD we enforce to be a DAG (seems better)
#        """
        
    def topological_nodesort(self):
        """ Return the topological-ordered list of nodes """


class Vertex():
    def __init__(self):
        self._start = None
        self._end = None
    
    

    
class Node():
    def __init__(self, nodegraph, key, kind, x, y, **props):
        self.node_x = x
        self.node_y = y
        self.node_kind = kind
    
    def moved(self, x, y):
        self.node_x = x
        self.node_y = y
        # Special reordering the DAG if is a method
        if self.node_kind == "METHOD":
            print("reordering meth")
    
    def parents(self):
        pass
    
    def childs(self):
        pass
        
        
