import os
import sys

class GraphBuilder(object):

    def __init__(self, cfg = None):
        self.cfg = cfg

    def _create_nose_nodes(self, nodes):
        b_node_upn, e_node_upn = (25, 28)
        b_node_dwn, e_node_dwn = (29, 33)

        for v in range(b_node_upn, e_node_upn):
            edge = (v, v + 1)
            nodes.append(edge)

        for v in range(b_node_dwn, e_node_dwn):
            edge = (v, v + 1)
            nodes.append(edge)

        return nodes

    def _create_eyebrow_nodes(self, nodes, side = "right"):
        b_node, e_node = (20, 24) if side == "right" else (15, 19)
        for v in range(b_node, e_node):
            edge = (v, v + 1)
            nodes.append(edge)
        return nodes
        
    def _create_mouth_nodes(self, nodes, outer = True):
        b_node, e_node = (46, 57) if outer else (58, 63) 
        nodes.append((b_node, e_node))
        for v in range(b_node, e_node):
            edge = (v, v + 1)
            nodes.append(edge)
        return nodes

    def _create_eye_nodes(self, nodes, side):
        b_node, e_node = (40, 45) if side == "right" else (34, 39)
        nodes.append((b_node, e_node))
        for v in range(b_node, e_node):
            edge = (v, v + 1)
            nodes.append(edge)
        return nodes

    def _create_jaw_nodes(self, nodes):
        for v in range(0, 14):
            edge = (v, v + 1)
            nodes.append(edge)
        return nodes

    def create_face_graph(self):
        nodes = list()
        nodes = self._create_jaw_nodes(nodes)
        nodes = self._create_eye_nodes(nodes, side = "right")
        nodes = self._create_eye_nodes(nodes, side = "left")
        nodes = self._create_mouth_nodes(nodes)
        nodes = self._create_mouth_nodes(nodes, outer = False)
        nodes = self._create_eyebrow_nodes(nodes, side = "right")
        nodes = self._create_eyebrow_nodes(nodes, side = "left")
        nodes = self._create_nose_nodes(nodes)
        nodes = self._add_expression_edges(nodes)

        return nodes


    def _add_expression_edges(self, edges):
        edges_e = [
            (1, 16), (1, 18), (1, 39), (1, 38), (39, 35), (38, 36), (16, 39), (16, 35), (18, 38), (18, 36), #left eye
            (13, 21), (13, 23), (13, 45), (13, 44), (45, 41), (44, 42), (21, 45), (21, 41), (23, 44), (23, 42), #right eye
            (27, 31), (31, 48), (31, 50), #nose
            (4, 48), (4, 56), (7, 56), (7, 54), (10, 50), (10, 54), #sides and chin
            (59, 62), (48, 59), (50, 59), (56, 62), (54, 62) #outer and inner mouth
        ]
        return edges + edges_e

    def get_nodes_n(self, nodes):
        node_set = set()
        for e in nodes:
            node_set.add(e[0])
            node_set.add(e[1])
        return node_set


def main():

    builder = GraphBuilder()
    nodes = builder.create_face_graph()
    graph = Graph(64, nodes, 59)
    print(builder.get_nodes_n(nodes))
    
if __name__ == "__main__":
    main()