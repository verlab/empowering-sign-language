import os
import sys

class GraphBuilder(object):

    def __init__(self, cfg = None):
        self.cfg = cfg

    def _create_nose_nodes(self, nodes):
        b_node_upn, e_node_upn = (27, 30)
        b_node_dwn, e_node_dwn = (31, 35)

        for v in range(b_node_upn, e_node_upn):
            edge = (v, v + 1)
            nodes.append(edge)

        for v in range(b_node_dwn, e_node_dwn):
            edge = (v, v + 1)
            nodes.append(edge)

        return nodes

    def _create_eyebrow_nodes(self, nodes, side = "right"):
        b_node, e_node = (22, 26) if side == "right" else (17, 21)
        for v in range(b_node, e_node):
            edge = (v, v + 1)
            nodes.append(edge)
        return nodes
        
    def _create_mouth_nodes(self, nodes, outer = True):
        b_node, e_node = (48, 59) if outer else (61, 66) 
        nodes.append((b_node, e_node))
        for v in range(b_node, e_node):
            edge = (v, v + 1)
            nodes.append(edge)
        return nodes

    def _create_eye_nodes(self, nodes, side):
        b_node, e_node = (36, 41) if side == "right" else (42, 47)
        nodes.append((b_node, e_node))
        for v in range(b_node, e_node):
            edge = (v, v + 1)
            nodes.append(edge)
        return nodes

    def _create_jaw_nodes(self, nodes):
        for v in range(0, 16):
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

        return nodes

    def get_nodes_n(self, nodes):
        node_set = set()
        for e in nodes:
            node_set.add(e[0])
            node_set.add(e[1])
        return node_set


def main():

    builder = GraphBuilder()
    nodes = builder.create_face_graph()
    print(builder.get_nodes_n(nodes))
    
if __name__ == "__main__":
    main()