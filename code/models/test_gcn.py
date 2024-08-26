from gcn.stgc import *
from gcn.graph import Graph
from gcn.create_graph import GraphBuilder
import torch
import torch.nn as nn
import numpy as np

def main():

    cols1 = [1, 3, 5, 7, 10, 11, 14, 15, 20, 22, 25, 28, 31, 36, 40, 42]
    cols2 = [9, 10, 11, 12, 13, 14, 15]
    cols3 = [6]

    edges = np.load("edges.npy")
    edges = [tuple(e) for e in edges]
    edges += [(42, 36), (42, 40), (42, 37), (42, 35), (42, 20), (42, 34), (42, 38), (42, 41), (42, 39), (12, 13), (25, 28), (10, 42), (11, 42), (22, 42), (25, 42), (28, 42), (31, 42), (14, 42)]

    graph64 = Graph(43, edges, 42, max_hop = 1, k = 2)
    ca64 = torch.tensor(graph64.A, dtype=torch.float32, requires_grad=False)
    a64 = torch.tensor(graph64.getA(cols1), dtype=torch.float32, requires_grad=False)
    _, l1 = graph64.getLowAjd(cols1)

    graph16 = Graph(16, l1, 15, max_hop = 1, k = 2)
    ca16 = torch.tensor(graph16.A, dtype=torch.float32, requires_grad=False)
    a16 = torch.tensor(graph16.getA(cols2), dtype=torch.float32, requires_grad=False)
    _, l2 = graph16.getLowAjd(cols2)

    graph7 = Graph(7, l2, 6, max_hop = 1, k = 2)
    ca7 = torch.tensor(graph7.A, dtype=torch.float32, requires_grad=False)
    a7 = torch.tensor(graph7.getA(cols3), dtype=torch.float32, requires_grad=False)
    _, l3 = graph7.getLowAjd(cols3)


    graph1 = Graph(1, l3, 0)
    ca1 = torch.tensor(graph1.A, dtype=torch.float32, requires_grad=False)



    import pdb
    pdb.set_trace()

if __name__ == "__main__":
    main()
