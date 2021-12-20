import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import os

# read data from file
dirname = os.path.realpath('..')
filename = os.path.join(dirname, 'data/ucidata-zachary/out.ucidata-zachary')
rows = []
with open(filename) as file:
    for line in file:
        rows.append(line.rstrip())
del rows[0:2]

for i in range(len(rows)):
    split_row = rows[i].split()
    rows[i] = (int(split_row[0]), int(split_row[1]))

# construct graph
G = nx.Graph()
G.add_edges_from(rows)

def randomWalk(G, v, t):
    visited = []
    node = v
    for k in range(t):
        node = random.choice(list(G.neighbors(node)))
        visited.append(node)
    
    return visited

def randomWalk_adj(G, v, t):
    A = nx.adjacency_matrix(G)
    A = A.todense()
    A = np.array(A, dtype=np.float64)
    T = A/A.sum(axis=1, keepdims=True)

    p = np.zeros(A.shape[0]).reshape(-1,1)
    p[v] = 1
    
    visited = []
    for k in range(t):
        p = np.dot(T,p)
        visited.append(np.argmax(p))

    return visited

def skipGram(phi, W, w):

    return None

def deepWalk(G, w, d, gamma, t):
    # initialization
    V = 0
    # build binary tree
    for i in range(gamma):
        V_shuffle = 0
        for v in v_shuffle:
            W = randomWalk(G, v, t)
            skipGram(phi, W, w)
    return 0