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
    visited.append(node)
    for k in range(t-1):
        node = random.choice(list(G.neighbors(node)))
        visited.append(node)
    
    return visited

def skipGram(phi, W, w):
    print('Walk:', W)
    for i in range(len(W) - 2*w):
        v = W[i+w]
        u = W[i:i+2*w+1]
        print('v:', v, 'Window:', u)
        # calculate loss from v and u, then update phi
        phi -= 0

    return phi

def deepWalk(G, w, d, gamma, t):
    V = list(G)
    phi = np.random.rand(len(V), d)
    for i in range(gamma):
        random.shuffle(V)
        for v in V:
            W = randomWalk(G, v, t)
            phi = skipGram(phi, W, w)

    return phi

deepWalk(G, 3, 2, 1, 10)