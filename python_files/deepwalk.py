import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# read data from file
filename = 'data/ucidata-zachary/out.ucidata-zachary'
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

# deepwalk algorithm
