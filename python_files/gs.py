import math
import copy
import torch
import numpy as np
from torch import nn
import networkx as nx
import random as rand

""" Aggregator functions. """
class Aggregator:
    MEAN = 1
    MEANPOOL = 2
    MAXPOOL = 3
    LSTM = 4

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

# Vectorized sigmoid
sigmoid_np = np.vectorize(sigmoid)

""" TODO: Implement aggregator functions for MEANPOOL, MAXPOOL, and LSTM. """

class GraphSAGE:
    """ In: Graph, features, depth, aggregator. """
    def __init__(self, G, X, K, A, embed_dim):
        rand.seed()
        self.G, self.X, self.K, self.A, self.embed_dim = G, X, K, A, embed_dim

        # Set features to ones if not available
        if X is None:
            self.X = np.ones((len(list(G)), 4))
        self.num_features = np.shape(self.X)[1]
        self.num_nodes = len(list(G))
        self.H = np.zeros((self.k, self.num_nodes)) # Representations at different layers

        # Hyperparameters
        self.learning_rate = 5e-3
        self.batch_size = 32
        self.walksPerNode = 4
        self.walkLength = 3
        self.S = [10, 10] # Fixed-size neighbors per layer

        if self.A == Aggregator.MEAN:
            """ Single layer neural nets for each layer. Each node will in turn
            have different computational graphs where information about neighbors
            is aggregated and passed on to their respective parent, by K hops. """
            self.models_self, self.optimizers_self = [], []
            in_dims = [self.num_features] + [embed_dim] * (k - 1)
            for k in range(K):
                self.models_self.append(nn.Sequential(
                    nn.Linear(in_dims[k], embed_dim),
                    nn.Relu() # Choose ReLU as non-linearity
                    ))
                # Use Adam optimizer as in paper
                self.optimizers_self.append(torch.optim.Adam(self.models_self[k].parameters(), lr=self.learning_rate))

    """ Get fixed-size unifrom sample of neighbors. """
    def N(self, v, k):
        nbrs = list(G.neighbors(v))
        return rand.choices(nbrs, k)

    def randomWalk(self, v, k):
        nodes = [v]
        for i in range(k-1):
            nbrs = list(G.neighbors(v))
            v = rand.choice(nbrs)
            nodes.append(v)
        return nodes

    """ Unsupervised loss function based on negative sampling. """
    def loss_fn(self, output, v, repr):
        pos_sum, neg_sum = 0.0, 0.0
        for ind, node in enumerate(v):
            # Positive samples
            visited = []
            for w in range(self.walksPerNode):
                rw = self.randomWalk(node, self.walkLength)
                visited.extend(rw)
            dotp = repr[visited] @ output[ind].T
            logsig = np.log(sigmoid_np(dotp))
            pos_sum += np.sum(logsig)


            """ TODO: Fix representations. """ 
            # Negative samples
            neg = rand.choices(list(G), len(visited))
            neg_dotp = -(repr[neg] @ output[ind].T)
            neg_logsig = np.log(sigmoid_np(neg_dotp))
            neg_sum += np.sum(neg_logsig)

        return -(pos_sum + neg_sum) / len(v)

    def feedForward(self, layer, X, v, repr, train=True):
        if not train:
            with torch.no_grad():
                pred = self.models_self[layer](X)
            return pred

        Y = self.models_self[layer](X)
        loss = self.loss_fn(Y, v, repr)

        # Backprop
        self.optimizers_self[layer].zero_grad()
        loss.backward()
        self.optimizers_self[layer].step()

        return pred

    def train(self):
        """ Pre-compute neighborhood function. """
        self.nbrs = []
        for layer in range(self.K):
            self.nbrs.append([])
            for v in list(G):
                self.nbrs[layer].append(self.N(v, self.S[layer]))

            # Shuffle and run minibatches
            perm = np.random.permutation(self.num_nodes)
            batches = np.array_split(np.array(list(G))[perm], self.batch_size)
            for b in batches:
                runMinibatch(b, layer+1)


    """ IN: (1) Batch of nodes to do SGD on, (2) At which depth to train weight matrix. """
    def runMinibatch(self, batch, depth, training=True):
        comp_graph = [b] # Computation graph

        """ First get neighbors of nodes in batch, then neighbors of neighbors and so on.
        Two-hop neighbors thus become 100 nodes if each node sample 10 neighbors. """ 
        for layer in range(depth):
            prev = np.ravel(comp_graph[layer])
            dime = (np.size(prev), self.S[-1 - layer])
            next_layer = np.array(dime, dtype=int)
            for n in range(dime[0]):
                next_layer[n] = np.array(self.nbrs[-1 - layer][prev[n]], dtype=int)
            comp_graph.append(next_layer)

        
        # Initialize representations with features
        h_prev = copy.deepcopy(self.X)
        for k in range(depth):
            h_next = np.zeros(np.size(h_prev)).reshape(h_prev.shape)
            nodes = np.ravel(comp_graph[-2 - k]).tolist()
            N_v = comp_graph[-1 - k].tolist() # Neighbors of 'nodes'

            if self.A == Aggregator.MEAN:
                """ Get element-wise mean of current node and neighbors'
                 representations and feed into neural net. """
                dim_size = self.num_features if k == 0 else self.embed_dim
                h_u_v = np.zeros((len(nodes), dim_size))
                for ind, v in enumerate(nodes):
                    _nbrs = h_prev[N_v[ind]]
                    h_u_v[ind] = np.mean(np.vstack((h_prev[v], _nbrs), axis=0))
                h_next[nodes] = self.feedForward(k, h_u_v, nodes, h_prev, ((k+1)==depth and training))

            """ Apply clip-by-norm of gradient instead?
            # Normalize
            l2norms = np.linalg.norm(h_next, axis=1, keepdims=True)
            h_next /= l2norms
            h_prev = h_next"""

        return h_next[nodes]