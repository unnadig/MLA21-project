import numpy as np
import networkx as nx
import copy
import random as rand
import torch

""" Aggregator functions. """
class Aggregator:
    MEAN = 1
    MEANPOOL = 2
    MAXPOOL = 3
    LSTM = 4

""" TODO: Implement aggregator functions for MEANPOOL, MAXPOOL, and LSTM. """

class GraphSAGE:
    """ In: Graph, features, depth, weights, aggregator, fixed-size neighbors per sweep. """
    def __init__(self, G, X, K, W, A, S):
        rand.seed()
        self.G, self.X, self.K, self.A, self.S = G, X, K, A, S
        self.num_features = np.shape(X)[1]

        if self.A == Aggregator.MEAN:
            self.W_self = W

            # Single layer neural net
            self.model_self = nn.Sequential(
                nn.Linear(self.num_features, self.num_features),
                #nn.Relu()
                nn.Sigmoid()
                )

        # Adam optimizer used in paper
        self.optimizer_self = torch.optim.Adam(self.model_self.parameters(), lr=1e-2)

        # TODO: Implement loss function used in paper
        self.loss_fn = nn.CrossEntropyLoss()

    """ Get fixed-size unifrom sample of neighbors. """
    def N(self, v, k):
        nbrs = list(G.neighbors(v))
        #return rand.sample(nbrs, min(k, len(nbrs)))
        return rand.choices(nbrs, k)

    """ TODO: Add batches to speed up learning. """
    def forward(self, X):
        pred = self.model_self(X)

        # ======= Prelim =========
        loss = self.loss_fn(pred)
        # ========================

        # Backprop
        self.optimizer_self.zero_grad()
        loss.backward()
        self.optimizer_self.step()

        return pred


    def train(self):
        h_prev = self.X
        V = list(G)
        for k in range(K):
            h_next = np.zeros(np.size(h_prev)).reshape(h_prev.shape)
            for v in V:
                # Get neighbors
                N_v = self.N(v, self.S[k])

                if self.A == Aggregator.MEAN:
                    """ Get element-wise mean of current node and neighbors'
                     representations and feed into neural net. """
                    h_u_v = np.vstack((h_prev[v], h_prev[N_v]))
                    mean = np.mean(h_u_v, axis=0)
                    h_next[v] = self.forward(mean)

            # Normalize
            l2norms = np.linalg.norm(h_next, axis=1, keepdims=True)
            h_next /= l2norms
            h_prev = h_next
        return h_next












































#asd