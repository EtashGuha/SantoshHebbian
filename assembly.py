import numpy as np
rng = np.random.default_rng()
import matplotlib.pyplot as plt
import networkx as nx
import random
import scipy
from heuristics import min_degree, greedy, extended_greedy
import itertools
import tqdm as tqdm
import time 

n_inputs = 500
n_neurons = 1000
cap_size = 30
n_rounds = 100
n_nets = 40
sparsity = 0.05

plt.rcParams.update({
    "figure.facecolor":  (1.0, 1.0, 1.0, 1.0),  # red   with alpha = 30%
    "axes.facecolor":    (1.0, 1.0, 1.0, 1.0),  # green with alpha = 50%
    "savefig.facecolor": (1.0, 1.0, 1.0, 1.0),  # blue  with alpha = 20%
})

def entropy4(labels, base=None):
    value,counts = np.unique(labels, return_counts=True)
    norm_counts = counts / counts.sum()
    base = np.e if base is None else base
    return -(norm_counts * np.log(norm_counts)/np.log(base)).sum()
    

def k_cap(input, cap_size):
    output = np.zeros_like(input)
    if len(input.shape) == 1:
        idx = np.argsort(input)[-cap_size:]
        output[idx] = 1
    else:
        idx = np.argsort(input, axis=-1)[..., -cap_size:]
        np.put_along_axis(output, idx, 1, axis=-1)
    return output

class BrainNet():
    def __init__(self, n_inputs, n_neurons, cap_size, n_rounds, sparsity, homeostasis, tau = .1, plasticity = 0):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.cap_size = cap_size
        self.tau = tau
        self.n_rounds = n_rounds
        self.sparsity = sparsity
        self.generate_graph()
        self.reset_weights()
        # self.plasticity_params = np.zeros((2, 6))
        self.plasticity = plasticity
        self.debug_rec_weights = np.ones((n_rounds, n_neurons, n_neurons))
        self.homeostasis = homeostasis

    def generate_graph(self):
        self.input_adj = rng.random((n_inputs, self.n_neurons)) < self.sparsity
        self.rec_adj = np.logical_and(rng.random((self.n_neurons, self.n_neurons)) < self.sparsity, 
                                      np.logical_not(np.eye(self.n_neurons, dtype=bool)))
        
    def reset_weights(self):
        self.rec_weights = np.ones((self.n_neurons, self.n_neurons))
        
    def forward(self, update=False):
        self.activations = np.zeros((self.n_rounds, self.n_neurons))


        self.activations[0, :self.cap_size] = 1

        for i in range(self.n_rounds-1):
            self.tau = self.tau/(1 + 5 *self.tau)
            self.activations[i+1] = k_cap(self.activations[i] @ (self.rec_adj * self.rec_weights), self.cap_size)
            if update: 
                self.update_weights(i)
        return self.activations
    
    def update_weights(self, step):
        self.debug_rec_weights[step] = self.rec_weights
        flattened_weights = self.rec_weights.flatten()/np.sum(self.rec_weights)
        entropy_grad = -1 * (entropy4(flattened_weights) + np.log(flattened_weights))/(1 - flattened_weights)
        entropy_grad = entropy_grad.reshape((n_neurons, n_neurons))
        entropy_grad = entropy_grad - np.min(entropy_grad)
        
        plasticity_rule = self.activations[step-1,:][:,np.newaxis] * self.activations[step, :][np.newaxis, :] * self.rec_adj - self.tau * entropy_grad
        # print("max p: {} min p: {} max e: {} min e: {}".format(np.max(plasticity_rule), np.min(plasticity_rule), np.max(entropy_grad), np.min(entropy_grad)))
        self.rec_weights += plasticity_rule * self.plasticity 
        if self.homeostasis:
            self.rec_weights = self.rec_weights / (self.rec_weights.sum(axis=1)[:, np.newaxis])

