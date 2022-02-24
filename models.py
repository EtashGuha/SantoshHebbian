import numpy as np

import numpy as np
rng = np.random.default_rng()
import matplotlib.pyplot as plt

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
    def __init__(self, n_inputs, n_neurons, cap_size, n_rounds, sparsity):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.cap_size = cap_size
        self.n_rounds = n_rounds
        self.sparsity = sparsity
        self.generate_graph()
        self.reset_weights()
        self.plasticity_params = np.zeros((2, 6))
        
    def generate_graph(self):
        self.input_adj = rng.random((self.n_inputs, self.n_neurons)) < self.sparsity
        self.rec_adj = np.logical_and(rng.random((self.n_neurons, self.n_neurons)) < self.sparsity, 
                                      np.logical_not(np.eye(self.n_neurons, dtype=bool)))
        
    def reset_weights(self):
        self.input_weights = np.ones((self.n_inputs, self.n_neurons))
        self.rec_weights = np.ones((self.n_neurons, self.n_neurons))
        
    def forward(self, inputs, update=False):
        self.activations = np.zeros((inputs.shape[0], self.n_rounds, self.n_neurons))
        self.activations[:, 0] = k_cap(inputs @ (self.input_adj * self.input_weights), self.cap_size)
        for i in range(self.n_rounds-1):
            self.activations[:, i+1] = k_cap(inputs @ (self.input_adj * self.input_weights) + self.activations[:, i] @ (self.rec_adj * self.rec_weights), self.cap_size)
            if update: 
                self.update_weights(i, inputs)
        return self.activations
    
    def update_weights(self, step, inputs):
        input_plasticity_terms = np.zeros((6, self.n_inputs, self.n_neurons))
        rec_plasticity_terms = np.zeros((6, self.n_neurons, self.n_neurons))
        
        input_plasticity_terms[0] = self.input_weights * (inputs[:, :, np.newaxis] * self.activations[:, step+1][:, np.newaxis, :]).mean(axis=0)
        input_plasticity_terms[1] = self.input_weights * ((1 - inputs)[:, :, np.newaxis] * self.activations[:, step+1][:, np.newaxis, :]).mean(axis=0)
        input_plasticity_terms[2] = self.input_weights * ((inputs[:, :, np.newaxis] * (self.input_adj * self.input_weights)[np.newaxis, :, :]).mean(axis=1, keepdims=True) * self.activations[:, step+1][:, np.newaxis, :]).mean(axis=0)
        input_plasticity_terms[3] = self.input_weights * (((1 - inputs)[:, :, np.newaxis] * (self.input_adj * self.input_weights)[np.newaxis, :, :]).mean(axis=1, keepdims=True) * self.activations[:, step+1][:, np.newaxis, :]).mean(axis=0)
        input_plasticity_terms[4] = self.input_weights * ((self.activations[:, step][:, :, np.newaxis] * (self.rec_adj * self.rec_weights)[np.newaxis, :, :]).mean(axis=1, keepdims=True) * self.activations[:, step+1][:, np.newaxis, :]).mean(axis=0)
        input_plasticity_terms[5] = self.input_weights * (((1 - self.activations[:, step])[:, :, np.newaxis] * (self.rec_adj * self.rec_weights)[np.newaxis, :, :]).mean(axis=1, keepdims=True) * self.activations[:, step+1][:, np.newaxis, :]).mean(axis=0)
#         input_plasticity_terms[6] = self.input_weights * ((inputs[:, :, np.newaxis] * (self.input_adj * self.input_weights)[np.newaxis, :, :]).mean(axis=2, keepdims=True) * self.activations[:, step+1][:, np.newaxis, :]).mean(axis=0)
#         input_plasticity_terms[7] = self.input_weights * (((1 - inputs)[:, :, np.newaxis] * (self.input_adj * self.input_weights)[np.newaxis, :, :]).mean(axis=2, keepdims=True) * self.activations[:, step+1][:, np.newaxis, :]).mean(axis=0)

        rec_plasticity_terms[0] = self.rec_weights * (self.activations[:, step][:, :, np.newaxis] * self.activations[:, step+1][:, np.newaxis, :]).mean(axis=0)
        rec_plasticity_terms[1] = self.rec_weights * ((1 - self.activations[:, step])[:, :, np.newaxis] * self.activations[:, step+1][:, np.newaxis, :]).mean(axis=0)
        rec_plasticity_terms[2] = self.rec_weights * ((inputs[:, :, np.newaxis] * (self.input_adj * self.input_weights)[np.newaxis, :, :]).mean(axis=1, keepdims=True) * self.activations[:, step+1][:, np.newaxis, :]).mean(axis=0)
        rec_plasticity_terms[3] = self.rec_weights * (((1 - inputs)[:, :, np.newaxis] * (self.input_adj * self.input_weights)[np.newaxis, :, :]).mean(axis=1, keepdims=True) * self.activations[:, step+1][:, np.newaxis, :]).mean(axis=0)
        rec_plasticity_terms[4] = self.rec_weights * ((self.activations[:, step][:, :, np.newaxis] * (self.rec_adj * self.rec_weights)[np.newaxis, :, :]).mean(axis=1, keepdims=True) * self.activations[:, step+1][:, np.newaxis, :]).mean(axis=0)
        rec_plasticity_terms[5] = self.rec_weights * (((1 - self.activations[:, step])[:, :, np.newaxis] * (self.rec_adj * self.rec_weights)[np.newaxis, :, :]).mean(axis=1, keepdims=True) * self.activations[:, step+1][:, np.newaxis, :]).mean(axis=0)
#         rec_plasticity_terms[6] = self.rec_weights * ((self.activations[:, step][:, :, np.newaxis] * (self.rec_adj * self.rec_weights)[np.newaxis, :, :]).mean(axis=2, keepdims=True) * self.activations[:, step+1][:, np.newaxis, :]).mean(axis=0)
#         rec_plasticity_terms[7] = self.rec_weights * (((1 - self.activations[:, step])[:, :, np.newaxis] * (self.rec_adj * self.rec_weights)[np.newaxis, :, :]).mean(axis=2, keepdims=True) * self.activations[:, step+1][:, np.newaxis, :]).mean(axis=0)
        
        self.input_weights += (self.plasticity_params[0][:, np.newaxis, np.newaxis] * input_plasticity_terms).sum(axis=0)
        self.rec_weights += (self.plasticity_params[1][:, np.newaxis, np.newaxis] * rec_plasticity_terms).sum(axis=0)