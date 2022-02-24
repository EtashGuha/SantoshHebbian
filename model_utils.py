from models import BrainNet
import numpy as np

def initialize_net(n_inputs=500, n_neurons=500, cap_size=10, n_rounds=10, sparsity=0.05):
	inputs = np.zeros((1, n_inputs))
	inputs[0, :cap_size] = 1
	net = BrainNet(n_inputs, n_neurons, cap_size, n_rounds, sparsity)
	net.plasticity_params = np.zeros((2, 6))
	net.plasticity_params = np.array([[1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]]) * 1e-2
	return net, inputs