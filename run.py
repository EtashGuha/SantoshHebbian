import numpy as np
from model_utils import initialize_net

def run(val, n_iter, step_size, n_inputs, n_neurons, cap_size, n_round, sparsity):
	net, inputs = initialize_net(n_inputs=n_inputs, n_neurons=n_neurons, cap_size=cap_size, n_rounds=n_rounds, sparsity=0.05)

	obj = np.zeros(n_iter)
	w_obj = np.zeros(n_iter)

	outputs = net.forward(inputs, update=True)[:, -1]
	obj[0] = (outputs[:, :, np.newaxis] * outputs[:, np.newaxis, :] * net.rec_adj[np.newaxis, :, :]).sum(axis=(1,2)).mean()
	w_obj[0] = (outputs[:, :, np.newaxis] * outputs[:, np.newaxis, :] * (net.rec_adj * net.rec_weights)[np.newaxis, :, :]).sum(axis=(1,2)).mean()

	for i in range(1, n_iter):
		direction = np.zeros((2, 6))
		direction[0,0] = val
		direction[0,1] = val
		
		old_params = net.plasticity_params.copy()
		net.plasticity_params = direction * 1e-3
		
		outputs = net.forward(inputs, update=True)[:, -1]
		obj[i] = (outputs[:, :, np.newaxis] * outputs[:, np.newaxis, :] * net.rec_adj[np.newaxis, :, :]).sum(axis=(1,2)).mean()
		w_obj[i] = (outputs[:, :, np.newaxis] * outputs[:, np.newaxis, :] * (net.rec_adj * net.rec_weights)[np.newaxis, :, :]).sum(axis=(1,2)).mean()

	return obj.mean()