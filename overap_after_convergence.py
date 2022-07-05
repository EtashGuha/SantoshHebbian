import numpy as np
rng = np.random.default_rng()
import matplotlib.pyplot as plt
import networkx as nx
import random
import scipy
from heuristics import min_degree, greedy, extended_greedy
import itertools
from tqdm import tqdm
import time 
import multiprocessing as mp
import argparse

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
	def __init__(self, n_inputs, n_neurons, cap_size, n_rounds, sparsity, homeostasis, tau = 0, plasticity = 0):
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
		# self.input_adj = rng.random((n_inputs, self.n_neurons)) < self.sparsity
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
		# flattened_weights = self.rec_weights.flatten()/np.sum(self.rec_weights)
		# entropy_grad = -1 * (entropy4(flattened_weights) + np.log(flattened_weights))/(1 - flattened_weights)
		# entropy_grad = entropy_grad.reshape((n_neurons, n_neurons))
		# entropy_grad = entropy_grad - np.min(entropy_grad)
		
		plasticity_rule = self.activations[step-1,:][:,np.newaxis] * self.activations[step, :][np.newaxis, :] * self.rec_adj 
		# print("max p: {} min p: {} max e: {} min e: {}".format(np.max(plasticity_rule), np.min(plasticity_rule), np.max(entropy_grad), np.min(entropy_grad)))
		self.rec_weights += plasticity_rule * self.plasticity 
		if self.homeostasis:
			self.rec_weights = self.rec_weights / (self.rec_weights.sum(axis=1)[:, np.newaxis])


rounds_to_all = dict()
all_values = []

params = np.linspace(0, 1, num=1)

def run_test(val, n_neurons, cap_size):
	tau = 0 
	n_iter = 2
	n_nets = 1
	n_rounds = 100
	n_inputs = 500
	sparsity = 0.05
	# print(n_rounds)
	obj = np.zeros((n_rounds, n_iter, n_nets))
	num_new_comers = np.zeros((n_rounds, n_iter, n_nets))

	one_period_overlaps = []
	two_period_overlaps = []
	
	nets = [BrainNet(n_inputs, n_neurons, cap_size, n_rounds, sparsity, homeostasis=False, tau=tau, plasticity=val) for i in range(n_nets)]
	for i, net in enumerate(nets):
		net.reset_weights()
		all_outputs = net.forward(update=True)
		
		for round_idx in range(n_rounds):
			outputs = all_outputs[round_idx]
			obj[round_idx, 0, i] = (outputs[ :, np.newaxis] * outputs[ np.newaxis, :] * net.rec_adj[np.newaxis, :, :]).sum(axis=(1,2)).mean()

	for i in range(1, n_iter):
		for j, net in enumerate(nets):
			net.reset_weights()

			all_outputs = net.forward(update=True)
			all_activations = np.zeros((n_neurons))

			past_convergence = False
			for round_idx in range(1, n_rounds):
				outputs = all_outputs[round_idx]
				new_comers = np.logical_and(outputs > 0 , all_activations == 0)
				all_activations += outputs
				num_new_comers[round_idx, i, j] = np.sum(new_comers)
				
				if not past_convergence and np.sum(new_comers) == 0:
					past_convergence = True
					how_far = 0

				if past_convergence:
					how_far += 1
					if how_far == 30:
						one_period_overlaps = np.sum(np.logical_and(outputs > 0 , all_outputs[round_idx - 1] > 0))
						two_period_overlaps = np.sum(np.logical_and(outputs > 0 , all_outputs[round_idx - 2] > 0))	
						three_period_overlaps = np.sum(np.logical_and(outputs > 0 , all_outputs[round_idx - 3] > 0))	
						four_period_overlaps = np.sum(np.logical_and(outputs > 0 , all_outputs[round_idx - 4] > 0))	
						break
	# occurence_chart = [one_period_overlaps > period_threshold, two_period_overlaps > period_threshold, three_period_overlaps > period_threshold]
	return (one_period_overlaps, two_period_overlaps, three_period_overlaps, four_period_overlaps)

if __name__ == '__main__':
	rounds_to_all = dict()
	total_dict = {}
	for n_neurons in np.linspace(500, 5000, num=50):
		cap_size = np.sqrt(n_neurons)
		for beta in [1]:
			all_jobs = []
			pool = mp.Pool(10)
			params = [beta] * 250
			for val in params:
				all_jobs.append(pool.apply_async(run_test, args=[val, int(n_neurons), int(cap_size)]))
				# run_test(val, int(n_neurons), int(cap_size))
			one_period_overlaps = []
			two_period_overlaps = []
			three_period_overlaps = []
			occurence_chars = []

			for job in tqdm(all_jobs):
				try:
					one, two, three, four = job.get()
					one_period_overlaps.append(one)
					two_period_overlaps.append(two)
					three_period_overlaps.append(three)
					occurence_chars.append(four)
				except:
					continue
			
			if (beta, n_neurons) not in total_dict:
				total_dict[(beta, n_neurons)] = []
			total_dict[(beta, n_neurons)].append((one_period_overlaps, two_period_overlaps, three_period_overlaps, occurence_chars))
			pool.close()
			pool.terminate()
			pool.join()

	import pickle

	with open("/nethome/eguha3/SantoshHebbian/prob_periodicity_large_n_more_trials.pkl", "wb") as f:
		pickle.dump(total_dict, f)

	

