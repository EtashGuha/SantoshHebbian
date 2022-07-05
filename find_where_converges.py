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

n_inputs = 500
n_neurons = 1000
cap_size = 30
n_rounds = 5000
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

def run_test(val):
	rounds_to_all = dict()
	print("hello")
	tau = 0 
	t1 = time.time()
	n_iter = 2
	n_nets = 5
	step_size = 1e-3
	obj = np.zeros((n_rounds, n_iter, n_nets))
	greedy_results = np.zeros((n_nets))
	min_degree_results = np.zeros((n_nets))
	extended_greedy_results = np.zeros((n_nets))
	comps_before_after = np.zeros((n_rounds, n_iter, n_nets))
	comps_before_after_two = np.zeros((n_rounds, n_iter, n_nets))
	comps_before_after_three = np.zeros((n_rounds, n_iter, n_nets))
	comps_before_after_four = np.zeros((n_rounds, n_iter, n_nets))
	comps_before_after_five = np.zeros((n_rounds, n_iter, n_nets))
	num_new_comers = np.zeros((n_rounds, n_iter, n_nets))

	periodics = np.zeros((n_rounds, n_iter, n_nets))
	overlap_penalty = 5e-1
	done_before = False
	
	nets = [BrainNet(n_inputs, n_neurons, cap_size, n_rounds, sparsity, homeostasis=False, tau=tau, plasticity=val) for i in range(n_nets)]
	for i, net in enumerate(nets):
		net.reset_weights()
		all_outputs = net.forward(update=True)
		
		for round_idx in range(n_rounds):
			outputs = all_outputs[round_idx]
			obj[round_idx, 0, i] = (outputs[ :, np.newaxis] * outputs[ np.newaxis, :] * net.rec_adj[np.newaxis, :, :]).sum(axis=(1,2)).mean()

	old_params = np.zeros((2, 6))
	
	for i in range(1, n_iter):
		for j, net in tqdm(enumerate(nets)):
			net.reset_weights()

			all_outputs = net.forward(update=True)
			all_activations = np.zeros((n_neurons))

			for round_idx in range(1, n_rounds):
				outputs = all_outputs[round_idx]
				new_comers = np.logical_and(outputs > 0 , all_activations == 0)
				all_activations += outputs
				num_new_comers[round_idx, i, j] = np.sum(new_comers)
				
				comps_before_after_four[round_idx, i, j] = (all_outputs[round_idx] * all_outputs[round_idx-4]).sum()
				comps_before_after_three[round_idx, i, j] = (all_outputs[round_idx] * all_outputs[round_idx-3]).sum()
				comps_before_after_two[round_idx, i, j] = (all_outputs[round_idx] * all_outputs[round_idx-2]).sum()
				comps_before_after[round_idx, i, j] = (all_outputs[round_idx] * all_outputs[round_idx-1]).sum()
				comps_before_after_five[round_idx, i, j] = (all_outputs[round_idx] * all_outputs[round_idx-5]).sum()


				if comps_before_after[round_idx, i, j] + 15 < comps_before_after_two[round_idx, i, j]:
					periodics[round_idx, i, j] = 1
				else:
					periodics[round_idx, i, j] = 0
				# print((all_outputs[round_idx] * all_outputs[round_idx-1]).sum())
				# print("{} {} {}".format(round_idx > 80, (all_outputs[round_idx] * all_outputs[round_idx-1]).sum() < 2, not done_before))                
				obj[round_idx, i, j] = (outputs[ :, np.newaxis] * outputs[ np.newaxis, :] * net.rec_adj[np.newaxis, :, :]).sum(axis=(1,2)).mean()
	for j, net in enumerate(nets):
		# greedy_results[j] = greedy(net.rec_adj, k=cap_size)
		# min_degree_results[j] = min_degree(net.rec_adj, k=cap_size)
		# extended_greedy_results[j] = extended_greedy(net.rec_adj, k=cap_size, n=0)
		greedy_results[j] = 0
		min_degree_results[j] = 0
		extended_greedy_results[j] = 0
	for round_idx in range(n_rounds):
		rounds_to_all[round_idx] = (0, val, obj[round_idx, -1, :].mean(), comps_before_after[round_idx, -1, :].mean(), comps_before_after_two[round_idx, -1, :].mean(), comps_before_after_three[round_idx, -1, :].mean(), comps_before_after_four[round_idx, -1, :].mean(), comps_before_after_five[round_idx, -1, :].mean(),  periodics[round_idx, -1, :].mean(), greedy_results.mean(), min_degree_results.mean(), extended_greedy_results.mean(), num_new_comers[round_idx, -1, :].mean())
	return rounds_to_all
rounds_to_all = dict()
pool = mp.Pool(10)
all_jobs = []
params = list(np.linspace(.001, .005, 4))
params = [params[2]]
breakpoint()
for val in params:
	# all_jobs.append(pool.apply_async(run_test, args=[val]))
	all_jobs.append(run_test(val))
for job in all_jobs:
	# temp_dict = job.get()
	temp_dict = job
	for key in temp_dict:
		if key in rounds_to_all:
			rounds_to_all[key].append(temp_dict[key])
		else:
			rounds_to_all[key] = [temp_dict[key]]

pool.close()
pool.terminate()
pool.join()

import pickle
with open("/nethome/eguha3/SantoshHebbian/checking_cutoff.pkl", "wb") as f:
	pickle.dump(rounds_to_all, f)