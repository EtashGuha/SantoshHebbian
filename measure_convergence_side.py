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
	

def k_cap_idx(input, cap_size):
	return np.argsort(input, axis=-1)[..., -cap_size:]
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
	def __init__(self, n_neurons, cap_size, n_rounds, sparsity, plasticity):
		self.n_neurons = n_neurons
		self.cap_size = cap_size
		self.n_rounds = n_rounds
		self.sparsity = sparsity
		self.generate_graph()
		self.reset_weights()
		self.plasticity = plasticity
		
	def generate_graph(self):
		# self.input_adj = rng.random((n_inputs, self.n_neurons)) < self.sparsity
		self.adj = np.logical_and(rng.random((self.n_neurons, self.n_neurons)) < self.sparsity, 
									  np.logical_not(np.eye(self.n_neurons, dtype=bool)))
		
	def reset_weights(self):
		self.weights = np.ones((self.n_neurons, self.n_neurons)) * self.adj
			
	def forward(self, update=False):	
		self.idx = np.zeros((self.n_rounds, self.cap_size), dtype=int)
		self.activations = np.zeros((self.n_rounds, self.n_neurons))

		self.idx[0] = np.arange(self.cap_size)
		self.activations[0][self.idx[0]] = 1
		for t in range(self.n_rounds-1):
			self.idx[t+1] = self.weights[self.idx[t]].sum(axis=0).argsort()[-self.cap_size:]
			self.activations[t+1][self.idx[t+1]] = 1
			if update: 
				self.update_weights(t)
		return self.activations
	
	def update_weights(self, step):
		self.weights[np.ix_(self.idx[step], self.idx[step+1])] *= 1 + self.plasticity
		self.weights *= self.adj

rounds_to_all = dict()
all_values = []

params = np.linspace(0, 1, num=1)

def run_test(val, n_neurons, cap_size, sparsity_val):
	tau = 0 
	n_iter = 2
	n_nets = 100
	n_rounds = 200
	n_inputs = 500
	# print(n_rounds)
	nets = [BrainNet(n_neurons, int(cap_size), n_rounds, sparsity_val, plasticity=val) for i in range(n_nets)]
	obj = np.zeros((n_rounds, n_iter, n_nets))
	num_new_comers = np.zeros((n_rounds, n_iter, n_nets))
	one_period_overlaps = np.zeros((n_rounds, n_iter, n_nets))
	two_period_overlaps = np.zeros((n_rounds, n_iter, n_nets))
	three_period_overlaps = np.zeros((n_rounds, n_iter, n_nets))
	four_period_overlaps = np.zeros((n_rounds, n_iter, n_nets))

	for i in range(1, n_iter):

		for j, net in tqdm(enumerate(nets)):
			net.reset_weights()
			all_outputs = {}
			temp_outputs = net.forward(update=True)
			for round_idx in range(1, n_rounds):
				all_outputs[round_idx] = set(np.where(temp_outputs[round_idx] > 0)[0])
			all_activations = set()
			set_again = False
			past_convergence = False
			for round_idx in range(5, n_rounds):
				outputs = set(all_outputs[round_idx])
				new_comers = outputs.difference(all_activations)
				all_activations.update(outputs)
				# print(len(new_comers))
				num_new_comers[round_idx, i, j] = len(new_comers)
				one_period_overlaps[round_idx, i, j] = len(outputs.intersection(set(all_outputs[round_idx - 1])))
				two_period_overlaps[round_idx, i, j] = len(outputs.intersection(set(all_outputs[round_idx - 2])))
				three_period_overlaps[round_idx, i, j] = len(outputs.intersection(set(all_outputs[round_idx - 3])))
				four_period_overlaps[round_idx, i, j] = len(outputs.intersection(set(all_outputs[round_idx - 4])))

				# if past_convergence:
				# 	how_far += 1
				# 	if how_far == 30:
				# 		breakpoint()
				# 		set_again = True
				# 		one_period_overlaps = len(outputs.intersection(set(all_outputs[round_idx - 1])))
				# 		two_period_overlaps = len(outputs.intersection(set(all_outputs[round_idx - 2])))
				# 		three_period_overlaps = len(outputs.intersection(set(all_outputs[round_idx - 3])))
				# 		four_period_overlaps = len(outputs.intersection(set(all_outputs[round_idx - 4])))
				# 		if one_period_overlaps == []:
				# 			breakpoint()
				# 		break	
	# occurence_chart = [one_period_overlaps > period_threshold, two_period_overlaps > period_threshold, three_period_overlaps > period_threshold]
	
	return (one_period_overlaps, two_period_overlaps, three_period_overlaps, four_period_overlaps, num_new_comers)
if __name__ == '__main__':
	rounds_to_all = dict()
	total_dict = {}
		
	for n_neurons in [1000]:
		for cap_size_type in ["con_30"]:
			if cap_size_type == "sqrt":
				cap_size = int(np.sqrt(n_neurons))
			elif "con" in cap_size_type:
				cap_size = int(cap_size_type.split("_")[-1])
				print(cap_size)
			elif cap_size_type == "div":
				cap_size = int(n_neurons/20)
			# for cap_size in [30]:
			for beta in [1]:
				for sparsity_val in [1]:
					all_jobs = []
					pool = mp.Pool(15)
					params = [beta]
					for val in params:
						one_period_overlaps, two_period_overlaps, three_period_overlaps, four_period_overlaps, num_new_comers =run_test(beta, int(n_neurons), int(cap_size), sparsity_val)
					
	import matplotlib.pyplot as plt

	plt.plot(list(range(200)), np.mean(one_period_overlaps[:, 1, :], axis=1), label="one period")
	plt.plot(list(range(200)), np.mean(two_period_overlaps[:, 1, :], axis=1) - np.mean(one_period_overlaps[:, 1, :], axis=1), label="two period")
	plt.plot(list(range(200)), np.mean(three_period_overlaps[:, 1, :], axis=1) - np.mean(one_period_overlaps[:, 1, :], axis=1), label="three period")
	plt.plot(list(range(200)), np.mean(four_period_overlaps[:, 1, :], axis=1) - np.mean(two_period_overlaps[:, 1, :], axis=1), label="four period")

	plt.plot(list(range(200)), np.mean(num_new_comers[:, 1, :], axis=1), label="newcomers")
	plt.legend()
	plt.savefig("newcomers_pls.png")
	
	# plt.hist([one_period_overlaps[195, 1, :], two_period_overlaps[195, 1, :] - one_period_overlaps[195, 1, :],three_period_overlaps[195, 1, :] - one_period_overlaps[195, 1, :]], label=["one period", "two", "three"])
	# plt.legend()
	# plt.savefig("again_hist.png")
	# breakpoint()
	import pickle
	with open("/nethome/eguha3/SantoshHebbian/new_code_smooth_n_2p_con.pkl", "wb") as f:
		pickle.dump(total_dict, f)

	

