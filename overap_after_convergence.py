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

		self.idx[0] = random.sample(range(self.n_neurons), self.cap_size)
		self.activations[0][self.idx[0]] = 1
		for t in range(self.n_rounds-1):
			a = self.weights[self.idx[t]].sum(axis=0)
			b =np.random.random(a.size)
			self.idx[t+1] = np.lexsort(np.stack((b, a), axis=0))[-self.cap_size:]
			# assert((a[np.argsort(a)[-self.cap_size:]] == a[self.idx[t+1]]).all())
			# breakpoint()
			self.activations[t+1][self.idx[t+1]] = 1
			if update: 
				self.update_weights(t)
			
		return self.activations
	
	def update_weights(self, step):
		try:
			self.weights[np.ix_(self.idx[step], self.idx[step+1])] *= 1 + self.plasticity
			self.weights *= self.adj
		except:
			breakpoint()

rounds_to_all = dict()
all_values = []

params = np.linspace(0, 1, num=1)

def run_test(val, n_neurons, cap_size, sparsity_val):
	tau = 0 
	n_iter = 2
	n_nets = 1
	n_rounds = 200
	n_inputs = 500
	# print(n_rounds)
	nets = [BrainNet(n_neurons, int(cap_size), n_rounds, sparsity_val, plasticity=val) for i in range(n_nets)]
	obj = np.zeros((n_rounds, n_iter, n_nets))
	num_new_comers = np.zeros((n_rounds, n_iter, n_nets))

		

	one_period_overlaps = []
	two_period_overlaps = []
	

	for i in range(1, n_iter):

		for j, net in enumerate(nets):
			net.reset_weights()
			all_outputs = {}
			temp_outputs = net.forward(update=True)
			for round_idx in range(1, n_rounds):
				all_outputs[round_idx] = set(np.where(temp_outputs[round_idx] > 0)[0])
			all_activations = set()
			set_again = False
			past_convergence = False
			for round_idx in range(1, n_rounds):
				outputs = set(all_outputs[round_idx])
				new_comers = outputs.difference(all_activations)
				all_activations.update(outputs)
				# print(len(new_comers))
				num_new_comers[round_idx, i, j] = len(new_comers)
				if not past_convergence and round_idx > 30 and np.sum(num_new_comers[round_idx-30:round_idx, i, j]) == 0:
					past_convergence = True
					# breakpoint()
					how_far = 0
					one_period_overlaps = len(outputs.intersection(set(all_outputs[round_idx - 1])))
					two_period_overlaps = len(outputs.intersection(set(all_outputs[round_idx - 2])))
					three_period_overlaps = len(outputs.intersection(set(all_outputs[round_idx - 3])))
					four_period_overlaps = len(outputs.intersection(set(all_outputs[round_idx - 4])))
					five_period_overlaps = len(outputs.intersection(set(all_outputs[round_idx - 5])))
					six_period_overlaps = len(outputs.intersection(set(all_outputs[round_idx - 6])))

					break
					if one_period_overlaps == []:
						breakpoint()

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
	try:
		return (one_period_overlaps, two_period_overlaps, three_period_overlaps, four_period_overlaps, five_period_overlaps, six_period_overlaps, num_new_comers)
	except:
		one_period_overlaps = len(outputs.intersection(set(all_outputs[round_idx - 1])))
		two_period_overlaps = len(outputs.intersection(set(all_outputs[round_idx - 2])))
		three_period_overlaps = len(outputs.intersection(set(all_outputs[round_idx - 3])))
		four_period_overlaps = len(outputs.intersection(set(all_outputs[round_idx - 4])))
		five_period_overlaps = len(outputs.intersection(set(all_outputs[round_idx - 5])))
		six_period_overlaps = len(outputs.intersection(set(all_outputs[round_idx - 6])))
		return (one_period_overlaps, two_period_overlaps, three_period_overlaps, four_period_overlaps, five_period_overlaps, six_period_overlaps, num_new_comers)
if __name__ == '__main__':
	rounds_to_all = dict()
	total_dict = {}
		
	for n_neurons in [500, 750, 1000, 1250, 1500]:
		# for cap_size in np.linspace(10, 250, 25):
		# 	cap_size_type = "con"
		for cap_size_type in ["con_25", "sqrt", "div"]:
			if cap_size_type == "sqrt":
				cap_size = int(np.sqrt(n_neurons))
			elif "con" in cap_size_type:
				cap_size = int(cap_size_type.split("_")[-1])
				print(cap_size)
			elif cap_size_type == "div":
				cap_size = int(n_neurons/20)
			
			for beta in [.25]:
				for k_div_pn_val in [1, 3, 7, 10]:
					all_jobs = []
					pool = mp.Pool(15)
					params = [beta] * 500
					sparsity_val = 1/k_div_pn_val * cap_size/np.log(n_neurons)
					for val in params:
						all_jobs.append(pool.apply_async(run_test, args=[beta, int(n_neurons), int(cap_size), sparsity_val]))
						# run_test(beta, int(n_neurons), int(cap_size), sparsity_val)
						# breakpoint()
					one_period_overlaps = []
					two_period_overlaps = []
					three_period_overlaps = []
					four_period_overlaps = []
					five_period_overlaps = []
					six_period_overlaps = []
					num_newcomers = []

					for job in tqdm(all_jobs):
						one, two, three, four, five, six, new_comers = job.get()
						one_period_overlaps.append(one)
						two_period_overlaps.append(two)
						three_period_overlaps.append(three)
						four_period_overlaps.append(four)
						five_period_overlaps.append(five)
						six_period_overlaps.append(six)
						num_newcomers.append(new_comers)
						
						
					
					if (beta, cap_size_type, n_neurons, cap_size, k_div_pn_val) not in total_dict:
						total_dict[(beta, cap_size_type, n_neurons, cap_size, k_div_pn_val)] = []
					
					total_dict[(beta, cap_size_type, n_neurons, cap_size, k_div_pn_val)].append((one_period_overlaps, two_period_overlaps, three_period_overlaps, four_period_overlaps, five_period_overlaps, six_period_overlaps, num_newcomers))
					pool.close()
					pool.terminate()
					pool.join()
	import pickle
	with open("/nethome/eguha3/SantoshHebbian/kdivpn.pkl", "wb") as f:
		pickle.dump(total_dict, f)

	

