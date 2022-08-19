from tkinter import N
import matplotlib.pyplot as plt
import argparse
import pickle
from operator import itemgetter
import numpy as np


def plot_smaller_converge(name, data):
	with open(data, "rb") as f:
		rounds_to_all= pickle.load(f)
	
	round_idxes = []
	smallest_where_converges = []

	for round_idx in rounds_to_all.keys():
		list_of_res = rounds_to_all[round_idx]
		list_of_res.sort(key=itemgetter(1))
		for item in list_of_res:
			if item[-1] == 0:
				smallest_where_converges.append(item[1])
				round_idxes.append(round_idx)
				break
	plt.plot(round_idxes, smallest_where_converges, label='Smallest Plasticity Value with Convergence')
	plt.xlabel("Time Horizon")
	plt.ylabel("Smallest Plasticity Value with Convergence")
	# plt.savefig(name)

def draw_overlap_after_convergence(name, data):
	with open(data, "rb") as f:
		dictionanry = pickle.load(f)
	# breakpoint()
	one_period_overlaps, two_period_overlaps, three_period_overlaps, num_newcomers = dictionanry[list(dictionanry.keys())[0]][0]
	one_period_overlaps = np.asarray(one_period_overlaps)
	two_period_overlaps = np.asarray(two_period_overlaps)
	three_period_overlaps = np.asarray(three_period_overlaps)
	# four_period_overlaps = np.asarray(four_period_overlaps)
	num_newcomers = np.mean(np.mean(np.asarray(num_newcomers),axis=0), axis=1)
	plt.plot(list(range(len(num_newcomers))), num_newcomers)
	plt.xlabel("Round")
	plt.ylabel("Number of newcomers")
	plt.savefig("new_code_numnewcomers.png")
	plt.clf()
	# plt.hist([one_period_overlaps, two_period_overlaps - one_period_overlaps, three_period_overlaps - one_period_overlaps], bins=30, label=["one period", "two period", 'three period'])
	plt.hist([two_period_overlaps - one_period_overlaps], bins=30, label=["two period"])

	breakpoint()
	# breakpoint()
	# plt.hist(two_period_overlaps, bins=30, label='two period')
	plt.xlabel("Overlap Size")
	plt.ylabel("Frequency")
	plt.legend()
	plt.title("One vs Two period overlap after convergence")
	plt.savefig(name)

def draw_newcomers(name, data):
	with open(data, "rb") as f:
		rounds_to_all= pickle.load(f)
	new_dict = {}

	plot_tau = False
	if not plot_tau:
		for key in rounds_to_all.keys():
			all_values = rounds_to_all[key]
			taus, vals, densities, comps, comps_two, comps_three, comps_four, comps_five, periodics,  greedy_results, min_degree_results, greedy_extended_results, num_newcomers = list(zip(*all_values))
			for i in range(len(vals)):
				if vals[i] not in new_dict:
					new_dict[vals[i]] = [(key, taus[i], densities[i], comps[i], comps_two[i], comps_three[i], comps_four[i], comps_five[i], greedy_results[i], min_degree_results[i], greedy_extended_results[i], num_newcomers[i])]
				else:
					new_dict[vals[i]].append((key, taus[i], densities[i], comps[i], comps_two[i], comps_three[i], comps_four[i],  comps_five[i], greedy_results[i], min_degree_results[i], greedy_extended_results[i], num_newcomers[i]))
	else:
		for key in rounds_to_all.keys():
			all_values = rounds_to_all[key]
			taus, vals, densities, comps, greedy_results, min_degree_results, greedy_extended_results, num_newcomers = list(zip(*all_values))
			for i in range(len(vals)):
				if taus[i] not in new_dict:
					new_dict[taus[i]] = [(key, vals[i], densities[i], comps[i], greedy_results[i], min_degree_results[i], greedy_extended_results[i], num_newcomers[i])]
				else:
					new_dict[taus[i]].append((key, vals[i], densities[i], comps[i], greedy_results[i], min_degree_results[i], greedy_extended_results[i], num_newcomers[i]))
	import matplotlib.pyplot as plt 
	import math
	counter = 0
	already_plotted_greedy = False
	new_array = []
	for key in new_dict.keys():

		all_values = new_dict[key]

		vals, taus, densities, comps, comps_two, comps_three, comps_four, comps_five, greedy_results, min_degree_results, greedy_extended_results, num_newcomers = list(zip(*all_values))
		num_newcomers = np.asarray(num_newcomers)
		comps = np.asarray(comps)


		comps = np.asarray(comps)
		comps_two = np.asarray(comps_two)
		comps_three = np.asarray(comps_three)
		comps_four = np.asarray(comps_four)


		idx_of_max = np.argmax(densities)

		new_array.append((key, densities[idx_of_max]))
		num_rounds = len(vals)
		# plt.plot(vals, np.asarray(comps_four) - np.asarray(comps_two),  label=math.ceil(key * 10000)/10000)
		# plt.plot(vals, np.asarray(comps_five) - np.asarray(comps), label=math.ceil(key * 10000)/10000)
		# plt.plot(vals, 30 - num_newcomers, label="Number of Newcomers")
		# plt.plot(vals, comps, label="one period")
		# plt.plot(vals, comps_two - comps , label="two period")
		# plt.plot(vals, comps_three - comps, label="three period")
		# plt.plot(vals, comps_four, label="four period")
		plt.plot(vals, np.asarray(comps_two) + np.asarray(comps_three) - np.asarray(comps), label=math.ceil(key * 10000)/10000)
	plt.plot(vals, [30] * num_rounds, label="30 Line")



	
	plt.title("Three + Two - One")

	plt.xlabel("Round Index")
	plt.ylabel("Overlap Size")
	
	plt.legend(title="Plasticity Value")
	plt.savefig(name)

def draw_best_at_each_round(name, data):
	with open(data, "rb") as f:
		rounds_to_all= pickle.load(f)
	best_vals = []
	for key in rounds_to_all:
		best_val = max(rounds_to_all[key], key=itemgetter(2))[1]
		best_vals.append(best_val)
	print(best_vals)
	plt.plot(rounds_to_all.keys(), best_vals, label="True Optimal Plasticity Value")
	plt.xlabel("Time Horizon")
	plt.ylabel("Optimal Plasticity Value")
	# plt.savefig(name)

def draw_occurence_chart(name, data):
	with open(data, "rb") as f:
		one_period_overlaps, two_period_overlaps, three_period_overlaps, occurence_charts = pickle.load(f)
	plt.rcParams["figure.figsize"] = (20,3)
	one_period_overlaps = np.asarray(one_period_overlaps)
	two_period_overlaps = np.asarray(two_period_overlaps)
	three_period_overlaps = np.asarray(three_period_overlaps)
	only_one = 0
	only_two = 0
	only_three = 0
	one_two = 0
	two_three = 0
	one_three = 0
	none_at_all = 0
	all_of_them = 0
	occurence_threshold = 33
	for i in range(len(one_period_overlaps)):
		
		occurence_chart = [one_period_overlaps[i] > occurence_threshold, two_period_overlaps[i] - one_period_overlaps[i] > occurence_threshold, three_period_overlaps[i] - one_period_overlaps[i] > occurence_threshold]
		if occurence_chart == [1, 0, 0]:
			only_one += 1
		elif occurence_chart == [1, 1, 0]:
			one_two += 1
		elif occurence_chart == [1, 0, 1]:
			one_three += 1
		elif occurence_chart == [1, 1, 1]:
			all_of_them += 1
		elif occurence_chart == [0, 1, 0]:
			only_two += 1
		elif occurence_chart == [0, 1, 1]:
			two_three += 1
		elif occurence_chart == [0, 0, 1]:
			only_three += 1
		elif occurence_chart == [0, 0, 0]:
			none_at_all += 1

	plt.bar(["only_one","only_two","only_three","one_two","two_three","one_three","none_at_all","all_of_them"], [only_one,only_two,only_three,one_two,two_three,one_three,none_at_all,all_of_them])
	plt.xlabel("Perioicity Type")
	plt.ylabel("Frequency")
	# plt.figure(figsize=(8, 6), dpi=80)
	plt.savefig(name)
def overlap_after_convergence(name, data):
	with open(data, "rb") as f:
		rounds_to_all= pickle.load(f)
	new_dict = {}

	plot_tau = False
	if not plot_tau:
		for key in rounds_to_all.keys():
			if key > 10:
				break
			all_values = rounds_to_all[key]
			taus, vals, densities, comps, compstwo, greedy_results, min_degree_results, greedy_extended_results, num_newcomers = list(zip(*all_values))
			for i in range(len(vals)):
				if vals[i] not in new_dict:
					new_dict[vals[i]] = [(key, taus[i], densities[i], comps[i], greedy_results[i], min_degree_results[i], greedy_extended_results[i], num_newcomers[i])]
				else:
					new_dict[vals[i]].append((key, taus[i], densities[i], comps[i], greedy_results[i], min_degree_results[i], greedy_extended_results[i], num_newcomers[i]))
	else:
		for key in rounds_to_all.keys():
			all_values = rounds_to_all[key]
			taus, vals, densities, comps, greedy_results, min_degree_results, greedy_extended_results, num_newcomers = list(zip(*all_values))
			for i in range(len(vals)):
				if taus[i] not in new_dict:
					new_dict[taus[i]] = [(key, vals[i], densities[i], comps[i], greedy_results[i], min_degree_results[i], greedy_extended_results[i], num_newcomers[i])]
				else:
					new_dict[taus[i]].append((key, vals[i], densities[i], comps[i], greedy_results[i], min_degree_results[i], greedy_extended_results[i], num_newcomers[i]))
	import matplotlib.pyplot as plt 
	import math
	counter = 0
	already_plotted_greedy = False
	new_array = []
	for key in new_dict.keys():
		all_values = new_dict[key]
		vals, other_measures, densities, comps, greedy_results, min_degree_results, greedy_extended_results, num_newcomers = list(zip(*all_values))
		num_newcomers = np.asarray(num_newcomers)
		comps = np.asarray(comps)




		idx_of_max = np.argmax(densities)

		new_array.append((key, densities[idx_of_max]))

			
		plt.plot(vals, num_newcomers, label=math.ceil(key * 100)/100)
		
	  
		plt.title("Num Newcomers at each Round")

		plt.xlabel("Round Index")
		plt.ylabel("Num Newcomers")
		
		plt.legend(title="Cap size")
		plt.savefig(name)

def draw(name, data):
	with open(data, "rb") as f:
		rounds_to_all= pickle.load(f)
	new_dict = {}

	plot_tau = False
	if not plot_tau:
		for key in rounds_to_all.keys():
			all_values = rounds_to_all[key]
			taus, vals, densities, comps, compstwo, greedy_results, min_degree_results, greedy_extended_results  = list(zip(*all_values))
			# greedy_extended_results = np.zeros_like(greedy_results)
			for i in range(len(vals)):
				if vals[i] not in new_dict:
					new_dict[vals[i]] = [(key, taus[i], densities[i], comps[i], greedy_results[i], min_degree_results[i], greedy_extended_results[i])]
				else:
					new_dict[vals[i]].append((key, taus[i], densities[i], comps[i], greedy_results[i], min_degree_results[i], greedy_extended_results[i]))
	else:
		for key in rounds_to_all.keys():
			all_values = rounds_to_all[key]
			taus, vals, densities, comps, greedy_results, min_degree_results, greedy_extended_results = list(zip(*all_values))
			for i in range(len(vals)):
				if taus[i] not in new_dict:
					new_dict[taus[i]] = [(key, vals[i], densities[i], comps[i], greedy_results[i], min_degree_results[i], greedy_extended_results[i])]
				else:
					new_dict[taus[i]].append((key, vals[i], densities[i], comps[i], greedy_results[i], min_degree_results[i], greedy_extended_results[i]))
	import matplotlib.pyplot as plt 
	import math
	counter = 0
	already_plotted_greedy = False
	new_array = []
	for key in new_dict.keys():
		all_values = new_dict[key]
		vals, other_measures, densities, comps, greedy_results, min_degree_results, greedy_extended_results = list(zip(*all_values))
		densities = np.asarray(densities)
		greedies = np.asarray(greedy_results)
		min_degrees = np.asarray(min_degree_results)
		greedy_extendeds = np.asarray(greedy_extended_results)
		comps = np.asarray(comps)





		idx_of_max = np.argmax(densities)

		new_array.append((key, densities[idx_of_max]))
		plot_comps = False
		if not plot_comps:
			# plt.plot(vals, densities, label=math.ceil(key * 100)/100)
			if not already_plotted_greedy:
				already_plotted_greedy = True
				ax = plt.axes()
				ax.set(facecolor = "white")
				plt.plot(vals[:1000], greedies[:1000], label="greedy")
				plt.plot(vals[:1000], min_degrees[:1000], label="min_degree")
				plt.plot(vals[:1000], greedy_extendeds[:1000], label="greedy_extended")
		else:
			print(comps)
			ax = plt.axes()
			ax.set(facecolor = "white")
			print(key)
			plt.plot(vals, comps, label=math.ceil(key * 100)/100)

		plt.title("Different Heuristics for Density")

		plt.xlabel("N parameter for Extended")
		plt.ylabel("Density")
		
		plt.legend(title="Plasticity Params")
		plt.savefig(name)

def draw_period_converge(name, data):
	with open(data, "rb") as f:
		rounds_to_all= pickle.load(f)
	
	params = []
	periods = []
	for ele in rounds_to_all[99]:
		params.append(ele[1])
		periods.append(ele[4])
	
	plt.plot(params, periods)
	plt.xlabel("Hebbian Plasticity Parameters")
	plt.ylabel("Percent of Runs Exhibiting 2 Period Oscillations")
	plt.title("Plasticity vs. Probability of 2 Period Oscillation")
	plt.savefig("/nethome/eguha3/SantoshHebbian/plasticity_vs_osci.png")

def draw_interactions(name, data):
	with open(data, "rb") as f:
		di = pickle.load(f)

	
	for cap_size_val_type in [1, 3, 7, 10]:
		betas = []
		nneuronss = []
		prob_twos = []
		prob_ones = []
		prob_threes = []
		prob_fours = []
		prob_fives = []
		prob_sixes = []
		cap_sizes = []
		sparsity_vals = []
		stds = []
		cap_sizes = []
		for key in di.keys():		
		
			beta, cap_size_category, n_neurons, cap_size, sparsity_val = key
			if sparsity_val != cap_size_val_type:
				continue
		
			# if cap_size is not cap_size_type:
			# 	continue
			sparsity_vals.append(sparsity_val)

			# cap_sizes.append(cap_size)
			betas.append(beta)
			nneuronss.append(n_neurons)
			cap_sizes.append(cap_size)
			list_of_vals = np.squeeze(np.asarray(di[key]))
			means = np.mean(list_of_vals, axis=1)
			prob_ones.append(means[0]/cap_size)
			prob_twos.append((means[1] - means[0])/cap_size)
			prob_threes.append((means[2] - means[0])/cap_size)
			prob_fours.append((means[3] - means[1])/cap_size)
			prob_fives.append((means[4] - means[0])/cap_size)
			prob_sixes.append((means[5] - means[2] - means[1] + means[0])/cap_size)




			prob_period = (list_of_vals[1] - list_of_vals[0])/cap_size
			# print((means[1] - means[0])/cap_size)
			# plt.clf()
			# plt.hist(prob_period, bins=50)
			# plt.savefig("prob_periodicity_beta.png")

			# breakpoint()
			std = np.std(prob_period)
			stds.append(std)
		# breakpoint()
		breakpoint()
		plt.plot(nneuronss, prob_ones, label="1_{}".format(cap_size_val_type))
		plt.plot(nneuronss, prob_twos, label="2_{}".format(cap_size_val_type))
		plt.plot(nneuronss, prob_threes, label="3_{}".format(cap_size_val_type))
		# plt.plot(nneuronss, prob_fours, label="4_{}".format(cap_size_val_type))
		# plt.plot(nneuronss, prob_fives, label="5_{}".format(cap_size_val_type))
		# plt.plot(nneuronss, prob_sixes, label="6_{}".format(cap_size_val_type))

	# plt.xscale("log")
	plt.xlabel("number of neurons")
	plt.legend(title="periodicity")
	plt.ylabel("prob of Two period")
	plt.title("Prob of Two period for large n for given KP")
	# plt.legend(title="cap size")
	plt.savefig(name)



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--data")
	parser.add_argument("--name")


	args = parser.parse_args()
	# draw_best_at_each_round(args.name, args.data)
	# plot_smaller_converge(args.name, args.data)
	# plt.legend()
	# plt.savefig(args.name)
	draw_interactions(args.name, args.data)