import argparse
import multiprocessing as mp
from tqdm import tqdm
from run import run
import numpy as np
import pickle

def process(num_samples):
	pool = mp.Pool(10)
	all_values  = []
	for val in np.linspace(0, 5, 100):
		final_list = []
		all_densities = []
		for _ in range(num_samples):
			final_list.append(pool.apply_async(func=run, args=(val, 50, 1e-3)))
		for item in tqdm(final_list):
			density = item.get()
			all_densities.append(density)
		
		all_values.append((val, np.mean(all_densities)))
	with open("all_values.pkl", "wb") as f:
		pickle.dump(all_values, f)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--num_samples", default=10, type=int)
	parser.add_argument("--n_inputs", default=500, type=int)
	parser.add_argument("--n_neurons", default=500, type=int)
	parser.add_argument("--cap_size", default=10, type=int)
	parser.add_argument("--n_rounds", default=10, type=int)
	parser.add_argument("--sparsity", default=0.05, type=int)

	args = parser.parse_args()
	process(args.num_samples)
