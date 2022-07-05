import networkx as nx
import random
from operator import itemgetter

def min_degree(adj_matrix, k = 10):
	graph = nx.from_numpy_matrix(adj_matrix)
	nodes_degrees_pair = graph.degree()

	while len(graph.nodes) > k:
		nodes, degrees = list(zip(*graph.degree()))
		min_degree = min(degrees)
		idx = degrees.index(min_degree)
		graph.remove_node(nodes[idx])

	return graph.number_of_edges()

def greedy(adj_matrix, k = 10):
	graph = nx.from_numpy_matrix(adj_matrix)

	nodes = [random.randint(0, graph.number_of_nodes() - 1)]

	while len(nodes) < k - 1:
		all_neighbors = []
		for node in nodes:
			all_neighbors.extend(graph.neighbors(node))
		all_neighbors = [i for i in all_neighbors if i not in nodes]
		nodes.append(max(set(all_neighbors), key=all_neighbors.count))
	return graph.subgraph(nodes).number_of_edges()

def extended_greedy(adj_matrix, n=10, k = 10):
	graph = nx.from_numpy_matrix(adj_matrix)

	nodes = [random.randint(0, graph.number_of_nodes() - 1)]

	while len(nodes) < k - 1:
		all_neighbors = []
		for node in nodes:
			all_neighbors.extend(graph.neighbors(node))
		all_neighbors = [i for i in all_neighbors if i not in nodes]
		nodes.append(max(set(all_neighbors), key=all_neighbors.count))
	
	for t in range(int(n)):
		subgraph = graph.subgraph(nodes)
		connectivities = [(node, len(set(graph.neighbors(node)) & set(nodes))) for node in graph.nodes() - nodes]
		best_node, in_connectivity = max(connectivities, key=itemgetter(1))
		worst_node, out_connectivity = min(subgraph.degree, key=itemgetter(1))
		
		if in_connectivity < out_connectivity:
			break
		nodes.remove(worst_node)
		nodes.append(best_node)
	return graph.subgraph(nodes).number_of_edges()
