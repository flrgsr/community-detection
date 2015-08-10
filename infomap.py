
"""
This module implements th infomap community detection method
"""
#__all__ = [""]
__author__ = """Florian Gesser (gesser.florian@googlemail.com)"""

NODE_FREQUENCY  = 'NODE_FREQUENCY'
EXIT            = 'EXIT'
EPSILON_REDUCED = 0.0000001

import numpy as np
import networkx as nx
import math
import sys
sys.path.append("/Users/florian/Data/Pending/GSOC/code/community_evaluation/mini_pipeline_community/")
import buildTestGraph as btg

class Partition(object):
	"""Represents a partition of the graph"""
	def __init__(self, graph):
		super(Partition, self).__init__()
		self.graph         = graph                                                          
		self.modules = dict(zip(self.graph, range(graph.number_of_nodes())))


		self.Nnode         = self.graph.number_of_nodes() 
		self.Nmod          = self.Nnode            
		self.degree        = sum(self.graph.degree().values())
		self.inverseDegree = 1.0/self.degree                      
														 
	
		"""node frequency as computed by page rank currenlty not used"""
		page_ranks = nx.pagerank(self.graph)
		nx.set_node_attributes(self.graph, 'NODE_FREQUENCY', page_ranks)
		""" In the beginning exit := degree of the node
			Use degrees as proxy for node frequency which is obtained from markov stationary distribution (random surfer calculation via page rank)

			Later: Exit := totoal weight of links to other modules
		"""
		degrees={i:self.graph.degree(i) for i in self.graph}
		nx.set_node_attributes(self.graph, 'EXIT', degrees)


		self.exit_log_exit     = 0.0
		self.degree_log_degree = 0.0
		self.exitDegree        = 0.0
		self.exit 			   = 0.0
		self.code_length       = 0.0

		self.mod_exit    = dict([])
		self.mod_degree  = dict([])
		self.mod_members = dict([])


	def init(self):
		self.nodeDegree_log_nodeDegree = sum([self.plogp(self.graph.degree(node)) for node in self.graph])

		for index, node in enumerate(self.graph):
			node_i_exit   = self.graph.node[node][EXIT]
			node_i_degree = self.graph.degree(node)

			self.exit_log_exit     += self.plogp(node_i_exit);
			self.degree_log_degree += self.plogp(node_i_exit + node_i_degree);
			self.exitDegree        += node_i_exit;

			self.mod_exit[node]    = node_i_exit
			self.mod_degree[node]  = node_i_degree
			# TODO set this to the correct value, in what data structure do we hold the members?
			self.mod_members[node] = 0

		self.exit = self.plogp(self.exitDegree)
		self.code_length = self.exit - 2.0 * self.exit_log_exit + self.degree_log_degree - self.nodeDegree_log_nodeDegree;


	def plogp(self, degree):
		"""Entropy calculation"""
		p = self.inverseDegree * degree
		return p * math.log(p, 2) if degree != 0 else 0.0

	def get_random_permutation_of_nodes(self):
		nodes = self.graph.nodes()
		return np.random.permutation(nodes)


	def neighbourhood_link_strength(self, node):
		community_links = {}
		for neighbour in self.graph.neighbors(node):
			community_of_neighbour = self.modules[neighbour]
			community_links[community_of_neighbour] = community_links.get(community_of_neighbour, 0) + 1
		return community_links

	def determine_best_new_module(self):
		randomSequence = self.get_random_permutation_of_nodes()

		for index, curr_node in enumerate(self.graph):
			pick   = randomSequence[curr_node-1]
			Nlinks = len(self.graph.neighbors(pick))
			wNtoM  = self.neighbourhood_link_strength(pick)

			fromM  = self.modules[pick]
			wfromM = sum([self.graph.node[neighbour][EXIT] for neighbour in self.graph.neighbors(pick)])

			bestM       = fromM
			best_weight = 0.0
			best_delta  = 0.0
	
			NmodLinks = len((wNtoM.keys()))

			for key, value in wNtoM.items():
				toM  = key
				wtoM = value

				deltaL = 0

				if toM != fromM:
					node_i_exit   = self.graph.node[pick][EXIT]
					node_i_degree = self.graph.degree(pick)

					delta_exit = self.plogp(self.exitDegree - 2*wtoM + 2*wfromM) - self.exit;
				
					delta_exit_log_exit = - self.plogp(self.mod_exit[fromM + 1])                               \
										  - self.plogp(self.mod_exit[toM + 1])                                 \
										  + self.plogp(self.mod_exit[fromM + 1] - node_i_exit + 2*wfromM)      \
										  + self.plogp(self.mod_exit[toM + 1] + node_i_exit - 2*wtoM)          
				
					delta_degree_log_degree = - self.plogp(self.mod_exit[fromM +1 ] + self.mod_degree[fromM +1])                                          \
											  - self.plogp(self.mod_exit[toM + 1] + self.mod_degree[toM + 1])                                              \
											  + self.plogp(self.mod_exit[fromM +1 ] + self.mod_degree[fromM +1] - node_i_exit - node_i_degree + 2*wfromM) \
											  + self.plogp(self.mod_exit[toM + 1] + self.mod_degree[toM + 1] + node_i_exit + node_i_degree - 2*wtoM)
				
					deltaL = delta_exit - 2.0 * delta_exit_log_exit + delta_degree_log_degree;
				
				if deltaL < best_delta:
					bestM = toM;
					best_weight = wtoM;
					best_delta = deltaL; 

		if bestM != fromM:

		return bestM

	def move(bestM):
		node_i_exit   = self.graph.node[curr_node][EXIT]
		node_i_degree = self.graph.degree(curr_node)

		self.exitDegree        -= self.mod_exit[fromM] + self.mod_exit[bestM];
		self.exit_log_exit     -= self.plogp(mod_exit[fromM]) + self.plogp(mod_exit[bestM]);
		self.degree_log_degree -= self.plogp(mod_exit[fromM] + self.mod_degree[fromM]) + self.plogp(mod_exit[bestM] + self.mod_degree[bestM]); 

		self.mod_exit[fromM]    -= node_i_exit - 2*wfromM;
		self.mod_degree[fromM]  -= node_i_degree;
		# TODO member structure
		self.mod_members[fromM] -= self.graph.node[pick];
		self.mod_exit[bestM]    += node_i_exit - 2*best_weight;
		self.mod_degree[bestM]  += node_i_degree;
		# TODO member structure
		self.mod_members[bestM] += self.graph.node[pick];

		self.exitDegree        += self.mod_exit[fromM] + mod_exit[bestM];
		self.exit_log_exit     += self.plogp(self.mod_exit[fromM]) + self.plogp(self.mod_exit[bestM]);
		self.degree_log_degree += self.plogp(self.mod_exit[fromM] + self.mod_degree[fromM]) + self.plogp(self.mod_exit[bestM] + self.mod_degree[bestM]); 
		
		self.exit = self.plogp(exitDegree);
		
		self.code_length = exit - 2.0*exit_log_exit + degree_log_degree - nodeDegree_log_nodeDegree;
		
		# See other TODO
		#node[pick]['MODULE'] = bestM;


	def first_pass(self):
		 best_new_module = self.determine_best_new_module()
		 self.move(best_new_module)


	def second_pass(self):
		aggregated_graph = nx.Graph()

		# The new graph consists of as many "supernodes" as there are partitions
		aggregated_graph.add_nodes_from(set(partition.values()))
		# make edges between communites, bundle more edges between nodes in weight attribute
		edge_list=[(partition[node1], partition[node2], attr.get('weight', 1) ) for node1, node2, attr in graph.edges(data=True)]
		sorted_edge_list = sorted(edge_list)
		sum_z = lambda tuples: sum(t[2] for t in tuples)
		weighted_edge_list = [(k[0], k[1], sum_z(g)) for k, g in groupby(sorted_edge_list, lambda t: (t[0], t[1]))]
		aggregated_graph.add_weighted_edges_from(weighted_edge_list)

		return aggregated_graph


def infomap(graph):
	# import pdb; pdb.set_trace()

	# partition.move()


	partition = Partition(graph)
	partition.init()

	parition_list = list()
	first_pass()
	new_codelength = partition.code_length
	partition = renumber(partition)
	parition_list.append(partition)
	codelength = new_codelength
	current_graph = second_pass(partition, graph)
	parition = parition.reinitialize(graph)

	while True:
		first_pass()
		new_codelength = partition.codelength
		if new_codelength - codelength < EPSILON_REDUCED :
			break
		partition = renumber(partition)
		parition_list.append(partition)
		codelength = new_codelength
		graph = second_pass(partition, graph)
		parition = parition.reinitialize(graph)
	return parition_list[:]


def main():
	#test prep
	graph = btg.build_graph()
	# call to main algorithm method
	infomap(graph)


if __name__ == '__main__':
	main()