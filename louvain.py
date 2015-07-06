import networkx as nx
import random

def girvan_graphs(zout) :
    """
    Create a graph of 128 vertices, 4 communities.
    Used as ground truth to compare to.
    Girvan newman, 2002. PNAS June, vol 99 n 12

    Community is node modulo 4
    """

    pout = float(zout)/96.
    pin = (16.-pout*96.)/31.
    graph = nx.Graph()
    graph.add_nodes_from(range(128))
    for x in graph.nodes() :
        for y in graph.nodes() :
            if x < y :
                val = random.random()
                if x % 4 == y % 4 :
                    #nodes belong to the same community
                    if val < pin :
                        graph.add_edge(x, y)

                else :
                    if val < pout :
                        graph.add_edge(x, y)
    return graph


class Community(object):
	"""Class representing communities
	   A community is here a disjoint set of nodes
	"""

	# TODO communities class or instance?
	def __init__(self, graph):
		self.communities = dict(zip(graph, range(graph.number_of_nodes())))
		self.m = float(graph.size())
		self.degrees_per_node = graph.degree(graph.nodes())
		self.degrees_per_community = self.degrees_per_node.copy()
		# number of links within the community
		self.internals = {i:i-i for i in range(graph.number_of_nodes())}

	
	def get_all_nodes_of_community(self, community):
		"""
		Parameter community = int representing number of community
		"""
		nodelist = [node for node, com in self.communities.items() if community == com]
		return nodelist

	def renumber_communites(slf, communities):
		vals = set(self.communities.values())
		mapping = dict(zip(vals,range(len(vals))))

		for key in communities.keys():
			self.communities[key] = mapping[self.communities[key]]

def second_pass(communities, graph):
	""" Nodes belonging to the same community as detected in pass 1 are merged into a single node.
		The new graph is build up from these so called "super nodes"
	"""
	aggreated_graph = nx.Graph()

	# The new graph consists of as many "supernodes" as there are communities
	aggreated_grap.add_nodes_from(set(communities.values()))
	# make edges between communites, if communites are the same self-loops are generated
	new_edges=[(communities[node1], communities[node2]) for node1, node2 in graph.edges()]
	aggreated_graph.add_nodes_from(new_edges)

	return aggreated_graph


def modularity(graph, communityRef):
	"""
	Compute the modularity of the graph using:

	.. math::

	M = \sum_{c=1}^{n_C} \left [ \frac{L_c}{L} - \left (  \frac{k_c}{2L} \right ) \right ]

	where L_c is the total number of links within the community C and k_c is the total degree of the noes in this community

	References:
	-----------
	.. [1] M. E. J. Newman and M. Girvan (2008).
		   Finding and evaluating community structure in networks.	
	 	   Phys. Rev. E 69, 026113
	"""
	q = 0.0
	m = float(communityRef.m)
	for community in set(communityRef.communities.values()):
		L_c = communityRef.internals.get(community, 0.0)
		K_c = communityRef.degrees_per_community.get(community, 0.0)
		q += (L_c / m) - ((K_c / (2.0*m))** 2.)
    	return q


def delta_q(graph, community, node):
	"""Compute the gain of modularity delta_Q if node would be inserted in community

	 .. math::

		delta_Q = \left [  \frac{\sum_{in}+2k_{i_in}}{2m} -  \left (\frac{\sum_{tot+k_i}}{2m}  \right )^{2}   \right ] -  \left [  \frac{\sum_{in}}{2m} -  \left (\frac{\sum_{tot}}{2m}  \right )^{2} - \left ( \frac{k_i}{2m}  \right )^{2} \right ]

	References
	----------
	..[1] Blondel, V. D., Guillaume, J.-L., Lambiotte, R., & Lefebvre, E. (2008). 
	      Fast unfolding of communities in large networks. 
	      Journal of Statistical Mechanics: Theory and Experiment, 10(1), 008. http://doi.org/10.1088/1742-5468/2008/10/P10008
	"""	
	nodes = Community.get_all_nodes_of_community(community)

	#number of links inside the community
	sigma_in  = sum(graph.degree((nodes)).values())
	#number of inks from node i to other nodes in community
	k_i_in    = len(set(graph.neighbours(node)).intersection(nodes))
	#number links incident to nodes in community
	sigma_tot = len(set(sum([graph.neighbors(node) for node in nodes], [])) )
	#number links incident to node i
	k_i       = len(graph.neighbours(node))
	m         = graph.size()

	term1 = (((sigma_in+k_i_in)/2.0*m)-((sigma_tot+k_i)/(2.0*m))**2.0)
  	term2 = sigma_in/(2.0*m)-(sigma_tot/(2.0*m)**2.0)-(k_i/(2.0*m)**2.0)
  	delta_q = term1 - term2
  	return delta_q


if __name__ == '__main__':
	graph = girvan_graphs(4)
	community = Community(graph)
	delta_q(graph, 2, 4)
 