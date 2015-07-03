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
	def __init__(self, graph):
		# super(Community, self).__init__()
		communities = dict(zip(graph, range(graph.number_of_nodes())))
	
	@classmethod	
	def get_all_nodes_of_community(cls, community):
		nodelist = [node for node, com in cls.communities.items() if community == com]
		return nodelist

	def renumber_communites(self, community):
		vals = set(self.community.values())
		mapping = dict(zip(vals,range(len(vals))))

		for key in community.keys():
			self.community[key] = mapping[self.community[key]]



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
 