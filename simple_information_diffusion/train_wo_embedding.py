import numpy as np
from env.graph_env_wo_embedding import GraphEnv
# def create_graph(filename=''):
# 	if not filename:
# 		return
# 	graph = nx.Graph()
# 	with open(filename, 'r') as f:
# 		line = f.readline()
# 		node1, node2 = line.split()
# 		graph.add_edge(node1, node2)
	
# 	return graph

# num_nodes = 1005
# policy = np.zeros((num_nodes, num_nodes))
# q_values = np.zeros((num_nodes,num_nodes))
# return_q = np.zeros((num_nodes,num_nodes))
# counts = np.zeros((num_nodes,num_nodes))
# policy = np.zeros(num_nodes)
# g = GraphEnv(num_nodes, 'data/email_weight.edgelist')
# adj_list = g.get_adj_list()
# num_runs = 20000

# for node in range(num_nodes):
# 	policy[node] = np.random.choice(adj_list[node])

# print (policy)

def monte_carlo(policy, check_count):
	num_nodes = 1005
	policy = np.zeros((num_nodes, num_nodes))
	q_values = np.zeros((num_nodes,num_nodes))
	return_q = np.zeros((num_nodes,num_nodes))
	counts = np.zeros((num_nodes,num_nodes))
	policy = np.zeros(num_nodes)
	g = GraphEnv(num_nodes, 'data/email_weight.edgelist')
	adj_list = g.get_adj_list()
	num_runs = 20000

	for node in range(num_nodes):
		policy[node] = np.random.choice(adj_list[node])
		
	gamma = 0.7
	count = 0
	for run in range(num_runs):
		# print ("Run is : " + str(run))
		g.reset()
		new_policy = policy.copy()
		walk = g.generate_walk(policy)
		if not walk:
			continue
		# print (walk)
		# [(s1,a1,r1), (s2,a2,r2)..]
		total_return = 0
		# total_t = len(walk)
		walk.reverse()
		
		for a_s in walk:
			total_return = gamma * total_return + a_s[2]
			return_q[a_s[0]][a_s[1]] += total_return
			index = np.argmax(return_q[a_s[0]])
			# print (maximum)
			# print (np.where(return_q[a_s[0]] == maximum))
			new_policy[a_s[0]] = index
			# policy[a_s[0]] = return_q[a_s[0]].index(max(return_q[a_s[0]]))

		# for index, value in enumerate(policy):
		# 	if value != new_policy[index]:
				# print ("not equal " + str(index))
		if not any(policy[i]!= new_policy[i] for i in range(num_nodes)):
			count += 1
			if count == check_count:
				print (run)
				break
		else:
			count =0
		policy = new_policy

	return policy

# monte_carlo(policy, 50)
		