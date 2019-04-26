from train_wo_embedding import monte_carlo
from train_sarsa import sarsa
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from env.graph_env_wo_embedding import GraphEnv
import networkx as nx
from numpy import array

num_nodes = 1005
policy = np.zeros(num_nodes)
g = GraphEnv(num_nodes, 'data/email_weight.edgelist')
adj_list = g.get_adj_list()

for node in range(num_nodes):
    policy[node] = np.random.choice(adj_list[node])


# check_counts = [10, 20, 50, 100]
# sarsa_runs = []
# mc_runs = []

# for check_count in check_counts:
#   calls_run_sarsa = []
#   calls_run_mc = []
#   for call in range(10):
#       calls_run_sarsa.append(sarsa(policy, check_count))
#       calls_run_mc.append(monte_carlo(policy, check_count))
#   sarsa_runs.append(sum(calls_run_sarsa)/len(calls_run_sarsa))
#   mc_runs.append(sum(calls_run_mc)/len(calls_run_mc))

# plt.plot(sarsa_runs, label='sarsa')
# plt.plot(mc_runs, label='monte carlo')
# plt.xlabel('c')
# plt.ylabel('No of runs')
# plt.legend()
# plt.show()

class Results(object):
    
    def get_active_and_informed_nodes(self, action):
        adj_nodes = self.adj_list[action]
        active_nodes = self._get_active_list(action, adj_nodes) + [action]
        informed_nodes = [x for x in adj_nodes if x not in active_nodes] + [action]

        return active_nodes, informed_nodes

    def calculate_reward(self, active_nodes, inf_nodes):
        lambd = 1
        
        old_active_nodes = np.where(self.active_nodes == 1)[0]
        new_actives = [item for item in active_nodes if item not in old_active_nodes]

        old_informed_nodes = np.where(self.informed_nodes == 1)[0]
        new_informed = [item for item in inf_nodes if item not in old_informed_nodes]

        reward = len(new_actives) + lambd * len(new_informed)
        
        return reward

    def _add_edges(self, dataset):
        with open(dataset, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                row = line.split(' ')
                # reader = csv.reader(f)
                self.graph.add_edge(int(row[0]), int(row[1]))
                self.activating_prob[int(row[0])][int(row[1])] = float(row[2])
                self.activating_prob[int(row[1])][int(row[0])] = float(row[2])

    def reset(self):
        nodes = [x for x in range(0,self.num_nodes)]
        start_node = np.random.choice(nodes)
        self.active_nodes = np.zeros(self.num_nodes)
        self.informed_nodes = np.zeros(self.num_nodes)
        self.visited = np.zeros(self.num_nodes)
        
        return None

    def _get_active_list(self, action, adj_nodes):
        active_nodes = []
        for node in adj_nodes:
            # print (node)
            arb_num = np.random.uniform(0,1,1)[0]
            # print (arb_num)
            # print (action)
            # print (node)
            if arb_num <= self.activating_prob[action][node]:
                active_nodes.append(node)

        return active_nodes

    def policy_test(self, method, policy):
        if method == 'sarsa':
            policy = sarsa(policy, 20)
        else:
            policy = monte_carlo(policy, 20)

        self.graph = nx.Graph()
        self.num_nodes = num_nodes
        nodes = [x for x in range(0,num_nodes)]
        # print (nodes)
        self.graph.add_nodes_from(nodes)
        self.visited = np.zeros(num_nodes)
        self.active_nodes = np.zeros(num_nodes)
        self.informed_nodes = np.zeros(num_nodes)
        self.stopping_prob = np.random.rand(num_nodes) * 0.2
        self.activating_prob = np.zeros((num_nodes, num_nodes))
        self._add_edges('data/email_weight.edgelist')
        self.adj_matrix = nx.adjacency_matrix(self.graph).todense()
        self.adj_list = {}
        for node in range(num_nodes):
            adj_matrix_node = array(self.adj_matrix[node])[0]
            adj_nodes = np.where(adj_matrix_node == 1)[0]
            self.adj_list[node] = adj_nodes
        rewards_list = []

        node = 400
        while True:
            # reward = self.calculate_reward(node)
            node = int(policy[node])
            print (node)
            if self.visited[node] == 1:
                break
            self.visited[node] = 1
            
            active_nodes, informed_nodes = self.get_active_and_informed_nodes(node)
            reward = self.calculate_reward(active_nodes, informed_nodes)
            for node in active_nodes:
                self.active_nodes[node] = 1
                self.informed_nodes[node] = 1
            for node in informed_nodes:
                self.informed_nodes[node] = 1
            if len(rewards_list) == 0:
                rewards_list.append(reward)
            else:
                rewards_list.append(rewards_list[-1] + reward)
        return rewards_list

r = Results()
sarsa_rewards = r.policy_test('sarsa', policy)
r.reset()
print ('Starting MC')
mc_rewards = r.policy_test('mc', policy)
plt.plot(sarsa_rewards, label='sarsa')
plt.plot(mc_rewards, label='monte carlo')
plt.xlabel('No of runs')
plt.ylabel('Reward')
plt.legend()
plt.title('Total return, lambda = 1.0')
plt.show()