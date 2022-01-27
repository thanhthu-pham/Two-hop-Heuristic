import networkx as nx
import csv
import pandas as pd
import numpy as np
import collections
import time
import random
from networkx import read_adjlist
from collections import Counter
from numpy import arange

p = 0.09 #activation probability to be tuned in

#Read the data file
FileName = "/Users/thupham/Downloads/CA-HepTh.txt"

Graphtype = nx.DiGraph()

G = nx.read_edgelist(
    FileName,
    create_using=Graphtype, nodetype=int)

#Overview of the network size
nodes = G.number_of_nodes()
edges = G.number_of_edges()

#Pre-processing data
def node_to_index(node):
    list_of_index = []
    for u in node:
        a = list(G.nodes()).index(u)
        print(a)
        list_of_index.append(a)
    return list_of_index


def index_to_node(index):
    list_of_node = []
    for u in index:
        a = list(G.nodes())[u]
        list_of_node.append(a)
    return list_of_node


def ingoing_neighbor(G, node):
    return list(G.predecessors(node))


def outgoing_neighbor(G, node):
    return list(G.successors(node))


list_all_nodes = list(G.nodes())
nodes_max = len(list_all_nodes)

outgoing_neighbors_array = np.zeros((nodes_max, nodes_max), dtype=np.intc)
ingoing_neighbors_array = np.zeros((nodes_max, nodes_max), dtype=np.intc)

list_outgoing_neighbors = [outgoing_neighbor(G, node) for node in list_all_nodes]
list_ingoing_neighbors = [ingoing_neighbor(G, node) for node in list_all_nodes]
list_outgoing_neighbors_byIndex = [[list_all_nodes.index(node) for node in i] for i in list_outgoing_neighbors]
list_ingoing_neighbors_byIndex = [[list_all_nodes.index(node) for node in i] for i in list_ingoing_neighbors]

for node in range(nodes_max):
    outgoing_neighbors_array[node][list_outgoing_neighbors_byIndex[node]] = 1
    ingoing_neighbors_array[node][list_ingoing_neighbors_byIndex[node]] = 1

#sum_array = np.sum(ingoing_neighbors_array, axis=1)  # if np.all(np.sum(ingoing_neighbors_array, axis = 1)) > 0 else 0
#prob_array = np.zeros(nodes_max)
#print("shape of prob array", prob_array.shape)
prob_array = np.full(nodes_max, p) #fixed
#print("shape of prob array", prob_array.shape)
#for i in range(nodes_max):
    #if sum_array[i] > 0:
        #prob_array[i] = 1 / sum_array[i]

#Indepedent Cascade Model with fixed probability
def IC_model(seed_node):
    total_spread = len(seed_node)
    status_array = np.zeros(nodes_max)
    status_array[seed_node] = 1

    new_active = seed_node
    while len(new_active) > 0:

        activated_nodes_store = []  # create an empty list

        for node in new_active:

            x = outgoing_neighbors_array[node]

            inactive_neighbors = np.where(x - status_array > 0)

            random = np.random.uniform(0, 1, size=inactive_neighbors[0].size)


            prob = prob_array[inactive_neighbors]


            check = random < prob



            activated_nodes = inactive_neighbors[0][check]


            activated_nodes_store = list(activated_nodes_store)

            for i in activated_nodes:
                activated_nodes_store.append(i)  # This is not correct to append two array I think.



            activated_nodes_store = np.array(activated_nodes_store)


            total_spread += activated_nodes.size

            status_array[activated_nodes] = 1
        new_active = activated_nodes_store

    return total_spread

#Spread Computation
def IC_model_expected_value(n, seed_nodes):
    # if seed_nodes is given as the numbering in the graph G, we must convert seed_node to the corresponding index
    # seed_nodes_idx = list_all_nodes.index(seed_nodes)
    # store = [np.intc(x) for x in range(0)]
    expected_spread = 0.0
    for i in range(n):
        a = IC_model(seed_nodes)
        expected_spread += a
        # print(a)
    expected_spread /= n
    # print("expected value", expected_spread)
    return expected_spread


#Def first search algorithm
def def_first_search(node, G, seed_nodes):
    #print("f")
    G2 = nx.DiGraph()
    G2.add_node(node)
    for m in outgoing_neighbor(G, node):
        G2.add_edge(node,m)
    for u in seed_nodes:
        if G2.has_node(u) == True:
            G2.remove_node(u)
            #print("G2 remove node", u, "because it's in the seed nodes")
    N1 = [i for i in list(G2.nodes()) if i not in [node]]
    N2 = []
    for v in N1:
        a = [i for i in outgoing_neighbor(G,v) if i not in seed_nodes and i not in N1 + [node] ]
        N2+=a
        for u in a:
            G2.add_edge(v,u)
    return G2

def expected_value(node, G2, tv, p):
    probability_list, node_step2 = [1],[]
    for u in outgoing_neighbor(G2,node):
        probability_list.append(p)
        for m in outgoing_neighbor(G2,u):
            node_step2.append(m)
    for m in list(set(node_step2)):
        p2 = 1 - (1 - p*p)**len(ingoing_neighbor(G2,m))
        probability_list.append(p2)
        #print("Node", m, "has expected value", p2)
    unactive_prob = (1-p)**tv
    #print("probability list", probability_list)
    final = unactive_prob*np.sum(probability_list)
    return final


def Expected_two_step(G, k, p):
    total_nodes = list(G.nodes()) #list of all nodes
    seed_nodes, degree_list = [], []
    tv_list = [0] * len(total_nodes)
    for node in total_nodes:
        G2 = def_first_search(node, G, seed_nodes)
        tv = tv_list[total_nodes.index(node)] #number of neighbors that node have which are
        a = expected_value(node, G2, tv, p) #tv=0 here
        degree_list.append(a)
    index = degree_list.index(max(degree_list))
    first_node = total_nodes[index]
    seed_nodes.append(first_node)
    new_active = seed_nodes[:]
    while len(seed_nodes) < k:
        for u in new_active:
            degree_list.remove(degree_list[total_nodes.index(u)])
            tv_list.remove(tv_list[total_nodes.index(u)])
            total_nodes.remove(u)
            neighbors = [i for i in outgoing_neighbor(G,u) if i not in seed_nodes]
            for v in neighbors:
                tv_list[total_nodes.index(v)]+=1 #update tv here
        unactive_nodes = [i for i in total_nodes if i not in seed_nodes]
        for m in unactive_nodes:
            G3 = def_first_search(m, G, seed_nodes)
            tv = tv_list[total_nodes.index(m)]
            b = expected_value(m, G3, tv, p)
            child_nodes = [t for t in list(G3.nodes()) if t not in [m]]
            for t in child_nodes:
                tv_2 = tv_list[total_nodes.index(t)]
                b = b - (1-p)**tv*(1-(1-p)**tv_2)
            degree_list[total_nodes.index(m)] = b
        selected_node = total_nodes[degree_list.index(max(degree_list))]
        new_active = [selected_node]
        seed_nodes.append(selected_node)
    #print("Seed nodes", seed_nodes)
    return seed_nodes

def Expected_two_step_ver2(G, k, p, alpha):
    total_nodes = list(G.nodes()) #list of all nodes
    seed_nodes, degree_list = [], []
    tv_list = [0] * len(total_nodes)
    for node in total_nodes:
        G2 = def_first_search(node, G, seed_nodes)
        tv = tv_list[total_nodes.index(node)] #number of neighbors that node have which are
        a = expected_value(node, G2, tv, p) #tv=0 here
        degree_list.append(a)
    index = degree_list.index(max(degree_list))
    first_node = total_nodes[index]
    seed_nodes.append(first_node)
    new_active = seed_nodes[:]
    while len(seed_nodes) < k:
        for u in new_active:
            degree_list.remove(degree_list[total_nodes.index(u)])
            tv_list.remove(tv_list[total_nodes.index(u)])
            total_nodes.remove(u)
            neighbors = [i for i in outgoing_neighbor(G,u) if i not in seed_nodes]
            for v in neighbors:
                tv_list[total_nodes.index(v)]+=1
        unactive_nodes = [i for i in total_nodes if i not in seed_nodes]
        for m in unactive_nodes:
            G3 = def_first_search(m, G, seed_nodes)
            tv = tv_list[total_nodes.index(m)]
            b = expected_value(m, G3, tv, p)
            child_nodes = [t for t in list(G3.nodes()) if t not in [m]]
            total_neighbors = []
            for u in seed_nodes:
                neighbors = [i for i in outgoing_neighbor(G,u) if i not in seed_nodes]
                total_neighbors+=neighbors
            update_node = [k for k in child_nodes if k in total_neighbors]
            for t in update_node:
                tv_2 = tv_list[total_nodes.index(t)]
                b -= alpha * ((1-p)**tv * (1 - (1 - p) ** tv_2))
            degree_list[total_nodes.index(m)] = b
        selected_node = total_nodes[degree_list.index(max(degree_list))]
        new_active = [selected_node]
        seed_nodes.append(selected_node)
    print("Seed nodes", seed_nodes)
    return seed_nodes



def Expected_two_step_checking(G, k, p):
    total_nodes = list(G.nodes()) #list of all nodes
    seed_nodes, degree_list = [], []
    tv_list = [0] * len(total_nodes)
    for node in total_nodes:
        G2 = def_first_search(node, G, seed_nodes)
        tv = tv_list[total_nodes.index(node)] #number of neighbors that node have which are
        a = expected_value(node, G2, tv, p) #tv=0 here
        degree_list.append(a)
    index = degree_list.index(max(degree_list))
    first_node = total_nodes[index]
    seed_nodes.append(first_node)
    new_active = seed_nodes[:]
    while len(seed_nodes) < k:
        for u in new_active:
            degree_list.remove(degree_list[total_nodes.index(u)])
            #print("degree after remove", degree_list)
            tv_list.remove(tv_list[total_nodes.index(u)])
            #print("tv list after remove", tv_list)
            total_nodes.remove(u)
            #print("total node after remove", total_nodes)
            neighbors = [i for i in outgoing_neighbor(G,u) if i not in seed_nodes]
            #print("Node", u, "in seed node has outgoing neighbors", neighbors)
            for v in neighbors:
                tv_list[total_nodes.index(v)]+=1 #update tv here
        unactive_nodes = [i for i in total_nodes if i not in seed_nodes]
        #print("Unactive nodes here", unactive_nodes)
        for m in unactive_nodes:
            G3 = def_first_search(m, G, seed_nodes)
            #print("node in subgraph of node", m, "is", list(G3.nodes()))
            tv = tv_list[total_nodes.index(m)]
            #print("Node", m, "in unactive nodes has tv value", tv)
            b = expected_value(m, G3, tv, p)
            #print("Node", m, "has expected influence is", b)
            degree_list[total_nodes.index(m)] = b
            #print("Degree_list now", degree_list)
        selected_node = total_nodes[degree_list.index(max(degree_list))]
        #print("selected nodes", selected_node)
        new_active = [selected_node]
        seed_nodes.append(selected_node)
    return seed_nodes

#Other algorithms to compared to

#Degree Discount
def Degree_discountIC(G, k, p):
    total_nodes = list(G.nodes()) #list of all nodes
    seed_nodes, degree_list, ddv_list = [], [], []
    tv_list = [0] * len(total_nodes)
    for node in total_nodes:
        a = len(outgoing_neighbor(G,node))
        degree_list.append(a)
        ddv_list.append(a)
    index = degree_list.index(max(degree_list))
    first_node = total_nodes[index]
    seed_nodes.append(first_node)
    new_active = seed_nodes[:]
    #print("new active", new_active)
    print("Node", first_node, "was chosen in iteration 1")
    while len(seed_nodes) < k:
        for u in new_active:
            neighbors = [i for i in outgoing_neighbor(G,u) if i not in seed_nodes]
            for v in neighbors:
                tv_list[total_nodes.index(v)]+=1
                a = len(outgoing_neighbor(G,v))
                b = 2*tv_list[total_nodes.index(v)]
                c = - (len(outgoing_neighbor(G,v)) - tv_list[total_nodes.index(v)])*tv_list[total_nodes.index(v)]*p
                ddv = a - b + c # Compute the discount
                ddv_list[total_nodes.index(v)] = ddv
        ddv_list.remove(ddv_list[total_nodes.index(u)])
        tv_list.remove(tv_list[total_nodes.index(u)])
        total_nodes.remove(u)
        selected_node = total_nodes[ddv_list.index(max(ddv_list))]
        new_active = [selected_node]
        seed_nodes.append(selected_node)
    print("Seed node", seed_nodes)
    return seed_nodes



def Degree_discount(G, k):
    total_nodes = list(G.nodes()) #list of all nodes
    seed_nodes, degree_list = [], [] #degree list and total_list has the same index
    #tv_list = [0] * len(total_nodes)
    for node in total_nodes:
        a = len(outgoing_neighbor(G,node))
        degree_list.append(a)
    index = degree_list.index(max(degree_list))
    first_node = total_nodes[index]
    seed_nodes.append(first_node)
    new_active = seed_nodes[:]
    #print("new active", new_active)
    #print("Node", first_node, "was chosen in iteration 1")
    while len(seed_nodes) < k:
        for u in new_active:
            neighbors = [i for i in ingoing_neighbor(G,u) + outgoing_neighbor(G,u) if i not in seed_nodes]
            for v in neighbors:
                degree_list[total_nodes.index(v)]-=1
                #print("Node", v,"has degree after discount", degree_list[total_nodes.index(v)])
        degree_list.remove(degree_list[total_nodes.index(u)])
        total_nodes.remove(u)
        selected_node = total_nodes[degree_list.index(max(degree_list))]
        new_active = [selected_node]
        seed_nodes.append(selected_node)
        #print("node", selected_node, "was chosen in iteration", len(seed_nodes), "with degree", max(degree_list))
    #print("Seed node", seed_nodes)
    print()
    return seed_nodes

#High Degree
def High_degree(k):
    degree = [len(outgoing_neighbor(G, i)) for i in list(G.nodes)]
    degree_list = sorted(zip(list(G.nodes()), degree), key=lambda x: x[1], reverse=True)
    #print(degree_list)
    seed_nodes = []
    for a in range(k):
        selected_nodes = degree_list[a][0]
        seed_nodes.append(selected_nodes)
    #print("Seed set",seed_nodes )
    return seed_nodes

#Run the algorithms
k=30
n=10000

print("This is result for p", p)

t0 = time.perf_counter()
seed_nodes = High_degree(k)
elapsed_time = time.perf_counter() - t0
print("Elapsed time", k, "is", elapsed_time, "of High degree")
seed_nodes = node_to_index(High_degree(k))
elapsed_time = time.perf_counter()
print("Elapsed time for High degree", elapsed_time)
obj_value = IC_model_expected_value(10000, seed_nodes)
print("Objective value for k", k, "is", obj_value, "of High degree")
#print("This is result for p",p)

