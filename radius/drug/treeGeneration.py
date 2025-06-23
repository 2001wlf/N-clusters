#!/usr/bin/env python
# encoding: utf-8
# author:  ryan_wu
# email:   imitator_wu@outlook.com
# date:    2020-11-26 16:09:29
import math
import os
import copy
import time
import pickle
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
import glob
import numpy as np
import networkx as nx
import pandas as pd
import sklearn
from scipy.spatial import distance
from sklearn.cluster import KMeans

from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
from sklearn.preprocessing import MinMaxScaler

# from BDQLSH import projection_overlap
#from kmeans import KMEANS
from drug.lib.coding_tree import PartitionTree
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
from sklearn.metrics import normalized_mutual_info_score
import argparse


from drug.reg_kmeans import RegularizedKMeans

# from kmeans import KMEANS
color_map = {0:'Green', 1:'Purple', 2:'Blue', 3:'red'}


PWD = os.path.dirname(os.path.realpath(__file__))


def trans_to_adj(graph):
    graph.remove_edges_from(nx.selfloop_edges(graph))
    nodes = range(len(graph.nodes))
    return nx.to_numpy_array(graph, nodelist=nodes)
    # return np.array(nx.adjacency_matrix(graph).todense())


def trans_to_tree(adj, k=2):
    undirected_adj = np.array(adj)
    y = PartitionTree(adj_matrix=undirected_adj)
    x = y.build_encoding_tree(k)
    return y.tree_node


def update_depth(tree):
    # set leaf depth
    wait_update = [k for k, v in tree.items() if v.children is None]
    while wait_update:
        for nid in wait_update:
            node = tree[nid]
            if node.children is None:
                node.child_h = 0
            else:
                node.child_h = tree[list(node.children)[0]].child_h + 1
        wait_update = set([tree[nid].parent for nid in wait_update if tree[nid].parent])


def update_node(tree):
    update_depth(tree)
    d_id= [(v.child_h, v.ID) for k, v in tree.items()]
    d_id.sort()
    new_tree = {}
    for k, v in tree.items():
        n = copy.deepcopy(v)
        n.ID = d_id.index((n.child_h, n.ID))
        if n.parent is not None:
            n.parent = d_id.index((n.child_h+1, n.parent))
        if n.children is not None:
            n.children = [d_id.index((n.child_h-1, c)) for c in n.children]
        n = n.__dict__
        n['depth'] = n['child_h']
        new_tree[n['ID']] = n
    return new_tree


step = 0


def pool_trans(input_):
    global step
    step += 1
    g, tree_depth = input_
    adj_mat = trans_to_adj(g['G'])
    assert np.all(adj_mat == adj_mat.T)   # undirected
    if not nx.is_connected(g['G']):
        # print(step, 'not connected')
        # return None   # FIXME
        pass
    tree = trans_to_tree(adj_mat, tree_depth)
    g['tree'] = update_node(tree)
    # display_graph(g['G'])
    # display_tree(g['tree'])
    return g


def display_tree(tree):
    G = nx.Graph()
    for k, v in tree.items():
        G.add_node(k)
        if v['parent'] is not None:
            G.add_edge(v['parent'], k)
    nx.draw(G, with_labels=True, pos=nx.spring_layout(G))
    plt.show()
    plt.clf()


def display_graph(g):
    nx.draw(g, with_labels=True, pos=nx.spring_layout(g))
    plt.show()
    plt.clf()


def struct_tree(dataset, tree_depth):
    if not os.path.exists('trees'):
        os.makedirs('trees')
    if os.path.exists('trees/%s_%s.pickle' % (dataset, tree_depth)):
        print('trees already exist', dataset, tree_depth)
    with open('graphs/%s.pickle' % dataset, 'rb') as fp:
        g_list = pickle.load(fp)
    n = len(g_list)
    print('total graphs', n)
    g_list = [pool_trans((g, tree_depth)) for g in g_list]
    g_list = list(filter(lambda g: g is not None, g_list))
    # assert len(g_list) == n
    print('valid graphs', len(g_list))
    with open('trees/%s_%s.pickle' % (dataset, tree_depth), 'wb') as fp:
        pickle.dump(g_list, fp)


def entropy(VOL, node_dict):
    ent = []
    for node_id,node in node_dict.items():
        if node['parent'] is not None:
            node_p = node_dict[node['parent']]
            node_vol = node['vol']
            node_g = node['g']
            node_p_vol = node_p['vol']
            ent.append(- (node_g / VOL) * math.log2(node_vol / node_p_vol))
    if ent == []:
        return np.array([0])
    return np.array(ent)


def all_entory(n, k, VOL, tree):
    ent = np.zeros((n,k))
    for id, tree_node in list(tree.items())[n:]:
        childrens = tree_node["children"]
        node_p_vol = tree_node['vol']
        h = tree_node["child_h"]
        for children in childrens:
            node = tree[children]
            node_vol = node['vol']
            node_g = node['g']
            e = - (node_g / VOL) * math.log2(node_vol / node_p_vol)
            # normal
            # ent[children["partition"]][h-1] = - (node_g / VOL) * math.log2(node_vol / node_p_vol)
            # bin modal
            temp = node["partition"]
            while (temp):
                lowbit = temp & (-temp)
                j = int(math.log2(lowbit))
                ent[j][h-1] = e
                temp ^= lowbit
    return ent


def find_min_gap(np_nums):
    nums = np_nums.reshape(1,-1)[0]
    nums = np.sort(nums,kind='heapsort')
    min = np.inf
    for i in range(len(nums)-1):
        current = nums[i+1] - nums[i]
        if current == 0:
            continue
        if current < min:
            min = current
    return min


def make_uniform(start, end, length, np_nums):
    s = np.linspace(start, end, length).reshape(-1,1)
    dis_map = distance.cdist(np_nums.reshape(-1,1), s)
    assignment = np.argmin(dis_map,axis=1)
    new_nums = s[assignment]
    return new_nums


def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(np.max(y_pred), np.max(y_true)) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = np.array(linear_assignment(np.max(w) - w)).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def bulit_knn_graph(data, ne, threshold = np.inf):
    kd_tree = KDTree(data, leaf_size=20)
    try:
        dist, inds = kd_tree.query(data, k=ne)
    except:
        dist, inds = kd_tree.query(data, k=len(data))
    G = nx.Graph()
    for m in range(len(inds)):
        ind = inds[m]
        i = ind[0]
        for n in range(len(ind[1:])):
            if n:
                if dist[m][n + 1] > threshold:
                    continue
            j = ind[n + 1]
            G.add_edge(i, j, weight=dist[m][n + 1])
            # G.add_edge(i, j, weight=dist[m][n + 1])
    # for i in G.nodes:
    #     node_i_key = []
    #     node_i_value = []
    #     for j in G[i]:
    #         node_i_key.append(j)  
    #         node_i_value.append(G[i][j]['weight'])
    #     node_i_key = np.array(node_i_key)
    #     node_i_value = np.array(node_i_value)
    #     forbidden = np.mean(node_i_value)+2*np.std(node_i_value)
    #     remove_edge = node_i_key[np.argwhere(node_i_value>forbidden)]
    #     for j in remove_edge:
    #         G.remove_edge(i,j[0])
    return G

def bulit_k_farthest_graph(data, ne, threshold = np.inf):
    dis_map = distance.cdist(data,data,metric='euclidean')
    inds = np.argpartition(dis_map, -ne, axis=1)[:,-ne:]
    G = nx.Graph()
    for m in range(len(inds)):
        ind = inds[m]
        for n in ind:
            if dis_map[m][n + 1] < threshold:
                continue
            G.add_edge(m, n, weight=dis_map[m][n])
    return G

def bulit_knn_adjm(data, ne, threshold=np.inf):
    # plt.figure(figsize=(4,4))
    # plt.xticks([])
    # plt.yticks([])
    # plt.scatter(data[:,0],data[:,1],c=[color_map[i] for i in label])
    adj_mat = np.zeros((len(data),len(data)))
    kd_tree = KDTree(data, leaf_size=20)
    try:
        dist, inds = kd_tree.query(data, k=ne+1)
    except:
        dist, inds = kd_tree.query(data, k=len(data))
    for m in range(len(inds)):
        ind = inds[m]
        i = ind[0]
        for n in range(len(ind[1:])):
            if n:
                if dist[m][n + 1] > threshold:
                    continue
            j = ind[n + 1]
            # edge = data[[i,j]]
            # plt.plot(edge[:,0],edge[:,1],c='black')
            adj_mat[i][j] = dist[m][n + 1]
            adj_mat[j][i] = dist[m][n + 1]
    # plt.savefig("2.png")
    return adj_mat

def bulit_k_farthest_adjm(data, ne, threshold = 0):
    dis_map = distance.cdist(data,data,metric='euclidean')
    inds = np.argpartition(dis_map, -ne, axis=1)[:,-ne:]
    adj_mat = np.zeros((len(data), len(data)))
    for m in range(len(inds)):
        ind = inds[m]
        for n in ind:
            if dis_map[m][n] < threshold:
                continue
            adj_mat[m][n] = dis_map[m][n]
            adj_mat[n][m] = dis_map[m][n]
    return adj_mat

def up_dimension(data, ne, tree_depth, gap, opt):
    print(ne,tree_depth,gap)
    adj_mat = bulit_knn_adjm(data, ne, gap * math.sqrt(opt / len(data)))
    tree = trans_to_tree(adj_mat, tree_depth)
    tree = update_node(tree)
    # print(tree)
    data_entory = all_entory(len(data), tree_depth, np.sum(adj_mat), tree)
    data_ent = np.sum(data_entory, axis=1).reshape(-1, 1)
    # data_ent_ = MinMaxScaler().fit_transform(data_ent) + 1e-15
    # data_ = np.hstack((data, np.log(data_ent_)))
    data_ = np.hstack((data, data_ent))
    data = MinMaxScaler().fit_transform(data_)
    # kmeans = KMeans(init='k-means++', n_clusters=k).fit(data)
    # print('acc=', round(acc(label, kmeans.labels_), 4))
    # print('nmi=', round(normalized_mutual_info_score(label, kmeans.labels_), 4))
    # opt = kmeans.inertia_
    # dis_map = np.sort(distance.cdist(data, data, 'euclidean').reshape(-1))
    # max_dis = dis_map[-1]
    # min_dis = dis_map[len(data)]
    # print('d=', data.shape[1], ',\taspect_ratio=', max_dis / min_dis)
    return data

def up_dimension_log(data, ne, tree_depth, gap, opt):
    print(ne,tree_depth,gap)
    adj_mat = bulit_knn_adjm(data, ne, gap * math.sqrt(opt / len(data)))
    tree = trans_to_tree(adj_mat, tree_depth)
    tree = update_node(tree)
    data_entory = all_entory(len(data), tree_depth, np.sum(adj_mat), tree)
    data_ent = np.sum(data_entory, axis=1).reshape(-1, 1)
    data_ent_ = MinMaxScaler().fit_transform(data_ent) + 1e-15
    data_ = np.hstack((data, np.log(data_ent_)))
    # data_ = np.hstack((data, data_ent))
    data = MinMaxScaler().fit_transform(data_)
    kmeans = KMeans(init='k-means++', n_clusters=k).fit(data)
    print('acc=', round(acc(label, kmeans.labels_), 4))
    print('nmi=', round(normalized_mutual_info_score(label, kmeans.labels_), 4))
    opt = kmeans.inertia_
    dis_map = np.sort(distance.cdist(data, data, 'euclidean').reshape(-1))
    max_dis = dis_map[-1]
    min_dis = dis_map[len(data)]
    print('d=', data.shape[1], ',\taspect_ratio=', max_dis / min_dis)
    return data, opt


