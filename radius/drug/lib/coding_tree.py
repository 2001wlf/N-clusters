import math
import heapq
import random
import time
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from multiprocessing import Pool, Manager

import networkx as nx
# import numba as nb
import numpy as np
import copy
from pdb import set_trace

import pandas as pd
import tqdm
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.neighbors import KDTree

def get_id():
    i = 0
    while True:
        yield i
        i += 1


# normal
# def graph_parse(adj_matrix):
#     g_num_nodes = adj_matrix.shape[0]
#     adj_table = {}
#     VOL = 0
#     node_vol = []
#     for i in range(g_num_nodes):
#         n_v = 0
#         adj=set()
#         for j in range(g_num_nodes):
#             if adj_matrix[i,j] != 0:
#                 n_v += adj_matrix[i,j]
#                 VOL += adj_matrix[i,j]
#                 adj.add(j)
#         adj_table[i] = adj
#         node_vol.append(n_v)
#     return g_num_nodes,VOL,node_vol,adj_table

# bin model
def graph_parse(adj_matrix):
    g_num_nodes = adj_matrix.shape[0]
    adj_table = {}
    VOL = 0
    # node_avg_vol = []
    node_vol = []
    node_g = []
    for i in range(g_num_nodes):
        adj = 0
        adj_indexes = np.where(adj_matrix[i] != 0)[0]
        # adj_matrix_res = adj_matrix[i][adj_indexes]
        # tmp_sum = np.sum(adj_matrix_res)
        tmp_sum = np.sum(adj_matrix[i])
        VOL += tmp_sum
        n_v = tmp_sum
        for j in adj_indexes:
            adj |= 1 << int(j)
        adj_table[i] = adj
        # adj = 0
        # for j in range(g_num_nodes):
        #     if adj_matrix[i,j] != 0:
        #         n_v += adj_matrix[i,j]
        #         VOL += adj_matrix[i,j]
        #         adj |= 1 << j
        # adj_table[i] = adj
        node_vol.append(n_v)
        node_g.append(n_v-adj_matrix[i][i])
    return g_num_nodes, VOL, node_vol, adj_table, node_g


    #     node_avg_vol.append(n_v / bin(adj).count("1"))
    #
    # return g_num_nodes, VOL, node_vol, adj_table, node_avg_vol


# normal
# @nb.jit(nopython=True)
# def cut_volume(adj_matrix, p1, p2):
#     c12 = 0
#     for i in range(len(p1)):
#         for j in range(len(p2)):
#             c = adj_matrix[p1[i], p2[j]]
#             if c != 0:
#                 c12 += c
#     return c12

# bin_modal
# def cut_volume(node_avg_vol, adj_table, p1, p2):
#     c12 = 0
#     if bin(p1).count("1") > bin(p2).count("1"):
#         q1, q2 = p2, p1
#     else:
#         q1, q2 = p1, p2
#     while (q1):
#         lowbit = q1 & (-q1)
#         j = int(math.log2(lowbit))
#         adj = adj_table[j] & p2
#         c12 += node_avg_vol[j] * bin(adj).count("1")
#         q1 ^= lowbit
#     return c12

def cut_volume(adj_matrix, adj_table, p1, p2):
    c12 = 0
    if bin(p1).count("1") > bin(p2).count("1"):
        q1, q2 = p2, p1
    else:
        q1, q2 = p1, p2
    while (q1):
        lowbit = q1 & (-q1)
        j = int(math.log2(lowbit))
        adj = adj_table[j] & q2
        while (adj):
            lowbit_ = adj & (-adj)
            i = int(math.log2(lowbit_))
            c12 += adj_matrix[j, i]
            adj ^= lowbit_
        q1 ^= lowbit
    return c12


def LayerFirst(node_dict, start_id):
    stack = [start_id]
    while len(stack) != 0:
        node_id = stack.pop(0)
        yield node_id
        if node_dict[node_id].children:
            for c_id in node_dict[node_id].children:
                stack.append(c_id)


# normal
# def merge(new_ID, id1, id2, cut_v, node_dict):
#     new_partition = node_dict[id1].partition + node_dict[id2].partition
#     v = node_dict[id1].vol + node_dict[id2].vol
#     g = node_dict[id1].g + node_dict[id2].g - 2 * cut_v
#     child_h = max(node_dict[id1].child_h,node_dict[id2].child_h) + 1
#     new_node = PartitionTreeNode(ID=new_ID,partition=new_partition,children={id1,id2},
#                                  g=g, vol=v,child_h= child_h,child_cut = cut_v)
#     node_dict[id1].parent = new_ID
#     node_dict[id2].parent = new_ID
#     node_dict[new_ID] = new_node


# bin modal
def merge(new_ID, id1, id2, cut_v, node_dict):
    new_partition = node_dict[id1].partition | node_dict[id2].partition
    v = node_dict[id1].vol + node_dict[id2].vol
    g = node_dict[id1].g + node_dict[id2].g - 2 * cut_v
    child_h = max(node_dict[id1].child_h, node_dict[id2].child_h) + 1
    new_node = PartitionTreeNode(ID=new_ID, partition=new_partition, children={id1, id2},
                                 g=g, vol=v, child_h=child_h, child_cut=cut_v)
    node_dict[id1].parent = new_ID
    node_dict[id2].parent = new_ID
    node_dict[new_ID] = new_node


def compressNode(node_dict, node_id, parent_id):
    p_child_h = node_dict[parent_id].child_h
    node_children = node_dict[node_id].children
    node_dict[parent_id].child_cut += node_dict[node_id].child_cut
    node_dict[parent_id].children.remove(node_id)
    node_dict[parent_id].children = node_dict[parent_id].children.union(node_children)
    for c in node_children:
        node_dict[c].parent = parent_id
    com_node_child_h = node_dict[node_id].child_h
    node_dict.pop(node_id)

    if (p_child_h - com_node_child_h) == 1:
        while True:
            max_child_h = max([node_dict[f_c].child_h for f_c in node_dict[parent_id].children])
            if node_dict[parent_id].child_h == (max_child_h + 1):
                break
            node_dict[parent_id].child_h = max_child_h + 1
            parent_id = node_dict[parent_id].parent
            if parent_id is None:
                break


def child_tree_deepth(node_dict, nid):
    node = node_dict[nid]
    deepth = 0
    while node.parent is not None:
        node = node_dict[node.parent]
        deepth += 1
    deepth += node_dict[nid].child_h
    return deepth


def CompressDelta(node1, p_node):
    a = node1.child_cut
    v1 = node1.vol
    v2 = p_node.vol
    return a * math.log(v2 / v1)


def CombineDelta(node1, node2, cut_v, g_vol):
    v1 = node1.vol
    v2 = node2.vol
    g1 = node1.g
    g2 = node2.g
    v12 = v1 + v2
    return ((v1 - g1) * math.log(v12 / v1, 2) + (v2 - g2) * math.log(v12 / v2, 2) - 2 * cut_v * math.log(g_vol / v12,
                                                                                                         2)) / g_vol


class PartitionTreeNode():
    def __init__(self, ID, partition, vol, g, children: set = None, parent=None, child_h=0, child_cut=0):
        self.ID = ID
        self.partition = partition
        self.parent = parent
        self.children = children
        self.vol = vol+1e-15
        self.g = g
        self.merged = False
        self.child_h = child_h  # 不包括该节点的子树高度
        self.child_cut = child_cut

    # def __str__(self):
    #     return "{" + "{}:{}".format(self.__class__.__name__, self.gatherAttrs()) + "}"
    #
    # def gatherAttrs(self):
    #     return ",".join("{}={}"
    #                     .format(k, getattr(self, k))
    #                     for k in self.__dict__.keys())


class PartitionTree():

    def __init__(self, adj_matrix):
        self.adj_matrix = adj_matrix
        self.tree_node = {}
        # self.g_num_nodes, self.VOL, self.node_vol, self.adj_table, self.node_avg_vol = graph_parse(adj_matrix)
        self.g_num_nodes, self.VOL, self.node_vol, self.adj_table, self.node_g = graph_parse(adj_matrix)
        self.id_g = 0
        self.leaves = []
        self.build_leaves()
        # self.multi_dict = Manager().dict()

    def build_leaves(self):
        for vertex in range(self.g_num_nodes):
            ID = self.id_g
            self.id_g += 1
            v = self.node_vol[vertex]
            g = self.node_g[vertex]
            # normal
            # leaf_node = PartitionTreeNode(ID=ID, partition=[vertex], g = v, vol=v)
            # bin modal
            leaf_node = PartitionTreeNode(ID=ID, partition=1 << vertex, g=g, vol=v)
            self.tree_node[ID] = leaf_node
            self.leaves.append(ID)

    def build_sub_leaves(self, node_list, p_vol):
        subgraph_node_dict = {}
        ori_ent = 0
        # bin modal
        while (node_list):
            lowbit = node_list & (-node_list)
            vertex = int(math.log2(lowbit))
            ori_ent += -(self.tree_node[vertex].g / self.VOL) \
                       * math.log2(self.tree_node[vertex].vol / p_vol)
            sub_n = self.adj_table[vertex] & node_list
            vol = 0
            while (sub_n):
                lowbit_ = sub_n & (-sub_n)
                vertex_n = int(math.log2(lowbit_))
                vol += self.adj_matrix[vertex, vertex_n]
                sub_n ^= lowbit_
            # vol = self.node_avg_vol[vertex] * (bin(sub_n).count("1"))
            # sub_n = 0
            # vol = 0
            # for vertex_n in node_list:
            #     c = self.adj_matrix[vertex, vertex_n]
            #     if c != 0:
            #         vol += c
                    # sub_n.add(vertex_n)
                    # sub_n |= 1 << vertex_n
            sub_leaf = PartitionTreeNode(ID=vertex, partition=1 << vertex, g=vol, vol=vol)
            subgraph_node_dict[vertex] = sub_leaf
            self.adj_table[vertex] = sub_n
            node_list ^= lowbit

        # normal
        # for vertex in node_list:
        #     ori_ent += -(self.tree_node[vertex].g / self.VOL)\
        #                * math.log2(self.tree_node[vertex].vol / p_vol)
        #     # sub_n = set()
        #     sub_n = 0
        #     vol = 0
        #     for vertex_n in node_list:
        #         c = self.adj_matrix[vertex,vertex_n]
        #         if c != 0:
        #             vol += c
        #             # sub_n.add(vertex_n)
        #             sub_n |= 1 << vertex_n
        #     # normal
        #     # sub_leaf = PartitionTreeNode(ID=vertex,partition=[vertex],g=vol,vol=vol)
        #     # bin modal
        #     sub_leaf = PartitionTreeNode(ID=vertex, partition=1 << vertex, g=vol, vol=vol)
        #     subgraph_node_dict[vertex] = sub_leaf
        #     self.adj_table[vertex] = sub_n
        #     self.node_avg_vol[vertex] = vol/bin(sub_n).count("1")
        return subgraph_node_dict, ori_ent

    def build_root_down(self):
        root_child = self.tree_node[self.root_id].children
        subgraph_node_dict = {}
        ori_en = 0
        g_vol = self.tree_node[self.root_id].vol
        for node_id in root_child:
            node = self.tree_node[node_id]
            ori_en += -(node.g / g_vol) * math.log2(node.vol / g_vol)
            # new_n = set()
            new_n = 0
            temp = self.adj_table[node_id]
            while (temp):
                lowbit = temp & (-temp)
                nei_id = int(math.log2(lowbit))
                if nei_id in root_child:
                    new_n |= lowbit
                temp ^= lowbit

            # for nei in self.adj_table[node_id]:
            #     if nei in root_child:
            #         new_n.add(nei)

            self.adj_table[node_id] = new_n
            new_node = PartitionTreeNode(ID=node_id, partition=node.partition, vol=node.vol, g=node.g,
                                         children=node.children)
            subgraph_node_dict[node_id] = new_node

        return subgraph_node_dict, ori_en

    def entropy(self, node_dict=None):
        if node_dict is None:
            node_dict = self.tree_node
        ent = 0
        for node_id, node in node_dict.items():
            if node.parent is not None:
                node_p = node_dict[node.parent]
                node_vol = node.vol
                node_g = node.g
                node_p_vol = node_p.vol
                ent += - (node_g / self.VOL) * math.log2(node_vol / node_p_vol)
        return ent

    # def init_worker(self):
    #     self.var_dict['unmerged_count'] = self.unmerged_count
    #     self.var_dict['data_shape'] = data_shape
    #     self.var_dict['w'] = w
    #     self.var_dict['w_shape'] = w_shape

    def merge_threading(self, new_id, id1, id2, nodes_dict, g_vol):
        # t3 = time.time()
        # print(new_id)
        # print(self.unmerged_count)
        min_heap = []
        cmp_heap = []
        # compress delta
        if nodes_dict[id1].child_h > 0:
            cmp_heap.append([CompressDelta(nodes_dict[id1], nodes_dict[new_id]), id1, new_id])
        if nodes_dict[id2].child_h > 0:
            cmp_heap.append([CompressDelta(nodes_dict[id2], nodes_dict[new_id]), id2, new_id])
        # print(self.adj_table[new_id])
        # for ID in self.adj_table[new_id]:
        #     try:
        #         if not nodes_dict[ID].merged:
        #             n1 = nodes_dict[ID]
        #             n2 = nodes_dict[new_id]
        #             cut_v = cut_volume(self.adj_matrix, np.array(n1.partition), np.array(n2.partition))
        #
        #             new_diff = CombineDelta(nodes_dict[ID], nodes_dict[new_id], cut_v, g_vol)
        #             min_heap.append((new_diff, ID, new_id, cut_v))
        #     except:
        #         pass
        temp = self.adj_table[new_id]
        while (temp):
            lowbit = temp & (-temp)
            ID = int(math.log2(lowbit))
            try:
                if not nodes_dict[ID].merged:
                    n1 = nodes_dict[ID]
                    n2 = nodes_dict[new_id]
                    # cut_v = cut_volume(self.node_avg_vol, self.adj_table, p1=n1.partition, p2=n2.partition)
                    cut_v = cut_volume(self.adj_matrix, self.adj_table, p1=n1.partition, p2=n2.partition)
                    new_diff = CombineDelta(nodes_dict[ID], nodes_dict[new_id], cut_v, g_vol)
                    min_heap.append((new_diff, ID, new_id, cut_v))
            except:
                pass
            temp ^= lowbit
        # t4 = time.time()
        # print("单个进程所用时间:",t4-t3)
        return min_heap, cmp_heap

    def merge_pool(self, args):
        return self.merge_threading(*args)

    # bin model
    def __build_k_tree(self, g_vol, nodes_dict: dict, k=None, ):
        # self.multi_dict['nodes_dict'] = nodes_dict
        self.min_heap = []
        self.cmp_heap = []
        nodes_ids = nodes_dict.keys()
        # print(list(nodes_ids))
        new_id = None
        # 堆
        for i in nodes_ids:
            temp = self.adj_table[i]
            while (temp):
                lowbit = temp & (-temp)
                j = int(math.log2(lowbit))
                if j > i:
                    n1 = nodes_dict[i]
                    n2 = nodes_dict[j]
                    # cut_v = cut_volume(self.node_avg_vol, self.adj_table, p1=n1.partition, p2=n2.partition)
                    cut_v = cut_volume(self.adj_matrix, self.adj_table, p1=n1.partition, p2=n2.partition)
                    diff = CombineDelta(nodes_dict[i], nodes_dict[j], cut_v, g_vol)
                    heapq.heappush(self.min_heap, (diff, i, j, cut_v))
                temp ^= lowbit

            # for j in self.adj_table[i]:
            #     if j > i:
            #         n1 = nodes_dict[i]
            #         n2 = nodes_dict[j]
            #         if len(n1.partition) == 1 and len(n2.partition) == 1:
            #             cut_v = self.adj_matrix[n1.partition[0],n2.partition[0]]
            #         else:
            #             cut_v = cut_volume(self.adj_matrix,p1 = np.array(n1.partition),p2=np.array(n2.partition))
            #         diff = CombineDelta(nodes_dict[i], nodes_dict[j], cut_v, g_vol)
            #         heapq.heappush(self.min_heap, (diff, i, j, cut_v))

        # 采样
        # unmerged_nodes = set()
        # unmerged_count = len(nodes_ids)
        # unmerged_node = list(nodes_dict.keys())
        # total_edge = np.sum(np.where(self.adj_matrix>0,1,0))/2
        # adj_unmerged_table = self.adj_table.copy()
        # while unmerged_count > 1 and unmerged_node:
        #     count = 0
        #     min_diff = np.inf
        #     while count < 10 and unmerged_node:
        #         if len(unmerged_node) <= 20:
        #             i = np.random.choice(unmerged_node)
        #             if not adj_unmerged_table[i]:
        #                 unmerged_node.remove(i)
        #                 # unmerged_nodes.add(i)
        #                 continue
        #             j = np.random.choice(list(adj_unmerged_table[i]))
        #             if nodes_dict[i].merged or nodes_dict[j].merged or j <= i:
        #                 continue
        #             count += 1
        #             n1 = nodes_dict[i]
        #             n2 = nodes_dict[j]
        #             if len(n1.partition) == 1 and len(n2.partition) == 1:
        #                 cut_v = self.adj_matrix[n1.partition[0], n2.partition[0]]
        #             else:
        #                 cut_v = cut_volume(self.adj_matrix, p1=np.array(n1.partition), p2=np.array(n2.partition))
        #             diff = CombineDelta(nodes_dict[i], nodes_dict[j], cut_v, g_vol)
        #             if diff < min_diff:
        #                 min_diff, id1, id2, cut_v_ = diff, i, j, cut_v
        #             break
        #         i = np.random.choice(unmerged_node)
        #         if not adj_unmerged_table[i]:
        #             unmerged_node.remove(i)
        #             continue
        #         j = np.random.choice(list(adj_unmerged_table[i]))
        #         if nodes_dict[i].merged or nodes_dict[j].merged or j <= i:
        #             continue
        #         count += 1
        #         n1 = nodes_dict[i]
        #         n2 = nodes_dict[j]
        #         if len(n1.partition) == 1 and len(n2.partition) == 1:
        #             cut_v = self.adj_matrix[n1.partition[0], n2.partition[0]]
        #         else:
        #             cut_v = cut_volume(self.adj_matrix, p1=np.array(n1.partition), p2=np.array(n2.partition))
        #         diff = CombineDelta(nodes_dict[i], nodes_dict[j], cut_v, g_vol)
        #         if diff < min_diff:
        #             min_diff, id1, id2, cut_v_ = diff, i, j, cut_v
        #     if min_diff == np.inf:
        #         break
        #     nodes_dict[id1].merged = True
        #     nodes_dict[id2].merged = True
        #     new_id = self.id_g
        #     self.id_g += 1
        #     unmerged_node.remove(id1)
        #     unmerged_node.remove(id2)
        #     unmerged_node.append(new_id)
        #     merge(new_id, id1, id2, cut_v_, nodes_dict)
        #     self.adj_table[new_id] = self.adj_table[id1].union(self.adj_table[id2])
        #     adj_unmerged_table[id1] -= {id2}
        #     adj_unmerged_table[id2] -= {id1}
        #     adj_unmerged_table[new_id] = adj_unmerged_table[id1].union(adj_unmerged_table[id2])
        #     for i in adj_unmerged_table[id1]:
        #         adj_unmerged_table[i].remove(id1)
        #     for i in adj_unmerged_table[id2]:
        #         adj_unmerged_table[i].remove(id2)
        #     for i in adj_unmerged_table[new_id]:
        #         adj_unmerged_table[i].add(new_id)
        #     total_edge += len(adj_unmerged_table[new_id])
        #     total_edge -= len(adj_unmerged_table[id1])
        #     total_edge -= len(adj_unmerged_table[id2])
        #     for i in self.adj_table[new_id]:
        #         self.adj_table[i].add(new_id)
        #     # compress delta
        #     if nodes_dict[id1].child_h > 0:
        #         heapq.heappush(self.cmp_heap,[CompressDelta(nodes_dict[id1],nodes_dict[new_id]),id1,new_id])
        #     if nodes_dict[id2].child_h > 0:
        #         heapq.heappush(self.cmp_heap,[CompressDelta(nodes_dict[id2],nodes_dict[new_id]),id2,new_id])
        #     unmerged_count -= 1
        #     # print(unmerged_count)

        # 并行化
        # self.unmerged_count = len(nodes_ids)
        # with Pool(10) as pool:
        #     while self.unmerged_count > 1:
        #         if len(self.min_heap) == 0:
        #             break
        #         if len(self.min_heap) < 10:
        #             diff, id1, id2, cut_v = heapq.heappop(self.min_heap)
        #             if nodes_dict[id1].merged or nodes_dict[id2].merged:
        #                 continue
        #             nodes_dict[id1].merged = True
        #             nodes_dict[id2].merged = True
        #             new_id = self.id_g
        #             self.id_g += 1
        #             self.adj_table[new_id] = self.adj_table[id1].union(self.adj_table[id2])
        #             for i in self.adj_table[new_id]:
        #                 self.adj_table[i].add(new_id)
        #             self.unmerged_count -= 1
        #             merge(new_id, id1, id2, cut_v, nodes_dict)
        #             result = self.merge_threading(new_id, id1, id2, nodes_dict, g_vol)
        #             for i in result:
        #                 for j in i[0]:
        #                     heapq.heappush(self.min_heap, j)
        #                 for j in i[1]:
        #                     heapq.heappush(self.cmp_heap, j)
        #         else:
        #             task = []
        #             # pool = Pool(2)
        #             while len(task) < 10:
        #                 if len(self.min_heap) == 0:
        #                     break
        #                 diff, id1, id2, cut_v = heapq.heappop(self.min_heap)
        #                 if nodes_dict[id1].merged or nodes_dict[id2].merged:
        #                     continue
        #                 nodes_dict[id1].merged = True
        #                 nodes_dict[id2].merged = True
        #                 new_id = self.id_g
        #                 self.id_g += 1
        #                 self.adj_table[new_id] = self.adj_table[id1].union(self.adj_table[id2])
        #                 for i in self.adj_table[new_id]:
        #                     self.adj_table[i].add(new_id)
        #                 self.unmerged_count -= 1
        #                 merge(new_id, id1, id2, cut_v, nodes_dict)
        #                 task.append((new_id, id1, id2, nodes_dict, g_vol))
        #                 # task.append((1,1,1,1,1,1))
        #                 # set_trace()
        #                 # pool.imap(self.merge_threading,[(new_id, id1, id2, cut_v, nodes_dict, g_vol)])
        #             # set_trace()
        #             #     pool.close()
        #             #     pool.join()
        #             if task:
        #                 t1 = time.time()
        #                 result = list(pool.imap_unordered(self.merge_pool, task))
        #                 t2 = time.time()
        #                 print("10个进程所用时间:",t2-t1)
        #                 # print(result)
        #                 # self.merge_threading(new_id, id1, id2, cut_v, nodes_dict, g_vol)
        #                 # result = list(tqdm.tqdm(pool.imap(self.merge_pool, [(1,2),(3,4)])))
        #
        #                 for i in result:
        #                     for j in i[0]:
        #                         heapq.heappush(self.min_heap,j)
        #                     for j in i[1]:
        #                         heapq.heappush(self.cmp_heap,j)
        #                 # for i in result:
        #                 #     print(result)

        # 堆
        unmerged_count = len(nodes_ids)
        while unmerged_count > 1:
            if len(self.min_heap) == 0:
                break
            diff, id1, id2, cut_v = heapq.heappop(self.min_heap)
            if nodes_dict[id1].merged or nodes_dict[id2].merged:
                continue
            nodes_dict[id1].merged = True
            nodes_dict[id2].merged = True
            new_id = self.id_g
            self.id_g += 1
            # self.merge_threading(new_id, id1, id2, cut_v, nodes_dict, g_vol)
            merge(new_id, id1, id2, cut_v, nodes_dict)
            self.adj_table[new_id] = self.adj_table[id1] | self.adj_table[id2]
            temp = self.adj_table[new_id]
            while (temp):
                lowbit = temp & (-temp)
                i = int(math.log2(lowbit))
                self.adj_table[i] |= 1 << new_id
                temp ^= lowbit

            # self.adj_table[new_id] = self.adj_table[id1].union(self.adj_table[id2])
            # for i in self.adj_table[new_id]:
            #     self.adj_table[i].add(new_id)
            #
            # compress delta
            if nodes_dict[id1].child_h > 0:
                heapq.heappush(self.cmp_heap, [CompressDelta(nodes_dict[id1], nodes_dict[new_id]), id1, new_id])
            if nodes_dict[id2].child_h > 0:
                heapq.heappush(self.cmp_heap, [CompressDelta(nodes_dict[id2], nodes_dict[new_id]), id2, new_id])
            unmerged_count -= 1

            temp = self.adj_table[new_id]
            while (temp):
                lowbit = temp & (-temp)
                ID = int(math.log2(lowbit))
                if not nodes_dict[ID].merged:
                    n1 = nodes_dict[ID]
                    n2 = nodes_dict[new_id]
                    # cut_v = cut_volume(self.node_avg_vol, self.adj_table, n1.partition, n2.partition)
                    cut_v = cut_volume(self.adj_matrix, self.adj_table, n1.partition, n2.partition)

                    new_diff = CombineDelta(nodes_dict[ID], nodes_dict[new_id], cut_v, g_vol)
                    heapq.heappush(self.min_heap, (new_diff, ID, new_id, cut_v))
                temp ^= lowbit

            # for ID in self.adj_table[new_id]:
            #     if not nodes_dict[ID].merged:
            #         n1 = nodes_dict[ID]
            #         n2 = nodes_dict[new_id]
            #         cut_v = cut_volume(self.adj_matrix, np.array(n1.partition), np.array(n2.partition))
            #
            #         new_diff = CombineDelta(nodes_dict[ID], nodes_dict[new_id], cut_v, g_vol)
            #         heapq.heappush(self.min_heap, (new_diff, ID, new_id, cut_v))
        #
        root = new_id
        # for u,v in self.tree_node.items():
        #     if v.g < 0:
        #         print(v.ID)

        if unmerged_count > 1:
            # combine solitary node
            # print('processing solitary node')
            # assert len(self.min_heap) == 0
            unmerged_nodes = {i for i, j in nodes_dict.items() if not j.merged}
            new_child_h = max([nodes_dict[i].child_h for i in unmerged_nodes]) + 1

            unmerged_g = 0
            for i in list(unmerged_nodes):
                unmerged_g += nodes_dict[i].g

            new_id = self.id_g
            self.id_g += 1
            new_node = PartitionTreeNode(ID=new_id, partition=(2 ** len(nodes_ids)) - 1, children=unmerged_nodes,
                                         vol=g_vol, g=unmerged_g, child_h=new_child_h)
            nodes_dict[new_id] = new_node

            for i in unmerged_nodes:
                nodes_dict[i].merged = True
                nodes_dict[i].parent = new_id
                if nodes_dict[i].child_h > 0:
                    heapq.heappush(self.cmp_heap, [CompressDelta(nodes_dict[i], nodes_dict[new_id]), i, new_id])
            root = new_id

        if k is not None:
            while nodes_dict[root].child_h > k:
                diff, node_id, p_id = heapq.heappop(self.cmp_heap)
                if child_tree_deepth(nodes_dict, node_id) <= k:
                    continue
                children = nodes_dict[node_id].children
                compressNode(nodes_dict, node_id, p_id)
                if nodes_dict[root].child_h == k:
                    break
                for e in self.cmp_heap:
                    if e[1] == p_id:
                        if child_tree_deepth(nodes_dict, p_id) > k:
                            e[0] = CompressDelta(nodes_dict[e[1]], nodes_dict[e[2]])
                    if e[1] in children:
                        if nodes_dict[e[1]].child_h == 0:
                            continue
                        if child_tree_deepth(nodes_dict, e[1]) > k:
                            e[2] = p_id
                            e[0] = CompressDelta(nodes_dict[e[1]], nodes_dict[p_id])
                heapq.heapify(self.cmp_heap)
        return root

    # normal
    # def __build_k_tree(self,g_vol,nodes_dict:dict,k = None,):
    #     # self.multi_dict['nodes_dict'] = nodes_dict
    #     self.min_heap = []
    #     self.cmp_heap = []
    #     nodes_ids = nodes_dict.keys()
    #     new_id = None
    #     # 堆
    #     for i in nodes_ids:
    #         temp = self.adj_table[i]
    #         while(temp):
    #             lowbit = temp & (-temp)
    #             j = int(math.log2(lowbit))
    #             if j > i:
    #                 n1 = nodes_dict[i]
    #                 n2 = nodes_dict[j]
    #                 if len(n1.partition) == 1 and len(n2.partition) == 1:
    #                     cut_v = self.adj_matrix[n1.partition[0], n2.partition[0]]
    #                 else:
    #                     cut_v = cut_volume(self.adj_matrix, p1=np.array(n1.partition), p2=np.array(n2.partition))
    #                 diff = CombineDelta(nodes_dict[i], nodes_dict[j], cut_v, g_vol)
    #                 heapq.heappush(self.min_heap, (diff, i, j, cut_v))
    #             temp ^= lowbit
    #
    #         # for j in self.adj_table[i]:
    #         #     if j > i:
    #         #         n1 = nodes_dict[i]
    #         #         n2 = nodes_dict[j]
    #         #         if len(n1.partition) == 1 and len(n2.partition) == 1:
    #         #             cut_v = self.adj_matrix[n1.partition[0],n2.partition[0]]
    #         #         else:
    #         #             cut_v = cut_volume(self.adj_matrix,p1 = np.array(n1.partition),p2=np.array(n2.partition))
    #         #         diff = CombineDelta(nodes_dict[i], nodes_dict[j], cut_v, g_vol)
    #         #         heapq.heappush(self.min_heap, (diff, i, j, cut_v))
    #
    #     # 采样
    #     # unmerged_nodes = set()
    #     # unmerged_count = len(nodes_ids)
    #     # unmerged_node = list(nodes_dict.keys())
    #     # total_edge = np.sum(np.where(self.adj_matrix>0,1,0))/2
    #     # adj_unmerged_table = self.adj_table.copy()
    #     # while unmerged_count > 1 and unmerged_node:
    #     #     count = 0
    #     #     min_diff = np.inf
    #     #     while count < 10 and unmerged_node:
    #     #         if len(unmerged_node) <= 20:
    #     #             i = np.random.choice(unmerged_node)
    #     #             if not adj_unmerged_table[i]:
    #     #                 unmerged_node.remove(i)
    #     #                 # unmerged_nodes.add(i)
    #     #                 continue
    #     #             j = np.random.choice(list(adj_unmerged_table[i]))
    #     #             if nodes_dict[i].merged or nodes_dict[j].merged or j <= i:
    #     #                 continue
    #     #             count += 1
    #     #             n1 = nodes_dict[i]
    #     #             n2 = nodes_dict[j]
    #     #             if len(n1.partition) == 1 and len(n2.partition) == 1:
    #     #                 cut_v = self.adj_matrix[n1.partition[0], n2.partition[0]]
    #     #             else:
    #     #                 cut_v = cut_volume(self.adj_matrix, p1=np.array(n1.partition), p2=np.array(n2.partition))
    #     #             diff = CombineDelta(nodes_dict[i], nodes_dict[j], cut_v, g_vol)
    #     #             if diff < min_diff:
    #     #                 min_diff, id1, id2, cut_v_ = diff, i, j, cut_v
    #     #             break
    #     #         i = np.random.choice(unmerged_node)
    #     #         if not adj_unmerged_table[i]:
    #     #             unmerged_node.remove(i)
    #     #             continue
    #     #         j = np.random.choice(list(adj_unmerged_table[i]))
    #     #         if nodes_dict[i].merged or nodes_dict[j].merged or j <= i:
    #     #             continue
    #     #         count += 1
    #     #         n1 = nodes_dict[i]
    #     #         n2 = nodes_dict[j]
    #     #         if len(n1.partition) == 1 and len(n2.partition) == 1:
    #     #             cut_v = self.adj_matrix[n1.partition[0], n2.partition[0]]
    #     #         else:
    #     #             cut_v = cut_volume(self.adj_matrix, p1=np.array(n1.partition), p2=np.array(n2.partition))
    #     #         diff = CombineDelta(nodes_dict[i], nodes_dict[j], cut_v, g_vol)
    #     #         if diff < min_diff:
    #     #             min_diff, id1, id2, cut_v_ = diff, i, j, cut_v
    #     #     if min_diff == np.inf:
    #     #         break
    #     #     nodes_dict[id1].merged = True
    #     #     nodes_dict[id2].merged = True
    #     #     new_id = self.id_g
    #     #     self.id_g += 1
    #     #     unmerged_node.remove(id1)
    #     #     unmerged_node.remove(id2)
    #     #     unmerged_node.append(new_id)
    #     #     merge(new_id, id1, id2, cut_v_, nodes_dict)
    #     #     self.adj_table[new_id] = self.adj_table[id1].union(self.adj_table[id2])
    #     #     adj_unmerged_table[id1] -= {id2}
    #     #     adj_unmerged_table[id2] -= {id1}
    #     #     adj_unmerged_table[new_id] = adj_unmerged_table[id1].union(adj_unmerged_table[id2])
    #     #     for i in adj_unmerged_table[id1]:
    #     #         adj_unmerged_table[i].remove(id1)
    #     #     for i in adj_unmerged_table[id2]:
    #     #         adj_unmerged_table[i].remove(id2)
    #     #     for i in adj_unmerged_table[new_id]:
    #     #         adj_unmerged_table[i].add(new_id)
    #     #     total_edge += len(adj_unmerged_table[new_id])
    #     #     total_edge -= len(adj_unmerged_table[id1])
    #     #     total_edge -= len(adj_unmerged_table[id2])
    #     #     for i in self.adj_table[new_id]:
    #     #         self.adj_table[i].add(new_id)
    #     #     # compress delta
    #     #     if nodes_dict[id1].child_h > 0:
    #     #         heapq.heappush(self.cmp_heap,[CompressDelta(nodes_dict[id1],nodes_dict[new_id]),id1,new_id])
    #     #     if nodes_dict[id2].child_h > 0:
    #     #         heapq.heappush(self.cmp_heap,[CompressDelta(nodes_dict[id2],nodes_dict[new_id]),id2,new_id])
    #     #     unmerged_count -= 1
    #     #     # print(unmerged_count)
    #
    #     # 并行化
    #     # self.unmerged_count = len(nodes_ids)
    #     # with Pool(10) as pool:
    #     #     while self.unmerged_count > 1:
    #     #         if len(self.min_heap) == 0:
    #     #             break
    #     #         if len(self.min_heap) < 10:
    #     #             diff, id1, id2, cut_v = heapq.heappop(self.min_heap)
    #     #             if nodes_dict[id1].merged or nodes_dict[id2].merged:
    #     #                 continue
    #     #             nodes_dict[id1].merged = True
    #     #             nodes_dict[id2].merged = True
    #     #             new_id = self.id_g
    #     #             self.id_g += 1
    #     #             self.adj_table[new_id] = self.adj_table[id1].union(self.adj_table[id2])
    #     #             for i in self.adj_table[new_id]:
    #     #                 self.adj_table[i].add(new_id)
    #     #             self.unmerged_count -= 1
    #     #             merge(new_id, id1, id2, cut_v, nodes_dict)
    #     #             result = self.merge_threading(new_id, id1, id2, nodes_dict, g_vol)
    #     #             for i in result:
    #     #                 for j in i[0]:
    #     #                     heapq.heappush(self.min_heap, j)
    #     #                 for j in i[1]:
    #     #                     heapq.heappush(self.cmp_heap, j)
    #     #         else:
    #     #             task = []
    #     #             # pool = Pool(2)
    #     #             while len(task) < 10:
    #     #                 if len(self.min_heap) == 0:
    #     #                     break
    #     #                 diff, id1, id2, cut_v = heapq.heappop(self.min_heap)
    #     #                 if nodes_dict[id1].merged or nodes_dict[id2].merged:
    #     #                     continue
    #     #                 nodes_dict[id1].merged = True
    #     #                 nodes_dict[id2].merged = True
    #     #                 new_id = self.id_g
    #     #                 self.id_g += 1
    #     #                 self.adj_table[new_id] = self.adj_table[id1].union(self.adj_table[id2])
    #     #                 for i in self.adj_table[new_id]:
    #     #                     self.adj_table[i].add(new_id)
    #     #                 self.unmerged_count -= 1
    #     #                 merge(new_id, id1, id2, cut_v, nodes_dict)
    #     #                 task.append((new_id, id1, id2, nodes_dict, g_vol))
    #     #                 # task.append((1,1,1,1,1,1))
    #     #                 # set_trace()
    #     #                 # pool.imap(self.merge_threading,[(new_id, id1, id2, cut_v, nodes_dict, g_vol)])
    #     #             # set_trace()
    #     #             #     pool.close()
    #     #             #     pool.join()
    #     #             if task:
    #     #                 t1 = time.time()
    #     #                 result = list(pool.imap_unordered(self.merge_pool, task))
    #     #                 t2 = time.time()
    #     #                 print("10个进程所用时间:",t2-t1)
    #     #                 # print(result)
    #     #                 # self.merge_threading(new_id, id1, id2, cut_v, nodes_dict, g_vol)
    #     #                 # result = list(tqdm.tqdm(pool.imap(self.merge_pool, [(1,2),(3,4)])))
    #     #
    #     #                 for i in result:
    #     #                     for j in i[0]:
    #     #                         heapq.heappush(self.min_heap,j)
    #     #                     for j in i[1]:
    #     #                         heapq.heappush(self.cmp_heap,j)
    #     #                 # for i in result:
    #     #                 #     print(result)
    #
    #     # 堆
    #     unmerged_count = len(nodes_ids)
    #     while unmerged_count > 1:
    #         if len(self.min_heap) == 0:
    #             break
    #         diff, id1, id2, cut_v = heapq.heappop(self.min_heap)
    #         if nodes_dict[id1].merged or nodes_dict[id2].merged:
    #             continue
    #         nodes_dict[id1].merged = True
    #         nodes_dict[id2].merged = True
    #         new_id = self.id_g
    #         self.id_g += 1
    #         # self.merge_threading(new_id, id1, id2, cut_v, nodes_dict, g_vol)
    #         merge(new_id, id1, id2, cut_v, nodes_dict)
    #         self.adj_table[new_id] = self.adj_table[id1] | self.adj_table[id2]
    #         temp = self.adj_table[new_id]
    #         while (temp):
    #             lowbit = temp & (-temp)
    #             i = int(math.log2(lowbit))
    #             self.adj_table[i] |= 1 << new_id
    #             temp ^= lowbit
    #
    #         # self.adj_table[new_id] = self.adj_table[id1].union(self.adj_table[id2])
    #         # for i in self.adj_table[new_id]:
    #         #     self.adj_table[i].add(new_id)
    #         #
    #         # compress delta
    #         if nodes_dict[id1].child_h > 0:
    #             heapq.heappush(self.cmp_heap, [CompressDelta(nodes_dict[id1], nodes_dict[new_id]), id1, new_id])
    #         if nodes_dict[id2].child_h > 0:
    #             heapq.heappush(self.cmp_heap, [CompressDelta(nodes_dict[id2], nodes_dict[new_id]), id2, new_id])
    #         unmerged_count -= 1
    #
    #         temp = self.adj_table[new_id]
    #         while (temp):
    #             lowbit = temp & (-temp)
    #             ID = int(math.log2(lowbit))
    #             if not nodes_dict[ID].merged:
    #                 n1 = nodes_dict[ID]
    #                 n2 = nodes_dict[new_id]
    #                 cut_v = cut_volume(self.adj_matrix, np.array(n1.partition), np.array(n2.partition))
    #
    #                 new_diff = CombineDelta(nodes_dict[ID], nodes_dict[new_id], cut_v, g_vol)
    #                 heapq.heappush(self.min_heap, (new_diff, ID, new_id, cut_v))
    #             temp ^= lowbit
    #
    #         # for ID in self.adj_table[new_id]:
    #         #     if not nodes_dict[ID].merged:
    #         #         n1 = nodes_dict[ID]
    #         #         n2 = nodes_dict[new_id]
    #         #         cut_v = cut_volume(self.adj_matrix, np.array(n1.partition), np.array(n2.partition))
    #         #
    #         #         new_diff = CombineDelta(nodes_dict[ID], nodes_dict[new_id], cut_v, g_vol)
    #         #         heapq.heappush(self.min_heap, (new_diff, ID, new_id, cut_v))
    #     #
    #     root = new_id
    #
    #     if unmerged_count > 1:
    #         # combine solitary node
    #         # print('processing solitary node')
    #         # assert len(self.min_heap) == 0
    #         unmerged_nodes = {i for i, j in nodes_dict.items() if not j.merged}
    #         new_child_h = max([nodes_dict[i].child_h for i in unmerged_nodes]) + 1
    #
    #         new_id = self.id_g
    #         self.id_g += 1
    #         new_node = PartitionTreeNode(ID=new_id,partition=list(nodes_ids),children=unmerged_nodes,
    #                                      vol=g_vol,g = 0,child_h=new_child_h)
    #         nodes_dict[new_id] = new_node
    #
    #         for i in unmerged_nodes:
    #             nodes_dict[i].merged = True
    #             nodes_dict[i].parent = new_id
    #             if nodes_dict[i].child_h > 0:
    #                 heapq.heappush(self.cmp_heap, [CompressDelta(nodes_dict[i], nodes_dict[new_id]), i, new_id])
    #         root = new_id
    #
    #     if k is not None:
    #         while nodes_dict[root].child_h > k:
    #             diff, node_id, p_id = heapq.heappop(self.cmp_heap)
    #             if child_tree_deepth(nodes_dict, node_id) <= k:
    #                 continue
    #             children = nodes_dict[node_id].children
    #             compressNode(nodes_dict, node_id, p_id)
    #             if nodes_dict[root].child_h == k:
    #                 break
    #             for e in self.cmp_heap:
    #                 if e[1] == p_id:
    #                     if child_tree_deepth(nodes_dict, p_id) > k:
    #                         e[0] = CompressDelta(nodes_dict[e[1]], nodes_dict[e[2]])
    #                 if e[1] in children:
    #                     if nodes_dict[e[1]].child_h == 0:
    #                         continue
    #                     if child_tree_deepth(nodes_dict, e[1]) > k:
    #                         e[2] = p_id
    #                         e[0] = CompressDelta(nodes_dict[e[1]], nodes_dict[p_id])
    #             heapq.heapify(self.cmp_heap)
    #     return root

    def auto_complete(self, node_dict, root_id, k):
        queue = [(root_id, 0)]
        while queue:
            c, depth = queue.pop(0)
            if node_dict[c].child_h == 0:
                if depth < k:
                    self.single_up(node_dict, c)
                    queue.append((c, depth+1))
            else:
                childs = copy.deepcopy(node_dict[c].children)
                depth += 1
                for child in childs:
                    queue.append((child, depth))

    def check_balance(self, node_dict, root_id):
        root_c = copy.deepcopy(node_dict[root_id].children)
        for c in root_c:
            if node_dict[c].child_h == 0:
                self.single_up(node_dict, c)

    def single_up(self, node_dict, node_id):
        new_id = self.id_g
        self.id_g += 1
        p_id = node_dict[node_id].parent
        grow_node = PartitionTreeNode(ID=new_id, partition=node_dict[node_id].partition, parent=p_id,
                                      children={node_id}, vol=node_dict[node_id].vol, g=node_dict[node_id].g)
        node_dict[node_id].parent = new_id
        node_dict[p_id].children.remove(node_id)
        node_dict[p_id].children.add(new_id)
        node_dict[new_id] = grow_node
        node_dict[new_id].child_h = node_dict[node_id].child_h + 1
        self.adj_table[new_id] = self.adj_table[node_id]
        temp = self.adj_table[new_id]
        while (temp):
            lowbit = temp & (-temp)
            i = int(math.log2(lowbit))
            self.adj_table[i] |= lowbit
            temp ^= lowbit
        # for i in self.adj_table[node_id]:
        #     self.adj_table[i].add(new_id)

    def root_down_delta(self):
        # if len(self.tree_node[self.root_id].children) < 3:
        #     return 0, None, None
        subgraph_node_dict, ori_entropy = self.build_root_down()
        g_vol = self.tree_node[self.root_id].vol
        new_root = self.__build_k_tree(g_vol=g_vol, nodes_dict=subgraph_node_dict, k=2)
        self.check_balance(subgraph_node_dict, new_root)

        new_entropy = self.entropy(subgraph_node_dict)
        delta = (ori_entropy - new_entropy) / len(self.tree_node[self.root_id].children)
        return delta, new_root, subgraph_node_dict

    def leaf_up_entropy(self, sub_node_dict, sub_root_id, node_id):
        ent = 0
        for sub_node_id in LayerFirst(sub_node_dict, sub_root_id):
            if sub_node_id == sub_root_id:
                sub_node_dict[sub_root_id].vol = self.tree_node[node_id].vol
                sub_node_dict[sub_root_id].g = self.tree_node[node_id].g

            elif sub_node_dict[sub_node_id].child_h == 1:
                node = sub_node_dict[sub_node_id]
                inner_vol = node.vol - node.g
                partition = node.partition
                # bin modal
                ori_vol = 0
                while partition:
                    lowbit = partition & (-partition)
                    i = int(math.log2(lowbit))
                    ori_vol += self.tree_node[i].vol
                    partition ^= lowbit
                # normal
                # ori_vol = sum(self.tree_node[i].vol for i in partition)
                ori_g = ori_vol - inner_vol
                node.vol = ori_vol
                node.g = ori_g
                node_p = sub_node_dict[node.parent]
                ent += -(node.g / self.VOL) * math.log2(node.vol / node_p.vol)
            else:
                node = sub_node_dict[sub_node_id]
                node.g = self.tree_node[sub_node_id].g
                node.vol = self.tree_node[sub_node_id].vol
                node_p = sub_node_dict[node.parent]
                ent += -(node.g / self.VOL) * math.log2(node.vol / node_p.vol)
        return ent

    def leaf_up(self):
        h1_id = set()
        h1_new_child_tree = {}
        id_mapping = {}
        for l in self.leaves:
            p = self.tree_node[l].parent
            h1_id.add(p)
        delta = 0
        for node_id in h1_id:
            candidate_node = self.tree_node[node_id]
            sub_nodes = candidate_node.partition
            if bin(sub_nodes).count("1") == 1:
                id_mapping[node_id] = None
            if bin(sub_nodes).count("1") == 2:
                id_mapping[node_id] = None
            if bin(sub_nodes).count("1") >= 3:
                sub_g_vol = candidate_node.vol - candidate_node.g
                subgraph_node_dict, ori_ent = self.build_sub_leaves(sub_nodes, candidate_node.vol)
                sub_root = self.__build_k_tree(g_vol=sub_g_vol, nodes_dict=subgraph_node_dict, k=2)
                self.check_balance(subgraph_node_dict, sub_root)
                new_ent = self.leaf_up_entropy(subgraph_node_dict, sub_root, node_id)
                delta += (ori_ent - new_ent)
                h1_new_child_tree[node_id] = subgraph_node_dict
                id_mapping[node_id] = sub_root
        delta = delta / self.g_num_nodes
        return delta, id_mapping, h1_new_child_tree

    def leaf_up_update(self, id_mapping, leaf_up_dict):
        for node_id, h1_root in id_mapping.items():
            if h1_root is None:
                children = copy.deepcopy(self.tree_node[node_id].children)
                for i in children:
                    self.single_up(self.tree_node, i)
            else:
                h1_dict = leaf_up_dict[node_id]
                self.tree_node[node_id].children = h1_dict[h1_root].children
                for h1_c in h1_dict[h1_root].children:
                    assert h1_c not in self.tree_node
                    h1_dict[h1_c].parent = node_id
                h1_dict.pop(h1_root)
                self.tree_node.update(h1_dict)
        self.tree_node[self.root_id].child_h += 1

    def root_down_update(self, new_id, root_down_dict):
        self.tree_node[self.root_id].children = root_down_dict[new_id].children
        for node_id in root_down_dict[new_id].children:
            assert node_id not in self.tree_node
            root_down_dict[node_id].parent = self.root_id
        root_down_dict.pop(new_id)
        self.tree_node.update(root_down_dict)
        self.tree_node[self.root_id].child_h += 1

    def build_encoding_tree(self, k=2, mode='v2'):
        if k == 1:
            return
        if mode == 'v1' or k is None:
            self.root_id = self.__build_k_tree(self.VOL, self.tree_node, k=k)
        elif mode == 'v2':
            self.root_id = self.__build_k_tree(self.VOL, self.tree_node, k=k)

            # self.check_balance(self.tree_node, self.root_id)

            # if self.tree_node[self.root_id].child_h < 2:
            #     self.tree_node[self.root_id].child_h = 2

            # flag = 0
            # while self.tree_node[self.root_id].child_h < k:
            #     if flag == 0:
            #         leaf_up_delta, id_mapping, leaf_up_dict = self.leaf_up()
            #         root_down_delta, new_id, root_down_dict = self.root_down_delta()
            #
            #     elif flag == 1:
            #         leaf_up_delta, id_mapping, leaf_up_dict = self.leaf_up()
            #     elif flag == 2:
            #         root_down_delta, new_id, root_down_dict = self.root_down_delta()
            #     else:
            #         raise ValueError
            #
            #     if leaf_up_delta < root_down_delta:
            #         # print('root down')
            #         # root down update and recompute root down delta
            #         flag = 2
            #         self.root_down_update(new_id, root_down_dict)
            #
            #     else:
            #         # leaf up update
            #         # print('leave up')
            #         flag = 1
            #         # print(self.tree_node[self.root_id].child_h)
            #         self.leaf_up_update(id_mapping, leaf_up_dict)
            #         # print(self.tree_node[self.root_id].child_h)
            #
            #         # update root down leave nodes' children
            #         if root_down_delta != 0:
            #             for root_down_id, root_down_node in root_down_dict.items():
            #                 if root_down_node.child_h == 0:
            #                     root_down_node.children = self.tree_node[root_down_id].children

        self.auto_complete(self.tree_node, self.root_id, k)
        count = 0
        for _ in LayerFirst(self.tree_node, self.root_id):
            count += 1
        assert len(self.tree_node) == count
#

# from scipy.optimize import linear_sum_assignment as linear_assignment
# def acc(y_true, y_pred):
#     """
#     Calculate clustering accuracy. Require scikit-learn installed
#     # Arguments
#         y: true labels, numpy.array with shape `(n_samples,)`
#         y_pred: predicted labels, numpy.array with shape `(n_samples,)`
#     # Return
#         accuracy, in [0,1]
#     """
#     y_true = y_true.astype(np.int64)
#     assert y_pred.size == y_true.size
#     D = max(np.max(y_pred), np.max(y_true)) + 1
#     w = np.zeros((D, D), dtype=np.int64)
#     for i in range(y_pred.size):
#         w[y_pred[i], y_true[i]] += 1
#     ind = np.array(linear_assignment(np.max(w) - w)).T
#     return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

if __name__ == "__main__":
    n_samples = 100
    random_state = 170
    data, label = make_blobs(n_samples=n_samples, cluster_std=[1.5, 0.5, 1.5], random_state=random_state,
                             center_box=[-5.0, 5.0])

    # data = pd.read_csv("../h.csv", header=None).to_numpy()
    # label = data[:, -1]
    # data = np.delete(data, -1, 1)
    k = len(np.unique(label))
    kd_tree = KDTree(data, leaf_size=20)
    try:
        dist, inds = kd_tree.query(data, k=5)
    except:
        dist, inds = kd_tree.query(data, k=len(data))
    G = nx.Graph()
    kmeans = KMeans(init='k-means++', n_clusters=k).fit(data)
    # print('k-means++ ACC:', round(acc(label, kmeans.labels_), 4))
    opt = kmeans.inertia_

    threshold = np.inf
    adj_mat = np.zeros((len(data), len(data)))
    for m in range(len(inds)):
        ind = inds[m]
        i = m
        for n in range(len(ind[1:])):
            if n:
                if dist[m][n + 1] > threshold:
                    continue
            j = ind[n + 1]
            adj_mat[i][j] = dist[m][n + 1]
            adj_mat[j][i] = dist[m][n + 1]

    t1 = time.time()
    y = PartitionTree(adj_matrix=adj_mat)
    x = y.build_encoding_tree(4)
    # for i in list(y.tree_node.keys()):
    #     print(y.tree_node[i].ID,":", y.tree_node[i].parent)
    t2 = time.time()
    print(t2 - t1)
    # for k, v in y.tree_node.items():
    #     print(k, v.__dict__)
