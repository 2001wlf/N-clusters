import math
import os
import copy
import time
import pickle
from multiprocessing import Pool

import numpy as np
import networkx as nx
import pandas as pd
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.datasets import make_blobs
from sklearn.metrics import normalized_mutual_info_score, f1_score, roc_auc_score, average_precision_score, \
    pairwise_distances_argmin_min
import argparse

import sys

sys.path.append('../')

# from Problem.outlier.FFS.FFS1 import Lloyd_minus

from drug.lib.coding_tree import PartitionTree

# from kmeans import KMEANS

color_map = {0: 'Green', 1: 'Purple', 2: 'Blue', 3: 'red'}
PWD = os.path.dirname(os.path.realpath(__file__))


def trans_to_adj(graph):
    graph.remove_edges_from(nx.selfloop_edges(graph))
    nodes = range(len(graph.nodes))
    return nx.to_numpy_array(graph, nodelist=nodes)


def trans_to_tree(adj, k=2):
    undirected_adj = np.array(adj)
    y = PartitionTree(adj_matrix=undirected_adj)
    x = y.build_encoding_tree(k)
    return y.tree_node


def update_depth(tree):
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
    d_id = [(v.child_h, v.ID) for k, v in tree.items()]
    d_id.sort()
    new_tree = {}
    for k, v in tree.items():
        n = copy.deepcopy(v)
        n.ID = d_id.index((n.child_h, n.ID))
        if n.parent is not None:
            n.parent = d_id.index((n.child_h + 1, n.parent))
        if n.children is not None:
            n.children = [d_id.index((n.child_h - 1, c)) for c in n.children]
        n = n.__dict__
        n['depth'] = n['child_h']
        new_tree[n['ID']] = n
    return new_tree


def pool_trans(input_):
    global step
    step += 1
    g, tree_depth = input_
    adj_mat = trans_to_adj(g['G'])
    assert np.all(adj_mat == adj_mat.T)
    if not nx.is_connected(g['G']):
        pass
    tree = trans_to_tree(adj_mat, tree_depth)
    g['tree'] = update_node(tree)
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
    print('valid graphs', len(g_list))
    with open('trees/%s_%s.pickle' % (dataset, tree_depth), 'wb') as fp:
        pickle.dump(g_list, fp)


def entropy(VOL, node_dict):
    ent = []
    for node_id, node in node_dict.items():
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
    ent = np.zeros((n, k))
    for id, tree_node in list(tree.items())[n:]:
        childrens = tree_node["children"]
        node_p_vol = tree_node['vol']
        h = tree_node["child_h"]
        for children in childrens:
            node = tree[children]
            node_vol = node['vol']
            node_g = node['g']
            e = - (node_g / VOL) * math.log2(node_vol / node_p_vol)
            temp = node["partition"]
            while (temp):
                lowbit = temp & (-temp)
                j = int(math.log2(lowbit))
                ent[j][h - 1] = e
                temp ^= lowbit
    return ent


def find_min_gap(np_nums):
    nums = np_nums.reshape(1, -1)[0]
    nums = np.sort(nums, kind='heapsort')
    min = np.inf
    for i in range(len(nums) - 1):
        current = nums[i + 1] - nums[i]
        if current == 0:
            continue
        if current < min:
            min = current
    return min


def make_uniform(start, end, length, np_nums):
    s = np.linspace(start, end, length).reshape(-1, 1)
    dis_map = distance.cdist(np_nums.reshape(-1, 1), s)
    assignment = np.argmin(dis_map, axis=1)
    new_nums = s[assignment]
    return new_nums


def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

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


def bulit_knn_graph(data, ne, threshold=np.inf):
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
    return G


def bulit_k_farthest_graph(data, ne, threshold=np.inf):
    dis_map = distance.cdist(data, data, metric='euclidean')
    inds = np.argpartition(dis_map, -ne, axis=1)[:, -ne:]
    G = nx.Graph()
    for m in range(len(inds)):
        ind = inds[m]
        for n in ind:
            if dis_map[m][n + 1] < threshold:
                continue
            G.add_edge(m, n, weight=dis_map[m][n])
    return G


def bulit_knn_adjm(data, ne, threshold=np.inf):
    adj_mat = np.zeros((len(data), len(data)))
    kd_tree = KDTree(data, leaf_size=20)
    try:
        dist, inds = kd_tree.query(data, k=ne + 1)
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
            adj_mat[i][j] = dist[m][n + 1]
            adj_mat[j][i] = dist[m][n + 1]
    return adj_mat


def bulit_k_farthest_adjm(data, ne, threshold=0):
    dis_map = distance.cdist(data, data, metric='euclidean')
    inds = np.argpartition(dis_map, -ne, axis=1)[:, -ne:]
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
    print(ne, tree_depth, gap)
    adj_mat = bulit_knn_adjm(data, ne, gap * math.sqrt(opt / len(data)))
    tree = trans_to_tree(adj_mat, tree_depth)
    tree = update_node(tree)
    data_entory = all_entory(len(data), tree_depth, np.sum(adj_mat), tree)
    data_ent = np.sum(data_entory, axis=1).reshape(-1, 1)
    data_ = np.hstack((data, data_ent))
    data = MinMaxScaler().fit_transform(data_)
    return data


def up_dimension_log(data, ne, tree_depth, gap, opt):
    print(ne, tree_depth, gap)
    adj_mat = bulit_knn_adjm(data, ne, gap * math.sqrt(opt / len(data)))
    tree = trans_to_tree(adj_mat, tree_depth)
    tree = update_node(tree)
    data_entory = all_entory(len(data), tree_depth, np.sum(adj_mat), tree)
    data_ent = np.sum(data_entory, axis=1).reshape(-1, 1)
    data_ent_ = MinMaxScaler().fit_transform(data_ent) + 1e-15
    data_ = np.hstack((data, np.log(data_ent_)))
    data = MinMaxScaler().fit_transform(data_)
    # kmeans = KMeans(init='k-means++', n_clusters=k).fit(data)
    # print('acc=', round(acc(label, kmeans.labels_), 4))
    # print('nmi=', round(normalized_mutual_info_score(label, kmeans.labels_), 4))
    # opt = kmeans.inertia_
    dis_map = np.sort(distance.cdist(data, data, 'euclidean').reshape(-1))
    max_dis = dis_map[-1]
    min_dis = dis_map[len(data)]
    print('d=', data.shape[1], ',\taspect_ratio=', max_dis / min_dis)
    return data, opt


def k_center_greedy(points, k):
    """
    k-center 贪心算法
    :param points: 点的集合，形状为 (n_points, n_dimensions) 的二维数组
    :param k: 需要选择的中心个数
    :return: 选择的中心索引列表和最大距离
    """
    n_points = points.shape[0]
    centers = []

    # 随机选择第一个中心
    first_center = np.random.randint(n_points)
    centers.append(first_center)

    # 计算每个点到第一个中心的距离
    distances = cdist(points, points[first_center, np.newaxis], metric='euclidean').flatten()

    for _ in range(1, k):
        # 选择使得当前最远距离最大的那个点作为新的中心
        next_center = np.argmax(distances)
        centers.append(next_center)

        # 更新每个点到最近中心的距离
        new_distances = cdist(points, points[next_center, np.newaxis], metric='euclidean').flatten()
        distances = np.minimum(distances, new_distances)

    # 计算最大距离
    max_distance = np.max(distances)

    return centers, max_distance


def add_noise(data,delta,z):
    print("number of outliers", z)
    scaler = preprocessing.StandardScaler().fit(data)
    d = data.shape[1]
    data = scaler.transform(data)
    noise = (np.random.rand(z,d) - 0.5) * 2 * delta
    #print(np.max(noise))
    data = np.vstack((data,noise))
    noise_label = [i for i in range(len(data)-z,len(data))]
    return data, noise_label

def SummaryOutliers(X, z, alpha, beta):
    """
        :param X: array of shape=(n_samples, n_features)
        :return
    """
    n_samples, _ = X.shape
    samples_ = []
    weights_ = []
    sample_indices_ = []
    X_i = np.arange(0, n_samples)  # indices for remained data points
    assignment = np.zeros(n_samples, dtype=np.int32)
    offset = 0

    while len(X_i) > max(z, 0):
        # print(len(X_i))
        kappa = np.log(len(X_i))
        S_i_size = int(alpha * kappa)
        # 6. construct a set S_i of size \alpha\kappa by random sampling (with replacement) from X_i
        S_i = np.random.choice(X_i, size=S_i_size, replace=True)
        S_i = np.unique(S_i)
        w_i = np.ones((len(S_i),))
        # print(len(S_i))
        # 6. for each point in X_i, compute the distance to its nearest point in S_i
        # each value in nearest would range from 0 to len(S_i)
        # sTime = time.time()
        nearest, distance = pairwise_distances_argmin_min(X[X_i], X[S_i])
        # print("distance done")
        # eTime = time.time()
        # print("pairwise_distances_argmin_min", eTime - sTime)

        # sTime = time.time()
        # dis_map = spatial.distance.cdist(X[S_i], X[X_i], 'euclidean')
        # nearest = np.argmin(dis_map, axis=0)
        # distance = np.min(dis_map, axis=0)
        # eTime = time.time()
        # print("our", eTime - sTime)

        # 8. let rho_i be the smallest radius s.t. |B(S_i, X_i, rho_i)| >= beta|X_i|.
        rho_i = np.sort(distance)[int(np.ceil((len(X_i) - 1) * beta))]
        # print("fine rho_i")
        # Let C_i = B(S_i, X_i, rho_i)
        # foreach x\in X_i, assign sigma(x)=x
        # for each x \in X_i, assign weight w_x = |\sigma^{−1}(x)| and add (x, w_x) into Q
        idxs, counts = np.unique(nearest[distance <= rho_i], return_counts=True)
        w_i[idxs] = counts

        samples_.append(X[S_i])
        weights_.append(w_i)
        sample_indices_.append(S_i)
        assignment[X_i[distance <= rho_i]] += (nearest[distance <= rho_i]+offset)
        offset += len(idxs)

        # 9. for each x \in C_i, choose the point y \in S_i that minimizes d(x, y) and assign \sigma(x) = y
        X_i = X_i[distance > rho_i]

    # Augmented-Summary-Outliers (Algorithm 2 in the paper)
    # if self.augmented_ and len(sample_indices_) > 0:
    #     self.sample_indices_, self.weights_, self.samples_ = self.augmenting_(X, X_i, np.hstack(sample_indices_))
    #     return self

    # append the remained "outliers"
    samples_.append(X[X_i])
    weights_.append(np.ones(len(X_i)))
    sample_indices_.append(X_i)
    assignment[X_i]+=(np.arange(offset, offset+len(X_i)))

    # concatenate all S_i's to build the final summary
    samples_ = np.vstack(samples_)
    weights_ = np.hstack(weights_)
    sample_indices_ = np.hstack(sample_indices_)

    assert len(samples_) == len(weights_)
    return sample_indices_, assignment, weights_

def run(dataset, data, z):
    z_number = (int)(len(data) * 0.1)

    start_time = time.time()
    sample_indices_, assignment, weights_ = SummaryOutliers(data, 200, 1.0, 0.4)
    data = data[sample_indices_]
    summary_time = time.time() - start_time

    # print("start")
    # print(data.shape)
    # print("after summary",data.shape)

    # print('-' * 120)
    # print(dataset)
    # print(data.shape)
    ne_start = args.KNN_start
    ne_end = args.KNN_end + 1
    ne_step = args.KNN_step
    ne_list = list(range(ne_start, ne_end, ne_step))
    tree_depth_start = args.tree_depth_start
    tree_depth_end = args.tree_depth_end + 1
    if tree_depth_end == 1:
        tree_depth_end = math.ceil(math.log(len(data), 2))
    tree_depth_step = args.tree_depth_step
    tree_depth_list = list(range(tree_depth_start, tree_depth_end, tree_depth_step))
    _, gap = k_center_greedy(data, math.floor(np.log2(len(data))))
    gap_range = [gap]
    gap_range_number = 1
    result = pd.DataFrame(columns=['dataset', 'KNN', 'tree_depth', 'gap', 'prec', 'recall', 'f1', "roc", "ap", "time"])

    maximun_prec = 0
    gap_prec_max = 0
    ne_prec_max = 0
    tree_depth_prec_max = 0

    maximun_recall = 0
    gap_recall_max = 0
    ne_recall_max = 0
    tree_depth_recall_max = 0

    maximun_f1 = 0
    gap_f1_max = 0
    ne_f1_max = 0
    tree_depth_f1_max = 0

    maximun_roc = 0
    gap_roc_max = 0
    ne_roc_max = 0
    tree_depth_roc_max = 0

    maximun_ap = 0
    gap_ap_max = 0
    ne_ap_max = 0
    tree_depth_ap_max = 0

    all = len(ne_list) * len(tree_depth_list) * gap_range_number

    try:
        for p in range(len(ne_list)):
            # res_time = 0
            ne = ne_list[p]
            for q in range(len(tree_depth_list)):
                tree_depth = tree_depth_list[q]
                for l in range(len(gap_range)):
                    gap = gap_range[l]

                    s_time = time.time()
                    adj_mat = bulit_knn_adjm(data, ne, gap * np.std(data))
                    # adj_mat = bulit_knn_adjm(data, ne)
                    assert np.all(adj_mat == adj_mat.T)
                    tree = trans_to_tree(adj_mat, tree_depth)
                    tree = update_node(tree)
                    data_entory = all_entory(len(data), tree_depth, np.sum(adj_mat), tree)
                    e_time = time.time()

                    data_ent = np.ones(data.shape[0])
                    for i in range(data.shape[0]):
                        for j in range(tree_depth):
                            if (data_entory[i][tree_depth - j - 1] > 0.000000001):
                                data_ent[i] = data_entory[i][tree_depth - j - 1]
                                break

                    Dscores = MinMaxScaler().fit_transform(data_ent.reshape(-1, 1)).reshape(-1)

                    outlier_scores = Dscores[assignment]

                    pred_z = np.argsort(outlier_scores.reshape(-1))[len(assignment) - z_number:]

                    label = np.zeros(len(assignment))
                    pre_label = np.zeros(len(assignment))
                    pre_label[pred_z] = 1
                    label[z] = 1

                    f1Score = f1_score(label, pre_label, average='binary')
                    prec = len(np.intersect1d(pred_z, z)) / len(pred_z)
                    recall = len(np.intersect1d(pred_z, z)) / len(z)
                    roc = roc_auc_score(label, outlier_scores)
                    ap = average_precision_score(label, outlier_scores)

                    new = pd.DataFrame(
                        [[dataset, ne, tree_depth, gap, prec, recall, f1Score, roc, ap, e_time - s_time]],
                        columns=['dataset', 'KNN', 'tree_depth', 'gap', 'prec', 'recall', 'f1', "roc", "ap", "time"])
                    result = pd.concat([result, new])
                    if prec > maximun_prec:
                        maximun_prec = prec
                        ne_prec_max = ne
                        tree_depth_prec_max = tree_depth
                        gap_prec_max = gap
                    if recall > maximun_recall:
                        maximun_recall = recall
                        ne_recall_max = ne
                        tree_depth_recall_max = tree_depth
                        gap_recall_max = gap
                    if f1Score > maximun_f1:
                        maximun_f1 = f1Score
                        ne_f1_max = ne
                        tree_depth_f1_max = tree_depth
                        gap_f1_max = gap
                    if roc > maximun_roc:
                        maximun_roc = roc
                        ne_roc_max = ne
                        tree_depth_roc_max = tree_depth
                        gap_roc_max = gap
                    if ap > maximun_ap:
                        maximun_ap = ap
                        ne_ap_max = ne
                        tree_depth_ap_max = tree_depth
                        gap_ap_max = gap

                    # now = p * len(tree_depth_list) * gap_range_number + q * gap_range_number + l + 1
                    # res_time = (time.time() - start_time) * (all / now - 1)
                    # print("\r", "已测试完成KNN={},depth={},gap={}\t进度[{}]{:.5f}%({}/{}.耗时{:.2f}s,还需{:.2f}s)".format(
                    #     ne, tree_depth, gap,
                    #     '|' + '■|' * int(now / all * 10 + 1) + ' |' * (
                    #             10 - int(now / all * 10 + 1)), now / all * 100, now,
                    #     all, time.time() - start_time, res_time), end='',
                    #       flush=True)
                    # if res_time > 5400:
                    #     undo_dataset.append(dataset)
                    #     break
                # if res_time > 5400:
                #     break
            # if res_time > 5400:
            #     break
    except Exception as e:
        print(e)
    print("dataset:", dataset,
          "\nsummary_time:", summary_time,
          # "\ntime:", e_time - s_time,
          "\n最大prec:", maximun_prec, "\tKNN:", ne_prec_max, "\ttree_depth:", tree_depth_prec_max, "\tgap:", gap_prec_max,
          "\n最大recall:", maximun_recall, "\tKNN:", ne_recall_max, "\ttree_depth:", tree_depth_recall_max, "\tgap:", gap_recall_max,
          "\n最大f1:", maximun_f1, "\tKNN:", ne_f1_max, "\ttree_depth:", tree_depth_f1_max, "\tgap:", gap_f1_max,
          "\n最大roc:", maximun_roc, "\tKNN:", ne_roc_max, "\ttree_depth:", tree_depth_roc_max, "\tgap:", gap_roc_max,
          "\n最大ap:", maximun_ap, "\tKNN:", ne_ap_max, "\ttree_depth:", tree_depth_ap_max, "\tgap:", gap_ap_max)
    result.to_csv("../../result/outlier/ours/result_" + dataset + "_z=0.1_threshold_parallel_fast.csv", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='encoding k-means')
    parser.add_argument('--dataset', '-d', type=str, default=False, help='dataset')
    parser.add_argument('--datadir', '-dr', type=str, default=False, help='datasetdir')
    parser.add_argument('--KNN_start', '-ks', type=int, default=4, help='minimun k of KNN')
    parser.add_argument('--KNN_end', '-ke', type=int, default=4, help='maximun k of KNN')
    parser.add_argument('--KNN_step', '-kp', type=int, default=1, help='k step')
    parser.add_argument('--tree_depth_start', '-tds', type=int, default=2, help='minimun depth of encoding tree')
    parser.add_argument('--tree_depth_end', '-tde', type=int, default=0, help='maximun depth of encoding tree')
    parser.add_argument('--tree_depth_step', '-tdp', type=int, default=1, help='depth step')
    parser.add_argument('--gap_start', '-gs', type=float, default=1.0, help='minimun gap')
    parser.add_argument('--gap_end', '-ge', type=float, default=1.0, help='maximun gap')
    parser.add_argument('--gap_number', '-gn', type=int, default=1, help='gap number')
    args = parser.parse_args()
    dataset = args.dataset
    datadir = args.datadir
    np.seterr(divide='ignore', invalid='ignore')
    if not datadir:
        if dataset:
            # learn
            print("start read")
            x = np.memmap("/home/vuser/junyucode/"+dataset+".bvecs", dtype='uint8', mode='r')
            print("read done")
            d = x[:4].view('int32')[0]
            data = x.reshape(-1, d + 4)[:, 4:]
            data = np.array(data)
            print("transform done")
            # print(dataset[:-6], data.shape)
            data, z = add_noise(data, 5, 10000)
            datas = [[dataset, data, z]]
        else:

            dataset = 'random_generate'
            n_samples = 50000
            random_state = 170
            data, label = make_blobs(n_samples=n_samples, cluster_std=[1.5, 0.5, 1.5], random_state=random_state,
                                     center_box=[-5.0, 5.0])
            data, z = add_noise(data, 5, int(len(data) * 0.15))
            datas = [[dataset, data, z]]
    else:
        datas = []
        for dataset in os.listdir(datadir):
            if ".npz" in dataset :
                data = np.load(datadir + "/" + dataset, allow_pickle=True)
                data, label = data['X'], data['y']
                if data.shape[0] > 50000:
                    continue
                z = np.where(label == 1)[0]
                # data = pd.read_csv(datadir + "/" + dataset, header=None).to_numpy()
                # label = data[:, -1]
                # data = np.delete(data, -1, 1)
                # k = len(np.unique(label))
                datas.append([dataset[:-4], data, z])
                # print(dataset[:-4], data.shape)
            if ".txt" in dataset:
                data = pd.read_csv(datadir + "/" + dataset, header=None).to_numpy()
                label = data[:, -1]
                data = np.delete(data, -1, 1)
                z = np.where(label == np.max(label))[0]
                datas.append([dataset[:-4], data, z])
            # print(dataset[:-4], data.shape, len(z))
    undo_dataset = []
    print("start")
    # run(*(datas[0]))
    with Pool(15) as p:
        p.starmap(run, datas)
    print("done")

