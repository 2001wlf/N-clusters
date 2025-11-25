import math
import time

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.neighbors import KDTree


def bulit_graph_table(data, ne, threshold):
    adj_table = [[-1 for j in range(ne << 2)] for i in range(len(data))]
    adj_dis = [[0 for j in range(ne << 2)] for i in range(len(data))]
    length = [0 for i in range(len(data))]
    vol = [0 for i in range(len(data))]
    VOL = 0
    kd_tree = KDTree(data, leaf_size=20)
    try:
        dist, inds = kd_tree.query(data, k=ne+1)
    except:
        dist, inds = kd_tree.query(data, k=len(data))
    for m in range(len(inds)):
        ind = inds[m]
        i = m
        for n in range(len(ind[1:])):
            if n:
                if dist[m][n + 1] > threshold:
                    continue
            j = ind[n + 1]
            if j not in adj_table[i]:
                adj_table[i][length[i]] = j
                adj_dis[i][length[i]] = dist[m][n + 1]
                vol[i] += dist[m][n + 1]
                VOL += dist[m][n + 1]
                length[i] += 1
            if i not in adj_table[j]:
                adj_table[j][length[j]] = i
                adj_dis[j][length[j]] = dist[m][n + 1]
                vol[j] += dist[m][n + 1]
                VOL += dist[m][n + 1]
                length[j] += 1
    return adj_table, adj_dis, length, vol, VOL


def run(ne, gap, tree_depth, block):
    dataset = 'random_generate'
    n_samples = 10000
    random_state = 170
    data, label = make_blobs(n_samples=n_samples, cluster_std=[1.5, 0.5, 1.5], random_state=random_state,
                             center_box=(-5.0, 5.0))
    axis_d_min = np.min(data)
    axis_d_max = np.max(data)

    for i in range(0, data.shape[1]):
        data[:, i] = (data[:, i] - axis_d_min) / (4 * (axis_d_max - axis_d_min)) + 1 / 4
    k = len(np.unique(label))

    kmeans = KMeans(init='k-means++', n_clusters=k).fit(data)
    opt = kmeans.inertia_

    start_time = time.time()
    adj_table, adj_dis, length, vol, VOL = bulit_graph_table(data, ne, gap * math.sqrt(opt / len(data)))
    # 建树
    encoding_tree, structural_entropy, entropy, data_to_leaf = mltiview_to+codingtree(data, adj_table, adj_dis, length, vol, VOL, tree_depth, block)
    end_time = time.time()
    print("耗时:"+str(end_time-start_time)+"s")
    print("结构熵:"+str(entropy))
    depth = [0 for i in range(len(data))]
    for i, j in data_to_leaf.items():
        now = j
        i_depth = 0
        while now != -1:
            p = encoding_tree[now]
            i_depth += 1
            now = p
        depth[i] = i_depth
    print("树高:"+str(max(depth)))
    start_time = time.time()
    adj_table, adj_dis, length, vol, VOL = bulit_graph_table(data, ne, gap * math.sqrt(opt / len(data)))
    encoding_tree, structural_entropy, entropy, data_to_leaf = bulit_encoding_tree(data, adj_table, adj_dis, length,
                                                                                   vol, VOL, max(depth) - 3, block)
    end_time = time.time()
    print("耗时:"+str(end_time-start_time)+"s")
    print("结构熵:"+str(entropy))
    depth = [0 for i in range(len(data))]
    for i, j in data_to_leaf.items():
        now = j
        i_depth = 0
        while now != -1:
            p = encoding_tree[now]
            i_depth += 1
            now = p
        depth[i] = i_depth
    print("树高:"+str(max(depth)))
    return encoding_tree, structural_entropy

if __name__ == "__main__":
    ne = 5
    gap = 0.1
    tree_depth = 2
    block = 10
    run(ne, gap, tree_depth, block)
