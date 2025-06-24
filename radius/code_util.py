import os
import argparse
import numpy as np
from utils.data_loader import DataLoader
import glob
from tqdm import trange
from scipy.optimize import linear_sum_assignment as linear_assignment
from net.sgcn_model import SparseGCNModel
from scipy.spatial import distance
from sklearn.utils.class_weight import compute_class_weight
import torch
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from torch.autograd import Variable
from sklearn.cluster import KMeans
from util3 import raidusQuery
import pickle
from generate_distributions3 import generate_datasets
import pandas as pd
from scipy.optimize import linear_sum_assignment
import numpy as np

import os
import glob
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist
def variable_eps_dbscan(X, radii, min_samples=5):
    """
    Variable-epsilon DBSCAN.

    参数
    ----
    X : ndarray, shape (n_samples, n_features)
        样本点坐标。
    radii : ndarray, shape (n_samples,)
        第 i 个点的 ε_i（可变半径）。
    min_samples : int
        构成“核心点”所需的最少邻居数量。

    返回
    ----
    labels : ndarray, shape (n_samples,)
        聚类标签，从 0 开始，噪声标签为 -1。
    """
    n = X.shape[0]
    labels = np.full(n, -1, dtype=int)   # 初始化所有点为噪声
    visited = np.zeros(n, dtype=bool)    # 标记是否已访问
    cluster_id = 0

    # 预计算距离矩阵（可用 KD-Tree 或者分块加速）
    # dist_matrix[i, j] = ||X[i] - X[j]||
    diff = X[:, None, :] - X[None, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True

        # 找到 i 的邻居：距离 <= radii[i]
        neighbors = np.where(dist_matrix[i] <= radii[i])[0]

        # 如果邻居太少，则 i 保持噪声
        if neighbors.size < min_samples:
            labels[i] = -1
        else:
            # 新簇
            labels[i] = cluster_id
            # 用一个队列 seeds 扩展
            seeds = list(neighbors.tolist())
            while seeds:
                j = seeds.pop(0)
                if not visited[j]:
                    visited[j] = True
                    # 以 j 为中心再找一次可变邻居
                    j_neighbors = np.where(dist_matrix[j] <= radii[j])[0]
                    # 如果 j 是核心点，将新邻居加入扩展队列
                    if j_neighbors.size >= min_samples:
                        for nb in j_neighbors:
                            if not visited[nb]:
                                seeds.append(int(nb))
                # 如果 j 还没被分配到任何簇（-1），就把它归到当前簇
                if labels[j] == -1:
                    labels[j] = cluster_id

            cluster_id += 1

    return labels
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
def code_for_up(true_data,true_label,k,radii):
    maximun_acc=0
    maximun_acc_index=[]
    radii = radii.reshape(-1, 1)
    data_ent = radii

    data=true_data
    data_entory=radii
    label=true_label

    # min_gap = find_min_gap(data_ent)
    data_ent_uniform = make_uniform(np.min(data_ent), np.max(data_ent), 19, data_ent)

    data_ = np.hstack((data,data_entory))

    
    data_ = MinMaxScaler().fit_transform(data_)
    # data_[:,2] = data_[:,2]*2

    kmeans_2 = KMeans(init='k-means++', n_clusters=k).fit(data_entory)
    # a = kmeans_2.labels_
    ACC_2 = round(acc(label, kmeans_2.labels_), 4)
    note = 'only entroy_list'

    # kmeans_3 = KMeans(init='k-means++', n_clusters=k).fit((data_entory - np.min(data_entory,axis=0))/(np.max(data_entory,axis=0)-np.min(data_entory,axis=0)))
    kmeans_3 = KMeans(init='k-means++', n_clusters=k).fit(MinMaxScaler().fit_transform(data_entory))
    ACC_3 = round(acc(label, kmeans_3.labels_), 4)
    note = 'only normalized entroy_list'


    kmeans_4 = KMeans(init='k-means++', n_clusters=k).fit(data_ent)
    ACC_4 = round(acc(label, kmeans_4.labels_), 4)
    note = 'only entroy'

    # kmeans_5 = KMeans(init='k-means++', n_clusters=k).fit((data_ent - np.min(data_ent,axis=0))/(np.max(data_ent,axis=0)-np.min(data_ent,axis=0)))
    kmeans_5 = KMeans(init='k-means++', n_clusters=k).fit(MinMaxScaler().fit_transform(data_ent))
    ACC_5 = round(acc(label, kmeans_5.labels_), 4)
    # np.savetxt("rings_ent.txt",data_ent,fmt="%f")
    note = 'only normalized entroy'


    kmeans_6 = KMeans(init='k-means++', n_clusters=k).fit(data_)
    ACC_6 = round(acc(label, kmeans_6.labels_), 4)
    note = 'normalized data with entroy_list'

    data_[:, -1] = data_[:, -1] * 2

    kmeans_7 = KMeans(init='k-means++', n_clusters=k).fit(data_)
    ACC_7 = round(acc(label, kmeans_7.labels_), 4)
    note = 'normalized data with twice entroy_list'
    
    data_ = np.hstack((data, data_ent))
    

    kmeans_9 = KMeans(init='k-means++', n_clusters=k).fit(MinMaxScaler().fit_transform(data_))
    ACC_9 = round(acc(label, kmeans_9.labels_), 4)

    note = 'normalized data with entroy'
    
    

    data_ent_ = MinMaxScaler().fit_transform(data_ent) + 1e-15
    data_ = np.hstack((data, np.log(data_ent_)))
    
    data_ = MinMaxScaler().fit_transform(data_)
    kmeans_11 = KMeans(init='k-means++', n_clusters=k).fit(data_)
    # kmeans_ = KMEANS().kmeans_plusplus(k,(data_ - np.min(data_, axis=0)) / (np.max(data_, axis=0) - np.min(data_, axis=0)))
    ACC_11 = round(acc(label, kmeans_11.labels_), 4)
    
    note = 'normalized data with log entroy'
    

    

    data_ = np.hstack((data, data_ent_uniform))
    # data_ = (data_ - np.min(data_, axis=0)) / (np.max(data_, axis=0) - np.min(data_, axis=0))
    # data_[np.isnan(data_)] = 0
    #print("yes")
    kmeans_12 = KMeans(init='k-means++', n_clusters=k).fit(MinMaxScaler().fit_transform(data_))
    ACC_12 = round(acc(label, kmeans_12.labels_), 4)

    note = 'normalized data with uniform entroy'
    
    #print("yes")
    ACC_list = np.array([ACC_2, ACC_3, ACC_4, ACC_5, ACC_6, ACC_7, ACC_9, ACC_11, ACC_12])
    ACC_max = np.max(ACC_list)
    ACC_max_index = np.argmax(ACC_list)

    if ACC_max>maximun_acc:
        maximun_acc = ACC_max
        maximun_acc_index = ACC_max_index

    print("\n最大ACC:", maximun_acc, "\t方法:", maximun_acc_index)
    return maximun_acc,maximun_acc_index  