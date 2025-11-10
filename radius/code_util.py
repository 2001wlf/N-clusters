
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
from scipy.spatial import distance
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist


import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def eval_topk_vs_kmeans(X, pred_scores, k, match_eps=0.05):
    """
    X: (N,2) 已缩放到[0,1]
    pred_scores: (N,) 模型半径得分（未归一化也可）
    k: 期望中心数
    match_eps: 判定“命中参考中心”的阈值(欧氏距离)

    返回：
      centers_idx: 你取的Top-k索引
      kmeans_centers: 参考中心坐标 (k,2)
      hit_ratio: Top-k中有多少能和参考中心一一匹配（<=eps）
      assign_acc: 用Top-k做“最近中心分配”的聚类ACC（和KMeans标签比）
    """
    # 1) 参考：KMeans
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    ref_centers = km.cluster_centers_
    ref_labels = km.labels_

    # 2) 你的做法：Top-k 作为中心
    centers_idx = np.argsort(pred_scores)[-k:]
    centers = X[centers_idx]

    # 3) 匹配命中率（中心对中心）
    cost = cdist(centers, ref_centers)      # (k,k)
    r, c = linear_sum_assignment(cost)      # 最小总距离匹配
    hits = (cost[r, c] <= match_eps).sum()
    hit_ratio = hits / k

    # 4) 用Top-k中心做一次“最近中心分配”，对比KMeans标签的一致性
    dist_to_centers = cdist(X, centers)
    pred_labels = np.argmin(dist_to_centers, axis=1)

    # 为了公平，用匈牙利把pred_labels映射到ref_labels的标签空间
    D = max(pred_labels.max(), ref_labels.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(pred_labels.size):
        w[pred_labels[i], ref_labels[i]] += 1
    rr, cc = linear_sum_assignment(w.max() - w)
    assign_acc = w[rr, cc].sum() / pred_labels.size

    return centers_idx, ref_centers, float(hit_ratio), float(assign_acc)

def kmedoid_from_radii(radii, X, k, max_iter=10):
    """
    Initialize k medoids by selecting top-k radii points, then iteratively
    update by choosing the medoid (min-sum-of-distances) in each cluster.
    Returns final labels and medoid indices.
    """
    # distance matrix
    dist_mat = cdist(X, X)
    # initial medoids: indices of top-k radii
    medoids = np.argsort(radii)[-k:]
    for _ in range(max_iter):
        # assign each point to nearest medoid
        labels = np.argmin(dist_mat[:, medoids], axis=1)
        new_medoids = []
        # update medoids for each cluster
        for ci in range(k):
            members = np.where(labels == ci)[0]
            if len(members) == 0:
                new_medoids.append(medoids[ci])
            else:
                subdist = dist_mat[np.ix_(members, members)]
                # sum of distances for each candidate
                sums = subdist.sum(axis=1)
                # pick member with minimal total distance
                new_medoids.append(members[np.argmin(sums)])
        new_medoids = np.array(new_medoids)
        # check for convergence
        if np.array_equal(new_medoids, medoids):
            break
        medoids = new_medoids
    # final assignment
    labels = np.argmin(dist_mat[:, medoids], axis=1)
    return labels, medoids
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