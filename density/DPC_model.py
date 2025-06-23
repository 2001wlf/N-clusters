import numpy as np
from pyclustering.container.kdtree import kdtree as KDTree

class DensityPeaksClustering:
    def __init__(self, rho):
        """
        初始化 DPC 模型
        :param rho: 预先计算好的局部密度，长度 = 样本数
        """
        self.rho = rho
        self.delta = None
        self.cluster_centers = None
        self.labels = None

    def fit(self, X, k):
        """
        训练 DPC 模型
        :param X: 输入数据集 (numpy 数组)，形状 (n_samples, n_features)
        :param k: 聚类数
        """
        n, dim = X.shape
        self.delta = np.full(n, np.inf)
        nearest_higher = np.full(n, -1)

        # 1. 按密度降序排序
        sorted_idx = np.argsort(-self.rho)

        # 2. 构建空的 KD-Tree，用来动态插入已处理点
        tree = KDTree()
        # 3. 先把密度最高的点插入树，并留作 processed
        first = sorted_idx[0]
        tree.insert(X[first].tolist(), first)
        # processed_points 不再用于距离计算，但保留以备后续逻辑可追溯
        processed_points = [first]

        # 4. 对剩余点按密度降序，查询最近高密度点并插入树
        for idx in sorted_idx[1:]:
            # 动态查询：在半径无限制下，返回 (node, distance)
            dist,node= tree.find_nearest_dist_node(
                X[idx].tolist(),
                float('inf'),
                retdistance=True
            )
            # node.payload 就是我们插入时传入的“点索引”
            nearest_higher[idx] = node.payload
            self.delta[idx] = dist

            # 再把当前点插进去，供后续点查询
            tree.insert(X[idx].tolist(), idx)
            processed_points.append(idx)

        # 5. 最高密度点没有更高密度邻居，定义为 δ 的最大值
        self.delta[first] = np.max(self.delta)

        # 6. 选簇中心：取 δ 排名前 k 的点
        #    （这里用简单的分位数阈值，也可直接取 top-k）
        threshold = np.percentile(self.delta, (1 - k/n) * 100)
        self.cluster_centers = np.where(self.delta > threshold)[0]

        # 7. 标记簇中心
        self.labels = np.full(n, -1, dtype=int)
        for cid, center in enumerate(self.cluster_centers):
            self.labels[center] = cid

        # 8. 依次按密度降序把其它点“继承”最近高密度点的标签
        for idx in sorted_idx:
            if self.labels[idx] == -1:
                self.labels[idx] = self.labels[nearest_higher[idx]]

        return self.labels

    def predict(self):
        """
        返回聚类标签
        """
        return self.labels






"""import numpy as np

import pyclustering.container.kdtree as KDTree

class DensityPeaksClustering:
    def __init__(self, rho):
        self.d_cut = None
        self.rho = rho   # 局部密度
        self.delta = None  # 依赖距离
        self.cluster_centers = None
        self.labels = None

    def fit(self, X,k):
        n = X.shape[0]
        self.delta = np.full(n, np.inf)
        nearest_higher = np.full(n, -1)  # 记录最近高密度点的索引
        #print(self.rho)
        # **🔹 按密度降序排序**
        sorted_indices = np.argsort(-self.rho)  
        processed_points = [sorted_indices[0]]  # 先存入密度最高的点
        
        for i in range(1, n):
            cur_idx = sorted_indices[i]

            # **遍历 processed_points，找到最近的高密度点**
            min_dist = np.inf
            nearest_higher_idx = -1
            for prev_idx in processed_points:
                dist = np.linalg.norm(X[cur_idx] - X[prev_idx])  # 计算欧式距离
                if dist < min_dist:
                    min_dist = dist
                    nearest_higher_idx = prev_idx

            # **更新 delta 和最近高密度点**
            self.delta[cur_idx] = min_dist
            nearest_higher[cur_idx] = nearest_higher_idx

            # **将当前点加入 processed_points**
            processed_points.append(cur_idx)

        # **🔹 处理密度最高的点（没有高密度邻居）**
        self.delta[sorted_indices[0]] = np.max(self.delta)

        # **🔹 选择簇中心**
        delta_threshold = np.percentile(self.delta, (1-k/n)*100)  # 选择前 10% 作为簇中心
        self.cluster_centers = np.where(self.delta > delta_threshold)[0]

        # **🔹 簇分配**
        self.labels = np.full(n, -1)  # 初始化标签
        for i, center in enumerate(self.cluster_centers):
            self.labels[center] = i
            #print(i)  # 簇中心赋予唯一标签
        
        # **🔹 传播标签**
        for idx in sorted_indices:
            if self.labels[idx] == -1:
                self.labels[idx] = self.labels[nearest_higher[idx]]
                #print(self.labels[idx])
        return self.labels
    def predict(self):
        return self.labels
"""

