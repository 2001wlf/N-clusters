import numpy as np
from scipy.spatial import KDTree

class DPC:
    def __init__(self, d_cut):
        """
        初始化 DPC 模型
        :param d_cut: 局部密度计算的截断距离
        """
        self.d_cut = d_cut
        self.rho = None   # 局部密度
        self.delta = None  # 依赖距离
        self.cluster_centers = None
        self.labels = None

    def fit(self, X,k):
        """
        训练 DPC 模型
        :param X: 输入数据集 (numpy 数组)
        """
        n = X.shape[0]
        self.delta = np.full(n, np.inf)
        nearest_higher = np.full(n, -1)  # 记录最近高密度点的索引
        #print(self.rho)
        #计算局部密度
        tree = KDTree(X)
        self.rho = np.zeros(n)
        for i in range(n):
            # 查询距离小于 d_cut 的所有点
            indices = tree.query_ball_point(X[i], r=self.d_cut)
            # 减去自身的计数
            self.rho[i] = len(indices) - 1
        
        
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
        """
        返回聚类标签
        """
        return self.labels