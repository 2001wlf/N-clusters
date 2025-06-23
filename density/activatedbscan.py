import numpy as np

class AdaptiveDBSCAN:
    def __init__(self, eps_min=0.05, eps_max=0.3, min_samples=5):
        """
        eps_min: 密度最高（归一化后d接近1）的点对应的最小eps
        eps_max: 密度最低（归一化后d接近0）的点对应的最大eps
        min_samples: 判断核心点所需的最小邻居数，保持全局不变
        """
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.min_samples = min_samples
        self.labels_ = None

    def density_to_eps(self, d):
        """
        将归一化后的密度 d 映射为局部的 eps 参数
        设定原则：密度高 -> eps 小，密度低 -> eps 大
        """
        return self.eps_max - d * (self.eps_max - self.eps_min)

    def fit(self, X, density):
        """
        X: 数据点，形状 (n_samples, n_features)（例如二维坐标）
        density: 每个点的密度值，要求已经归一化到 [0, 1]
        """
        n_samples = X.shape[0]
        # 根据每个点的密度计算局部 eps 值
        local_eps = np.array([self.density_to_eps(d) for d in density])
        
        labels = np.full(n_samples, -1)  # 初始化所有点标签为 -1（噪声）
        visited = np.zeros(n_samples, dtype=bool)
        cluster_id = 0
        
        for i in range(n_samples):
            if visited[i]:
                continue
            visited[i] = True
            # 用局部 eps 进行邻居查询：采用条件 distance(i,j) <= min(local_eps[i], local_eps[j])
            neighbors = self.region_query(X, local_eps, i)
            if len(neighbors) < self.min_samples:
                labels[i] = -1  # 如果邻居不足则视为噪声
            else:
                self.expand_cluster(X, labels, visited, i, neighbors, cluster_id, local_eps)
                cluster_id += 1
        self.labels_ = labels
        return self

    def region_query(self, X, local_eps, point_idx):
        """
        对于给定的点 point_idx，返回在它局部 eps 内的所有点索引。
        注意：这里的判断条件是：两点之间的距离 <= min(local_eps[point_idx], local_eps[j])
        """
        distances = np.linalg.norm(X - X[point_idx], axis=1)
        eps_i = local_eps[point_idx]
        neighbors = []
        for j in range(X.shape[0]):
            if distances[j] <= min(eps_i, local_eps[j]):
                neighbors.append(j)
        return neighbors

    def expand_cluster(self, X, labels, visited, point_idx, neighbors, cluster_id, local_eps):
        """
        将满足条件的点加入到当前簇中
        """
        labels[point_idx] = cluster_id
        i = 0
        while i < len(neighbors):
            j = neighbors[i]
            if not visited[j]:
                visited[j] = True
                new_neighbors = self.region_query(X, local_eps, j)
                if len(new_neighbors) >= self.min_samples:
                    # 合并新的邻居（避免重复添加）
                    for neighbor in new_neighbors:
                        if neighbor not in neighbors:
                            neighbors.append(neighbor)
            if labels[j] == -1:
                labels[j] = cluster_id
            i += 1