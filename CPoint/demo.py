import os
import glob
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from util3 import kmeans_penalty_feature
def deal_data(true_data,true_label,dataset_name):
    print(f"Processing dataset: {dataset_name}")
    
    # X: 数据点
    X = true_data
    if X.ndim != 2:
        X = X.reshape(-1, X.shape[-1])

    # 用真实标签种类数作为聚类数
    n_clusters = len(np.unique(true_label))

    # 1) k-means++ 聚类
    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=10,
        random_state=0,
    )
    kmeans.fit(X)

    centers = kmeans.cluster_centers_      # (K, d)
    labels = kmeans.labels_                # (N,)

    # 2) 计算每个点到所属簇中心的距离
    dists = np.linalg.norm(X - centers[labels], axis=1)

    # 3) 计算 kmeans_penalty_feature 特征
    penalty_feat = kmeans_penalty_feature(X, centers, labels)
    penalty_feat = np.asarray(penalty_feat).reshape(-1)

    # 4) 打印前 100 个点
    n_print = min(10, X.shape[0])
    print("dist_to_center\tpenalty_feature")
    for i in range(n_print):
        print(f"{dists[i]:.6f}\t{penalty_feat[i]:.6f}")
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'mydata2', 'transformed_dataset_csv')
    filename=""
    cnt=1
    # Iterate through all .txt files in the training folder
    for filepath in glob.glob(os.path.join(data_dir, '*.txt')):
        if cnt>=20:
            break
        filename = os.path.basename(filepath)    # 只保留 “xxx.txt”
        filename = os.path.splitext(filename)[0]  # 不带后缀的文件名 “xxx”
        #print(filename)
        df = pd.read_csv(filepath, header=None)  
        arr = df.values
        # Ensure 2D array
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        X = arr[:, :-1]
        y = arr[:, -1].astype(int)
        
        deal_data(X,y,filename)

        cnt+=1