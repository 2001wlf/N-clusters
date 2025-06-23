import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
import pickle
from sklearn.metrics import adjusted_rand_score
from scipy.optimize import linear_sum_assignment as linear_assignment
import matplotlib.pyplot as plt
from tqdm import trange
import pandas as pd
from DPC_model import DensityPeaksClustering
from DPC_origin import DPC
import numpy as np
from sklearn.metrics import silhouette_score
from generate_distributions import generate_datasets
from mytest import entropy_model
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import davies_bouldin_score
from sklearn.datasets import load_wine
from sklearn.datasets import load_iris
import sys
import os
from sklearn.metrics import calinski_harabasz_score
from mydata.generate_dataset import generate_smile,generate_parabola
from mydata.generate_dataset import make_nested_squares,make_spiral
import time
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
def cal_true(y_true,y_pred):
    ari = adjusted_rand_score(y_true, y_pred)
    print(f"ARI: {ari:.4f}")
    nmi = normalized_mutual_info_score(y_true, y_pred)
    print(f"NMI: {nmi:.4f}")
    print(f"ACC: {acc(y_true,y_pred):.4f}")
def draw_DBscan(labels,data,type):
    # 假设 data 是形状为 (n_points, 2) 的数组，labels 是 DBSCAN 聚类结果
    unique_labels = np.unique(labels)
    print(unique_labels)

    #sil_score = silhouette_score(data, labels)
    #print(f"Silhouette Score: {sil_score:.3f}")
# 为每个聚类分配颜色
    #db_score = davies_bouldin_score(data, labels)
    #print(f"Davies-Bouldin Score: {db_score:.3f}")
    #ch_score = calinski_harabasz_score(data, labels)
    #print(f"Calinski-Harabasz Score: {ch_score:.3f}")
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    plt.figure(figsize=(8, 6))
    for k, col in zip(unique_labels, colors):
        if k == -1:
        # 噪声点用黑色显示
            col = 'k'
    # 选取属于当前簇的所有点
        class_member_mask = (labels == k)
        xy = data[class_member_mask]
        plt.scatter(xy[:, 0], xy[:, 1], c=[col], edgecolors='k', s=50, label=f'Cluster {k}')

    plt.title(f" {type} Clustering Result")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    #plt.show()
    plt.savefig(f"The {type}.png",format="png",dpi=300)
    plt.close()
    return
def DPC_model(dataset,data,label):
    n_node=len(data)
    k=len(np.unique(label))

    generate_datasets(data,label)
    entropy_model(n_node)
    dataset_name=dataset
    dataset = pickle.load(open( f"mydata/val/{n_node}.pkl", "rb"))
    eval_batch_size=1
    for eval_batch in trange(1 // eval_batch_size):
        node_feat=dataset["node_feat"][eval_batch*eval_batch_size:(eval_batch+1)*eval_batch_size]
        data=node_feat[0]
        density=dataset["density_feat"][eval_batch*eval_batch_size]
        with open(f"cmps/{n_node}/"+str(eval_batch)+"_pred.txt","r",encoding="utf-8") as f:
            pred=f.read()
            pred=pred.strip().split('\n')
            #pred=[x for x in pred if x.strip()!=' ']
            pred=np.array([float(x)*1000 for x in pred if x.strip() != ''])
        #features = np.hstack((data, pred.reshape(-1, 1)))
        # 标准化特征
        #features_scaled = StandardScaler().fit_transform(features)
        features=data
        features_scaled = StandardScaler().fit_transform(features)
        # 使用 DBSCAN 进行聚类
        
        rad=1.414/2
        rad=rad*1/2
        #db = DBSCAN(eps=rad, min_samples=1).fit(features_scaled)
        #labels = db.labels_
        
        #print("\n",len(np.unique(labels)))
        #k=len(np.unique(labels))
        
        #iris = datasets.load_iris()
        #label=iris.target
        #draw_DBscan(label,data,"true-label")
        #cal_true(label,label);
        #k = len(np.unique(label))
        print(dataset_name)
        
        print("model")
        start_time = time.time()
        db = DensityPeaksClustering(pred).fit(data,k)
        labels = db
        draw_DBscan(labels,data,"model")
        cal_true(label,labels);
        elapsed_time = time.time() - start_time
        print(f"Model clustering time: {elapsed_time:.4f} seconds")

        print("DPC")
        start_time = time.time()       
        db=DPC(d_cut=rad).fit(data,k)
        labels=db
        draw_DBscan(labels,data,"DPC-origin")
        cal_true(label,labels);
        elapsed_time = time.time() - start_time
        print(f"DPC clustering time: {elapsed_time:.4f} seconds")
        
        print("DBSCAN")
        # 使用 DBSCAN 进行聚类
        start_time = time.time()  
        db = DBSCAN(eps=rad, min_samples=1).fit(features_scaled)
        labels = db.labels_
        draw_DBscan(labels,data,"origin")   
        cal_true(label,labels)
        elapsed_time = time.time() - start_time
        print(f"DBSCAN clustering time: {elapsed_time:.4f} seconds")
data_dir = "C:/Users/10998/Desktop/N-clusters/density/mydata/transformed_dataset_csv"  # 根据你的实际路径修改
datasets = []
if __name__ == '__main__':
    #iris=load_iris()
    #data = iris.data[:, :2]  # 只取前两列特征
    #label = iris.target
    #DPC_model("iris", data, label)
    #iris=load_wine()
    #data = iris.data[:, :2]  # 只取前两列特征
    #label = iris.target
    #DPC_model("wine", data, label)
    data,label=generate_smile(5000)
    DPC_model("smile",data,label)
    
    data,label=generate_parabola(5000);
    DPC_model("parabola",data,label)

    data,label=make_spiral(5000)
    DPC_model("spiral",data,label)

    data,label=make_nested_squares(5000)
    DPC_model("nested_squares",data,label)
    #给出数据集
    #for file_name in os.listdir(data_dir):
     #   if file_name.endswith(".txt"):
      #      file_path = os.path.join(data_dir, file_name)
       #     df = pd.read_csv(file_path, header=None, sep=r'\s+|,', engine='python')
        #    data = df.iloc[:, :-1].to_numpy()  # 所有特征列
         #   if data.shape[1]>2:
          #      data=data[:,:2]
           # label = df.iloc[:, -1].to_numpy()  # 最后一列是标签
            #DPC_model(file_name,data,label)