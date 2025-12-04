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
from util3 import raidusQuery,compute_cluster_size_density_radius
import matplotlib.pyplot as plt
import pickle
from generate_distributions3 import generate_datasets
import pandas as pd
from scipy.optimize import linear_sum_assignment
import numpy as np
import os
import glob
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist
from code_util import code_for_up,variable_eps_dbscan,kmedoid_from_radii,eval_topk_vs_kmeans
def coverage_sampling(radii, X, k):
    """
    Uniformly sample k centers based on radii and assign clusters by spatial distance:
    - Sample a center uniformly from remaining points.
    - Assign all points within the center's radius (Euclidean distance) to this cluster.
    - Remove assigned points and repeat.
    - Remaining unassigned points are noise.
    """
    n = radii.shape[0]
    assignments = np.full(n, -1, dtype=int)  # -1 means unassigned/noise
    available = list(range(n))
    centers = []
    for cluster_id in range(k):
        if not available:
            break
        # select the available point with the largest radius
        idx = max(available, key=lambda j: radii[j])
        centers.append(idx)
        threshold = radii[idx]
        center_point = X[idx]
        members = [j for j in available if np.linalg.norm(X[j] - center_point) <= threshold]
        for j in members:
            assignments[j] = cluster_id
        # update available to those still unassigned
        available = [j for j in available if assignments[j] == -1]
    noise = [j for j, a in enumerate(assignments) if a == -1]
    return assignments, centers, noise

def compute_accuracy(y_true, y_pred,noise_label=-1):
    """
    Calculate clustering accuracy by finding the optimal label mapping.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Create confusion matrix
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    # Solve assignment problem (maximize trace)
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() / y_pred.size

def compute_accuracy_filtered(y_true, y_pred, noise_label=-1):
    """
    Calculate clustering accuracy by optimal label mapping, excluding noise points.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Filter out noise points
    mask = (y_pred != noise_label)
    if mask.sum() == 0:
        return 0.0
    return compute_accuracy(y_true[mask], y_pred[mask])

def radius_model(my_node):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--eval_interval', type=int, default=1, help='')
    parser.add_argument('--eval_batch_size', type=int, default=1, help='')
    parser.add_argument('--eval_file_path', default='C:/Users/10998/Desktop/N-clusters/range/mydata2/val', help='')
    parser.add_argument('--model_path', type=str, default='C:/Users/10998/Desktop/N-clusters/range/saved/radius_centralpoint_uniform/5.pt', help='')
    args = parser.parse_args()
    edge_cw = None
    n_edges = 20
    n_edges = min(n_edges, my_node - 1)
    net = SparseGCNModel()
    net.cuda()
    saved = torch.load(args.model_path)
    net.load_state_dict(saved["model"])

    eval_results = []
    for n_node in [my_node]:
        dataset = pickle.load(open(args.eval_file_path + "/" + str(n_node) + ".pkl", "rb"))
        dataset_rank = []
        dataset_norms = []
        os.makedirs("cmps/{}/".format(n_node),exist_ok=True)
        for eval_batch in trange(1 // args.eval_batch_size):
            node_feat = dataset["node_feat"][eval_batch * args.eval_batch_size:(eval_batch + 1) * args.eval_batch_size]
            print(node_feat.shape)
            print(node_feat.shape[0]);
            print(node_feat.shape[1]);
            edge_feat = dataset["edge_feat"][eval_batch * args.eval_batch_size:(eval_batch + 1) * args.eval_batch_size]
            edge_index = dataset["edge_index"][eval_batch * args.eval_batch_size:(eval_batch + 1) * args.eval_batch_size]
            inverse_edge_index = dataset["inverse_edge_index"][eval_batch * args.eval_batch_size:(eval_batch + 1) * args.eval_batch_size]
            label = dataset["density_feat"][eval_batch * args.eval_batch_size:(eval_batch + 1) * args.eval_batch_size]
            with torch.no_grad():
                node_feat = Variable(torch.FloatTensor(node_feat).type(torch.cuda.FloatTensor), requires_grad=False) # B x 100 x 2
                edge_feat = Variable(torch.FloatTensor(edge_feat).type(torch.cuda.FloatTensor), requires_grad=False).view(args.eval_batch_size, -1, 1) # B x 1000 x 2
                label = Variable(torch.FloatTensor(label).type(torch.cuda.FloatTensor), requires_grad=False).view(args.eval_batch_size, -1) # B x 1000
                edge_index = Variable(torch.FloatTensor(edge_index).type(torch.cuda.FloatTensor), requires_grad=False).view(args.eval_batch_size, -1) # B x 1000
                inverse_edge_index = Variable(torch.FloatTensor(inverse_edge_index).type(torch.cuda.FloatTensor), requires_grad=False).view(args.eval_batch_size, -1) # B x 1000
                n_nodes = node_feat.size(1)
                #print(node_feat.shape)
                y_edges, loss_nodes, y_nodes = net.forward(node_feat, edge_feat, edge_index, inverse_edge_index, label, edge_cw, n_edges)
                loss_nodes = loss_nodes.mean()

                y_edges = y_edges.detach().cpu().numpy()
                label = label.cpu().numpy()
                np.savetxt("cmps/{}/{}_label.txt".format(n_node,eval_batch),label[0],fmt='%.4f')
                np.savetxt("cmps/{}/{}_pred.txt".format(n_node,eval_batch),list(y_nodes[0].flatten().cpu()),fmt='%.4f')
    print("eval: n = 100 | {}".format(loss_nodes))
#entropy_model(300)
#test
def make_uniform(start, end, length, np_nums):
    s = np.linspace(start, end, length).reshape(-1,1)
    dis_map = distance.cdist(np_nums.reshape(-1,1), s)
    assignment = np.argmin(dis_map,axis=1)
    new_nums = s[assignment]
    return new_nums
def quantize_to_deciles(values, vmin=None, vmax=None):
    """
    将一维连续数值划分为 10 个档位，返回 1~10 的整数标签。
    默认用自身的 min/max 做线性归一化。
    """
    vals = np.asarray(values, dtype=float)

    if vmin is None:
        vmin = vals.min()
    if vmax is None:
        vmax = vals.max()

    if vmax - vmin < 1e-12:
        # 所有值几乎一样，就都给 5 分（中间档）
        return np.ones_like(vals, dtype=int) * 5

    norm = (vals - vmin) / (vmax - vmin)  # 映射到 [0,1]
    deciles = np.floor(norm * 10).astype(int) + 1  # 1~10
    deciles = np.clip(deciles, 1, 10)
    return deciles
def draw_picture(centers,true_data,pred_label, dataset_name):

    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(pred_label)
    N=true_data.shape[0]
    for label in unique_labels:
        indices = np.where(pred_label == label)[0]
        plt.scatter(true_data[indices, 0], true_data[indices, 1], label=f'Cluster {label}')
    plt.scatter(true_data[centers, 0], true_data[centers, 1], color='black', marker='x', s=100, label='Centers')
    plt.title(f'Clustering Results for {dataset_name}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid()
    plt.savefig(f'cmps/{N}/{dataset_name}_clustering_results.png')
    #plt.show()
def draw_zone_picture(true_data, pred_zone, dataset_name):
    """
    绘制模型预测的 zone（1~10 档），按渐变色显示。
    
    true_data: (N,2)
    pred_zone: (N,) 取值 1~10
    """

    plt.figure(figsize=(8, 6))
    N = true_data.shape[0]

    # 选择一种渐变色（viridis/plasma/coolwarm/turbo 等）
    cmap = plt.cm.viridis  

    # 将 pred_zone 映射到 0~1（给 colormap 用）
    # pred_zone 在 1~10 → 映射为 0~1
    norm_vals = (pred_zone - pred_zone.min()) / (pred_zone.max() - pred_zone.min())

    # 绘制点
    plt.scatter(true_data[:, 0], true_data[:, 1],
                c=norm_vals,
                cmap=cmap,
                s=30,
                alpha=0.9)

    # 加一个颜色条用于标注 zone 数值
    cbar = plt.colorbar()
    cbar.set_label("predicted zone (1–10)")
    cbar.set_ticks(np.linspace(0, 1, 10))
    cbar.set_ticklabels([str(i) for i in range(1, 11)])

    plt.title(f"Predicted Zone Visualization for {dataset_name}")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)

    # 保存图片
    save_dir = f"cmps/{N}"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/{dataset_name}_zone_plot.png")
    # plt.show()
    plt.close()
origin_radius_list=[]
model_radius_list=[]
model_acc_simple_list = []   # 上面那次（radii→kmedoid，max_iter=0）的ACC
model_k_list = []            # 上面那次得到的簇数
def read_data(true_data,true_label,dataset_name):
    """
    Load dataset from the local mydata2/train folder under the script's directory.
    Assumes each .txt file is whitespace-delimited, with the last column as the label.
    """
    #wine "
    #print(f"dataset: {dataset_name}")
    true_data = true_data[:,:2]
    scaler = MinMaxScaler(feature_range=(0, 1))
    true_data = scaler.fit_transform(true_data)
    
    generate_datasets(true_data, true_label)
    #true_label = wine.target

    k = len(np.unique(true_label))
    kmeans = KMeans(init='k-means++',n_clusters=k, random_state=0).fit(true_data)

    feats=compute_cluster_size_density_radius(true_data,kmeans.labels_,k)
    origin_radius_list.append(feats[:,2])
    init_acc = compute_accuracy(true_label, kmeans.labels_)
    
    # ==== 基于 KMeans 距离的“真值”分区标签（1~10） ====
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_  # (k, 2)

    # 每个点到自己簇中心的欧氏距离
    dists = np.linalg.norm(true_data - centers[labels], axis=1)  # (N,)

    # 每个簇内部的最大距离，用来当 100%
    cluster_max = np.zeros(k, dtype=float)
    for c in range(k):
        mask = (labels == c)
        if mask.any():
            cluster_max[c] = dists[mask].max()
        else:
            cluster_max[c] = 0.0

    # 归一化到 [0,1]，再划分到 1~10 档
    norm_dist = np.zeros_like(dists)
    for i in range(len(dists)):
        m = cluster_max[labels[i]]
        norm_dist[i] = dists[i] / m if m > 1e-12 else 0.0

    # 理论上 norm_dist ∈ [0,1]，直接映射到 1~10
    true_zone = np.ceil(norm_dist * 10).astype(int)
    true_zone = np.clip(true_zone, 1, 10)

    n_node=len(true_data)
    
    n_node=k
    
    radius_model(n_node)
    
    pred_dir = os.path.join("cmps", str(n_node))
    # pick the first pred file (e.g., "0_pred.txt")
    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*_pred.txt")))
    if pred_files:
        Zone = np.loadtxt(pred_files[0])
        Zone = np.asarray(Zone).reshape(-1)  # 展平成一维

        model_radius_list.append(Zone)

        if Zone.shape[0] != true_data.shape[0]:
            print(f"[WARN] {dataset_name}: 预测值长度({Zone.shape[0]}) != 数据点数({true_data.shape[0]}), 无法计算 zone ACC")
            maximun_acc = init_acc
        else:
            # ==== 按照每个簇内部的预测值范围，分别做 10 档 ====
            pred_zone = np.zeros_like(Zone, dtype=int)

            for c in range(k):
                mask = (labels == c)
                if not mask.any():
                    continue
                vals_c = Zone[mask]
                vmin = vals_c.min()
                vmax = vals_c.max()

                if vmax - vmin < 1e-12:
                    # 这一类里预测值几乎相同，就给一个中间档位，例如 5
                    pred_zone[mask] = 5
                else:
                    norm_c = (vals_c - vmin) / (vmax - vmin)       # 映射到 [0,1]
                    deciles_c = np.floor(norm_c * 10).astype(int) + 1  # 1~10
                    deciles_c = np.clip(deciles_c, 1, 10)
                    pred_zone[mask] = deciles_c

            # ==== 与 true_zone 对比，计算按类归一化后的 zone-ACC ====
            zone_acc = (pred_zone == true_zone).mean()
            print(f"{dataset_name} per-cluster zone-ACC = {zone_acc:.4f}")

            model_acc_simple_list.append(zone_acc)
            maximun_acc = init_acc
        return init_acc, maximun_acc, dataset_name
    else:
        print(f"No pred files found in {pred_dir}")
        model_acc_simple_list.append(float('nan'))  # 占位，便于 .4f 打印
        model_k_list.append(0)
    # Concatenate all loaded data and labels
if __name__ == '__main__':
    model_acc_list = []
    init_acc_list = []
    dataset_name_list = []
    
    iris= load_iris()
    wine = load_wine()
    breast_cancer = load_breast_cancer()

    init_acc, maximun_acc, dataset_name=read_data(iris.data, iris.target, "iris")
    model_acc_list.append(maximun_acc)
    init_acc_list.append(init_acc)
    dataset_name_list.append(dataset_name)    

    init_acc, maximun_acc, dataset_name=read_data(wine.data, wine.target, "wine")
    model_acc_list.append(maximun_acc)
    init_acc_list.append(init_acc)
    dataset_name_list.append(dataset_name)    

    init_acc, maximun_acc, dataset_name=read_data(breast_cancer.data, breast_cancer.target, "breast_cancer")
    model_acc_list.append(maximun_acc)
    init_acc_list.append(init_acc)
    dataset_name_list.append(dataset_name)    

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
        print(filename)
        df = pd.read_csv(filepath, header=None)  
        arr = df.values
        # Ensure 2D array
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        X = arr[:, :-1]
        y = arr[:, -1].astype(int)
        init_acc, maximun_acc, dataset_name = read_data(X,y,filename)
        model_acc_list.append(maximun_acc)
        init_acc_list.append(init_acc)
        dataset_name_list.append(dataset_name)
        cnt+=1
    def fmt_list(lst, ndigits=4):
        return "[" + ", ".join(f"{float(x):.{ndigits}f}" for x in lst) + "]"

    #for i in range(len(model_acc_list)):
     #   print(
      #      f"Dataset: {dataset_name_list[i]}, "
       #     f"Initial ACC: {init_acc_list[i]:.4f}, "
        #    f"model acc: {model_acc_simple_list[i]:.4f}, "  # 这是‘上面的’（radii→kmedoid, 不搜索）的ACC
         #   f"coding tree ACC: {model_acc_list[i]:.4f}, "   # 这是 code_for_up 搜出来的“最大ACC”
          #  f"model k: {model_k_list[i]}"
      #  )
    # 可选的半径“汇总打印”（如果你只想要 read_data 里那两行，就把下面这段删掉）
    for i in range(len(dataset_name_list)):
        # 第1行：KMeans 的 k 个半径
        print(f"{dataset_name_list[i]} {fmt_list(origin_radius_list[i])}")
        # 第2行：模型预测 top-k 半径（如果没预测，就打印 []）
        if i < len(model_radius_list):
            print(fmt_list(model_radius_list[i]))
        else:
            print("[]")