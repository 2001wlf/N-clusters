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

def Cpoints_model(my_node):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--eval_interval', type=int, default=1, help='')
    parser.add_argument('--eval_batch_size', type=int, default=1, help='')
    parser.add_argument('--eval_file_path', default='C:/Users/10998/Desktop/N-clusters/CPoint/mydata2/val', help='')
    parser.add_argument('--model_path', type=str, default='C:/Users/10998/Desktop/N-clusters/CPoint/saved/radius_centralpoint_gm/15.pt', help='')
    args = parser.parse_args()
    edge_cw = None
    n_edges = 20
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
def get_marked_indices(cpoints, threshold=None, ratio=None):
    """
    同时支持：
    - threshold:超过阈值的点
    - ratio:前百分之 ratio 的点
    注意看大于小于
    任意一个不为 None 即可触发
    """
    if threshold is not None:
        return np.where(cpoints > threshold)[0] #最大
        #return np.where(cpoints < threshold)[0] #最小
    if ratio is not None:
        num = max(1, int(len(cpoints) * ratio))
        return np.argsort(cpoints)[-num:] #最大
        #return np.argsort(cpoints)[:num] #最小

    return np.array([])  # 默认返回空
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
    k=k;
    kmeans = KMeans(init='k-means++',n_init='auto',n_clusters=k, random_state=0).fit(true_data)

    # 保存用于后续评估（此处不再打印ACC，避免打断你的两行输出）
    init_acc = compute_accuracy(true_label, kmeans.labels_)

    n_node=len(true_data)
    Cpoints_model(n_node)
    
    # --- compute pairwise Euclidean distance matrix for iris_data ---
    #dist_matrix = cdist(true_data, true_data, metric='euclidean')
    #for i in range(15):
     #   for j in range(15):
      #      print(f"Distance between point {i} and point {j}: {dist_matrix[i, j]:.4f}")
    #print("Distance matrix shape:", dist_matrix.shape)
    # --- perform coverage-based sampling on predicted radii ---
    pred_dir = os.path.join("cmps", str(n_node))
    # pick the first pred file (e.g., "0_pred.txt")
    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*_pred.txt")))
    if pred_files:
        cpoint = np.loadtxt(pred_files[0])
        
        topk_indices = np.argsort(cpoint)[-k:]  # 从小到大排，取最后k个索引
        topk_values = cpoint[topk_indices]
        
        topk_values_tosee = np.sort(cpoint)[-k:][::-1]  # 降序
        print(topk_values_tosee.tolist())
        model_radius_list.append(topk_values_tosee.tolist())
        
        #print(f"model topk_radii: {topk_values}")
        #print(topk_values.tolist())
        
        # ensure radii is a column vector
        #centers_idx, ref_c, hit_ratio, assign_acc = eval_topk_vs_kmeans(true_data, cpoint, k=k, match_eps=0.05)
        #print("Top-k 命中参考中心比例:", hit_ratio)
        #print("用 Top-k 做最近中心分配 vs KMeans 标签的一致性 ACC:", assign_acc)
        # initialize centers from radii and assign by nearest medoid
        #pred_label, centers = kmedoid_from_radii(radii, true_data, k,max_iter=0)
        
        # 1) 先把 top-k 的索引按 Cpoint 值降序稳定一下（可选）
        order = np.argsort(cpoint[topk_indices])[::-1]
        centers_idx = topk_indices[order]

        # 2) 取这 k 个点作为“固定中心”
        centers = true_data[centers_idx]           # (k, d)

        # 3) 只做一次“最近中心分配”，不更新中心（无迭代）
        D = cdist(true_data, centers)               # (N, k)
        pred_label = np.argmin(D, axis=1)           # (N,)
        
        #根据预测的centers画出数据集并且标注中心点的位置
        #marked = get_marked_indices(cpoint, threshold=0.1)               # 方法 1
        marked = get_marked_indices(cpoint, ratio=0.10)               # 方法 2
        draw_picture(marked,true_data,true_label,dataset_name)
        
        model_acc_simple = compute_accuracy(true_label, pred_label)
        model_k_val = int(len(np.unique(pred_label)))
        model_acc_simple_list.append(model_acc_simple)
        model_k_list.append(model_k_val)
        print(f"\nInitial centers from radii: {centers},\tInit_acc: {init_acc:.4f},\t Model acc: {compute_accuracy(true_label, pred_label):.4f},\t Model k: {len(np.unique(pred_label))}")
        
        maximun_acc,maximun_acc_index = code_for_up(true_data, true_label, k, cpoint)
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

    for i in range(len(model_acc_list)):
        print(
            f"Dataset: {dataset_name_list[i]}, "
            f"Initial ACC: {init_acc_list[i]:.4f}, "
            f"model acc: {model_acc_simple_list[i]:.4f}, "  # 这是‘上面的’（radii→kmedoid, 不搜索）的ACC
            f"coding tree ACC: {model_acc_list[i]:.4f}, "   # 这是 code_for_up 搜出来的“最大ACC”
            f"model k: {model_k_list[i]}"
        )
    # 可选的半径“汇总打印”（如果你只想要 read_data 里那两行，就把下面这段删掉）
    for i in range(len(dataset_name_list)):
        # 第1行：KMeans 的 k 个半径
        print(f"{dataset_name_list[i]} {fmt_list(origin_radius_list[i])}")
        # 第2行：模型预测 top-k 半径（如果没预测，就打印 []）
        if i < len(model_radius_list):
            print(fmt_list(model_radius_list[i]))
        else:
            print("[]")