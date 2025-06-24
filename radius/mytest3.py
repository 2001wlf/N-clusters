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
from code_util import code_for_up,variable_eps_dbscan
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
    parser.add_argument('--eval_file_path', default='C:/Users/10998/Desktop/N-clusters/radius/mydata2/val', help='')
    parser.add_argument('--model_path', type=str, default='C:/Users/10998/Desktop/N-clusters/radius/saved/radius_centralpoint/3.pt', help='')
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
def read_data(true_data,true_label,dataset_name):
    """
    Load dataset from the local mydata2/train folder under the script's directory.
    Assumes each .txt file is whitespace-delimited, with the last column as the label.
    """
    #wine "
    print(f"dataset: {dataset_name}")
    true_data = true_data[:,:2]
    scaler = MinMaxScaler(feature_range=(0, 1))
    true_data = scaler.fit_transform(true_data)
    generate_datasets(true_data, true_label)
    #true_label = wine.target


    k=len(np.unique(true_label))
    #k=10
    kmeans=KMeans(n_clusters=k, random_state=0).fit(true_data)
    raidusQuery(true_data,kmeans.labels_)
    # Compute and print clustering accuracy
    init_acc = compute_accuracy(true_label, kmeans.labels_)
    print(f"Clustering accuracy: {init_acc:.4f}")

    n_node=len(true_data)
    radius_model(n_node)
    
    # --- compute pairwise Euclidean distance matrix for iris_data ---
    dist_matrix = cdist(true_data, true_data, metric='euclidean')
    #for i in range(15):
     #   for j in range(15):
      #      print(f"Distance between point {i} and point {j}: {dist_matrix[i, j]:.4f}")
    #print("Distance matrix shape:", dist_matrix.shape)
    # --- perform coverage-based sampling on predicted radii ---
    pred_dir = os.path.join("cmps", str(n_node))
    # pick the first pred file (e.g., "0_pred.txt")
    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*_pred.txt")))
    if pred_files:
        radii = np.loadtxt(pred_files[0])
        # ensure radii is a column vector
        radii=radii/5
       #maximun_acc=0
        #maximun_acc_k=0
        #maximun_acc_i=0;
        #for i in range(1,6):
            #dbsacn_label= variable_eps_dbscan(true_data, radii, 5)
            #dbsacn_k=len(np.unique(dbsacn_label))
            #print(f"DBSCAN clustering accuracy: {compute_accuracy(true_label, dbsacn_label):.4f}, clusters: {dbsacn_k}")
            #dbsacn_acc= compute_accuracy(true_label, dbsacn_label)
            #if dbsacn_acc>maximun_acc:
             #   maximun_acc=dbsacn_acc
              #  maximun_acc_k=dbsacn_k
               # maximun_acc_i=i
        #print(f"Dataset: {dataset_name}, Initial ACC: {init_acc:.4f}, Model ACC: {maximun_acc:.4f},Model K: {maximun_acc_k}, Model I: {maximun_acc_i}")
        #return init_acc, maximun_acc, dataset_name
        
        maximun_acc,maximun_acc_index = code_for_up(true_data, true_label, k, radii)
        print("\nInit_acc",init_acc,"\t最大ACC:", maximun_acc, "\t方法:", maximun_acc_index)     
        return init_acc, maximun_acc, dataset_name
        
        #assignments, centers, noise = coverage_sampling(radii, true_data, k)
        #print("Sampled centers (indices):", centers)
        #for cid in range(len(centers)):
        #    size = np.sum(assignments == cid)
        #    print(f"Cluster {cid}: size {size}")
        #print(f"Noise points count: {len(noise)}")
        # Compute and print accuracy excluding noise
        #acc_cov = compute_accuracy_filtered(true_label, assignments, noise_label=-1)
        #print(f"Coverage sampling accuracy (excluding noise): {acc_cov:.4f}")


    else:
        print(f"No pred files found in {pred_dir}")
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
    for i in range(len(model_acc_list)):
        print(f"Dataset: {dataset_name_list[i]}, Initial ACC: {init_acc_list[i]:.4f}, Model ACC: {model_acc_list[i]:.4f}")