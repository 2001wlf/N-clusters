import numpy as np
import random
import math
import time
from sklearn.neighbors import KDTree
from sklearn.neighbors import BallTree
from sklearn.datasets import make_blobs
from itertools import combinations
from functools import partial
from multiprocessing.pool import Pool
from multiprocessing import RawArray
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors
#data, _ =make_blobs(n_samples=500000,n_features=2,cluster_std=0.1)

#read data
MAX_PROCESSOR = 8
var_dict = {}

r_tot_c = 0
#print(data.shape, W.sum())


def f_swap(args):
    if(args[0]== "SWAP"):
        return SWAP_ORG(*args[1:])
    if(args[0]=="SEARCH"):
        return NEIGHBOR_SEARCH(*args[1:])
    
def SWAP_ORG(pair,init):
    id_out = pair[0]
    id_in = pair[1]
    data1 = np.frombuffer(var_dict['data']).reshape(var_dict['data_shape'])
    w = np.frombuffer(var_dict['w']).reshape(var_dict['w_shape'])
    init_new = init.copy()
    init_new[id_out] = id_in
    init_new_np = data1[init_new]
    init_new_np = np.array(init_new_np.copy())
    Tree1 = BallTree(init_new_np,leaf_size=40)
    dist1, _ = Tree1.query(data1,k=1)
    dist1 = dist1[:,0] ** 2
    cost1 = (dist1 * w).sum()
    return cost1, id_out, id_in


def fast_local_search(data, init, ybxl, r, t, k, W):
    print("-----------------------------------Start Original Local Search------------------------------------")
    "Strategy 1: Use more space comlexity and heap structure for fast local search implementation"
    "Preprocessing steps for storing the distance and assignment information"
    INF = float("inf")
    nbrs = NearestNeighbors(n_neighbors=2).fit(data[init])
    #dist = (pairwise_distances(X, centers, metric="euclidean")) ** 2
    dist, ind = nbrs.kneighbors(data)
    dist = dist ** 2
    dist_2 = dist.copy() 
    dist = dist[:, 0] * W
    dist_2[:,0] = dist_2[:, 0] * W
    dist_2[:,1] = dist_2[:, 1] * W
    init_ff = None
    # ind = np.argsort(dist, axis=1)
    affected_list = [[] for i in range(0,init.shape[0])]
    for i in range(0,data.shape[0]):
        affected_list[ind[i][0]].append(i)
    for i in range(0, len(affected_list)):
        affected_list[i] = np.array(affected_list[i], dtype=int)
    
    "Calculating the current clustering cost"
    cost_now = (dist_2[:,0]).sum()
        
    "Construct the sampling distribution"
    prob_modified = dist_2[:,0] / dist_2[:,0].sum()
    #boosting_target = math.ceil(self.n_outliers_ * (1 + self.epsilon_))
    sample_range = [i for i in range(0,data.shape[0])]
    #factor_l = self.Fast_Oversamling_Factor_Finding(prob, boosting_target, None, self.n_outliers_, self.epsilon_, self.delta_)
    #prob = prob * factor_l
    #prob_id_large = (np.argwhere(prob>1))[:,0]
    #prob[prob_id_large] = 1
    #prob_modified = prob.copy() / (prob.copy()).sum()

    
    "Start the Local Search Process"
    #print("------------------------------Start Local Search-----------------------------", cost_now)
    
    fail = 0
    cost_glob = float("inf")
    
    data2 = data ** 2
    data2_sum = np.sum(data2, axis=1)
    
    "Fast Local Search"
    for i in range(0, r):
        "preparation"
        cost_min = INF
        swap_id = None
        
        
        
        "Construct the sampling distribution"
        "Sample one data point from the modified probability"
        if(data.shape[0] < 10000 or data.shape[0]>20000):
            next_point = np.random.choice(sample_range, size=1, p=prob_modified)[0]
            # p1 = np.random.randint(low=0, high=5000, size=data.shape[0]) / 5000
            # pdiff = prob_modified - p1
            # #pdiff = np.random.binomial(1, prob_boost)
            # id_range = np.argwhere(pdiff > 0)
            # id_range = id_range[:, 0]
            # if (len(id_range) != 1):
            #     continue
            # next_point = id_range[0]
        
            
        else:
            p1 = np.random.randint(low=0, high=10000, size=data.shape[0]) / 10000
            pdiff = prob_modified - p1
            #pdiff = np.random.binomial(1, prob_boost)
            id_range = np.argwhere(pdiff > 0)
            id_range = id_range[:, 0]
            if (len(id_range) != 1):
                continue
        #print("Yes")
            next_point = id_range[0]
        centers_new = data[next_point]
        dist_tot_new = data2_sum + np.sum(centers_new ** 2, axis=0) - 2 * np.sum(data * centers_new, axis=1)
        dist_tot_new = dist_tot_new * W
        #print(dist_tot_new)
        #(pairwise_distances(X, centers_new, metric="euclidean"))[:,0] ** 2
        
        "Make the comparison between the distances of nearest and the new centers"
        dist_tot_new_modified = (dist_2.copy())[:,0]
        dist_diff = dist_tot_new_modified - dist_tot_new
        dist_large_id = np.where(dist_diff>0)
        dist_tot_new_modified[dist_large_id] = dist_tot_new[dist_large_id]
        #print(dist_tot_new_modified.sum())
        
        dist_tot_new_modified_sum = dist_tot_new_modified.sum()

        
        "Try to enumerate possible swap pairs"
        for j in range(0, k):
            "Find the points whose closest center are swapped out"
            "Now try to swap the j-th center out"
            dist_temp = dist_2.copy()
            id_affected = affected_list[j]
   
            "Compare the distances and calculate the new cost"
            dist_affected_modified = (dist_temp[id_affected])[:,1]
            pd = dist_tot_new[id_affected]
            dist_diff = dist_affected_modified - pd
            id_large = np.where(dist_diff > 0)
            dist_affected_modified[id_large] = pd[id_large]
            cost_new =  dist_tot_new_modified_sum - (dist_tot_new_modified[id_affected]).sum() + dist_affected_modified.sum()
            
            
            # if(i==100):
            #     centers_temp = centers.copy()
            #     centers_temp[j] = X[next_point]
            #     print("CheckCost", CheckCost(X, centers_temp), cost_new)
            
            
            "Judge if the swap is feasible"
            if(cost_new < cost_min):
                cost_min = cost_new
                swap_id = j
            
        "Check whether the minimum cost swap is feasible"
        if(cost_min < (1-1/(100*k))*cost_now):
            #print("Check", cost_min)
            "Perform this swap"
            init[swap_id] = next_point
            cost_now = cost_min
            nbrs = NearestNeighbors(n_neighbors=2).fit(data[init])
            "Renew the distance structures"
            #center_new = (data[next_point])
            #pd = data2_sum + np.sum(centers_new ** 2, axis=0) - 2 * np.sum(data * next_point, axis=1)
            #pd = (pairwise_distances(X,center_new, metric="euclidean"))[:,0] ** 2
            #dist[:,swap_id] = pd
            #ind = np.argsort(dist, axis=1)
            dist, ind = nbrs.kneighbors(data)
            dist = dist ** 2
            dist_2 = dist.copy() 
            dist = dist[:, 0] * W
            dist_2[:,0] = dist_2[:, 0] * W
            dist_2[:,1] = dist_2[:, 1] * W
            #print("Check", cost_now, dist.sum())
            affected_list = [[] for j1 in range(0, init.shape[0])]
            for j1 in range(0,data.shape[0]):
                affected_list[ind[j1][0]].append(j1)
            for j1 in range(0, len(affected_list)):
                affected_list[j1] = np.array(affected_list[j1], dtype=int)            
            
            prob_modified = dist_2[:,0].copy() / (dist_2[:,0].copy()).sum()
            #factor_l = self.Fast_Oversamling_Factor_Finding(prob, boosting_target, None, self.n_outliers_, self.epsilon_, self.delta_)
            #prob = prob * factor_l
            #prob_id_large = (np.argwhere(prob>1))[:,0]
            #prob[prob_id_large] = 1
            #prob_modified = prob.copy() / (prob.copy()).sum()
            #print("Round",i,"Has A Swap", cost_now)
            #print("CheckCost", CheckCost(X, centers))
        else:
            continue
    
    #print("Enter the Lloyd")
    nbrs = NearestNeighbors(n_neighbors=1).fit(data)
    # TreeL = KDTree(init_L.copy(), leaf_size=40)
    # _, indL = TreeL.query(data.copy(), k=1)
    indL = ind[:,0]
    #print(indL)
    while(1):
        #Create new centers
        init_L_new =  data[init.copy()]
        for i1 in range(0,k):
            id_i = np.argwhere(indL==i1)
            id_i = id_i[:,0]
            W_repeat = np.tile(W[id_i],(data.shape[1],1))
            W_repeat = W_repeat.transpose()
            #print(id_i)
            #print(np.sum(data[id_i] * W_repeat,axis=0))
            init_L_new[i1] = np.sum(data[id_i] * W_repeat,axis=0) / W[id_i].sum()
            #print("Check",init)
        #Find the nearest data points to approximate new centers
        #print("CHECK",init_L_new)
        #TreeL1 = KDTree(data.copy(),leaf_size=40)
        #_, indL1 = TreeL1.query(init_L_new, k=1)
        _, indL1 = nbrs.kneighbors(init_L_new)
        #print("CHECK",init_L_new.copy())
        indL1 = indL1[:,0]
        #print("Check For New Centers", indL1)
        init_temp = data[indL1]
        #Recalculate the cost
        #TreeL = KDTree(init_temp.copy(), leaf_size=40)
        #distL, indL = TreeL.query(data.copy(), k=1)
        nbrs1 = NearestNeighbors(n_neighbors=1).fit(init_temp)
        distL, indL = nbrs1.kneighbors(data)
        indL = indL[:,0]
        distL = distL[:,0] ** 2
        distL = distL * W
        if(distL.sum()<cost_now):
            cost_now = distL.sum()
            init = indL1.copy()
            #init_L = init_temp.copy()
            init_ff = init_temp.copy()
            #print("Lloyd Has A Swap", cost_now)
        else:
            #print("No Better",distL.sum())
            break
    
    

    if(cost_now<cost_glob):
        #print("CHECK",cost_now,cost_glob)
        cost_glob = cost_now
    else:
        fail += 1

    
    
    
    if(fail>=0):
        return cost_glob, init_ff
        print("--------------------Final Clustering Cost---------------------", cost_glob)

    
            
            
            
            
def generate_candidate_set(center_list):
    list_f = []
    for i in range(1,len(center_list)+1):
        for j in combinations(center_list,i):
            list_f.append(list(j))
    return list_f


def init_worker(data, data_shape,w, w_shape):
    var_dict['data'] = data
    var_dict['data_shape'] = data_shape
    var_dict['w'] = w
    var_dict['w_shape'] = w_shape
    

def swap_t(args):
    if(args[0] == "LOCAL_SEARCH"):
        return SWAP_SEARCH1(*args[1:])
    if(args[0] == "LOCAL_SEARCH1"):
        return SWAP_SEARCH2(*args[1:])
    if(args[0] == "NEIGHBOR_SEARCH"):
        return NEIGHBOR_SEARCH(*args[1:])
def SWAP_SEARCH1(groups, L, init):
    Min = 100000000000000000000000000000000005
    id_out = []
    id_in = []
    data1 = np.frombuffer(var_dict['data']).reshape(var_dict['data_shape'])
    j = L[0]
    nextpoint = L[1]
    group = [groups[i][0] for i in range(0,len(groups))]
    W1 = [groups[i][1] for i in range(0,len(groups))]
    init_new = init.copy()
    for i in range(0,len(j)):
        init_new[j[i]] = nextpoint[i]
    init_new_np = data1[init_new]
    Tree1 = BallTree(init_new_np, leaf_size=40)
    for s in range(0,len(group)):
        data_s = group[s]
        W_s = W1[s]
        dist1, _ = Tree1.query(data_s,k=1)
        dist1 = dist1[:,0] ** 2
        dist1 = dist1 * W_s
        cost1 = dist1.sum()
        if(cost1<Min):
            Min = cost1
            id_out = []
            id_out.append(j)
            id_in = []
            id_in.append(nextpoint)
    return Min, id_out, id_in

def SWAP_SEARCH2(groups,sample_size,L,init):
    Min = 100000000000000000000000000000000005
    id_out = []
    id_in = []
    data1 = np.frombuffer(var_dict['data']).reshape(var_dict['data_shape'])
    w = np.frombuffer(var_dict['w']).reshape(var_dict['w_shape'])
    nextpoint = L[0]
    lg = len(nextpoint)
    if(lg==1):
        for q in range(0,k):
            init_new = init.copy()
            init_new[q] = nextpoint[0]
            init_new_np = data1[init_new]
            Tree1 = BallTree(init_new_np, leaf_size=40)
            for s in range(0,groups):
               id_sample = random.sample(range(0,data1.shape[0]), sample_size)
               id_sample = np.array(id_sample,dtype=int)
               data_s = data1[id_sample]
               W_s = w.copy()[id_sample]
               dist1, _ = Tree1.query(data_s,k=1)
               dist1 = dist1[:,0] ** 2
               dist1 = dist1 * W_s
               cost1 = dist1.sum()
               if(cost1<Min):
                   id_out = []
                   id_in = []
                   id_out.append(q)
                   id_in.append(nextpoint[0])
                   Min = cost1
    elif(lg==2):
        for q in range(0,k):
            for q1 in range(q+1,k):
                init_new = init.copy()
                init_new[q] = nextpoint[0]
                init_new[q1] = nextpoint[1]
                init_new_np = data1[init_new]
                Tree1 = BallTree(init_new_np, leaf_size=40)
                for s in range(0,groups):
                   id_sample = random.sample(range(0,data1.shape[0]), sample_size)
                   id_sample = np.array(id_sample,dtype=int)
                   data_s = data1[id_sample]
                   W_s = w.copy()[id_sample]
                   dist1, _ = Tree1.query(data_s,k=1)
                   dist1 = dist1[:,0] ** 2
                   dist1 = dist1 * W_s
                   cost1 = dist1.sum()
                   if(cost1<Min):
                       id_out = []
                       id_in = []
                       id_out.append(q)
                       id_out.append(q1)
                       id_in.append(nextpoint[0])
                       id_in.append(nextpoint[1])
                       Min = cost1
    else:
        for q in range(0,k):
            for q1 in range(q+1,k):
                for q2 in range(q1+1):
                    init_new = init.copy()
                    init_new[q] = nextpoint[0]
                    init_new[q1] = nextpoint[1]
                    init_new[q2] = nextpoint[2]
                    init_new_np = data1[init_new]
                    Tree1 = BallTree(init_new_np, leaf_size=40)
                for s in range(0,groups):
                   id_sample = random.sample(range(0,data1.shape[0]), sample_size)
                   id_sample = np.array(id_sample,dtype=int)
                   data_s = data1[id_sample]
                   W_s = w.copy()[id_sample]
                   dist1, _ = Tree1.query(data_s,k=1)
                   dist1 = dist1[:,0] ** 2
                   dist1 = dist1 * W_s
                   cost1 = dist1.sum()
                   if(cost1<Min):
                       id_out = []
                       id_in = []
                       id_out.append(q)
                       id_out.append(q1)
                       id_out.append(q2)
                       id_in.append(nextpoint[0])
                       id_in.append(nextpoint[1])
                       id_in.append(nextpoint[2])
                       Min = cost1
    return Min, id_out, id_in

def NEIGHBOR_SEARCH(L, init):
    i = L[0]
    j = L[1]
    data1 = np.frombuffer(var_dict['data']).reshape(var_dict['data_shape'])
    w = np.frombuffer(var_dict['w']).reshape(var_dict['w_shape'])
    init_f_temp = init.copy()
    init_f_temp[i] = j
    init_new_np1 = data1[init_f_temp]

    Treeq = BallTree(init_new_np1,leaf_size=40)
    dist, _ = Treeq.query(data1,k=1)
    dist = dist[:,0] ** 2
    cost_f = (dist * w).sum()
    return cost_f, i, j, dist

        
def kmeans_plus(data,k, W):
    Min_f = float("inf")
    init_f = None
    for i1 in range(0, 10):
        #print(i1)
        init = []
        prob = 0
        id_range = [i for i in range(0,data.shape[0])]
        greedy_l = math.ceil(3*math.log2(k))
        for i in range(0,k):
            if(i==0):
                random_id = random.sample(range(0, data.shape[0]), greedy_l)
                Min = float("inf")
                next_id = None
                for j in range(0, len(random_id)):
                    cost_j = 0
                    for j1 in range(0, len(data)):
                        cost_j += (((data[j1] - data[random_id[j]]) ** 2) * W[j1]).sum()
                    if(cost_j < Min):
                        Min = cost_j
                        next_id = j
                init.append(j)
            else:
                Min = float("inf")
                next_id = None
                for j in range(0, greedy_l):
                    nextpoint = np.random.choice(id_range,p=prob,size=1,replace=False)[0]
                    data_query = (data[nextpoint]).reshape(1, -1)
                    dist_j = pairwise_distances(data, data_query) ** 2
                    dist_j = np.min(dist_j, axis=1) * W
                    dist_diff = dist_j - dist
                    id_small = np.where(dist_diff<0)
                    cost_j = (dist_diff[id_small]).sum()
                    if(cost_j < Min):
                        Min = cost_j
                        next_id = nextpoint
                init.append(nextpoint)
            
            init_id = np.array(init.copy(),dtype=int)
            init_np = data[init_id]
            Tree = BallTree(init_np.copy(), leaf_size=40)
            dist, _ = Tree.query(data.copy(),k=1)
            dist = (dist[:,0] ** 2) * W
            prob = dist.copy() / (dist.copy()).sum()
        if(dist.sum() < Min_f):
            #print("Yes")
            Min_f = dist.sum()
            init_f = init.copy()
    #print(init_f)
    return init_f
def Lloyd(data, k, W):
    km = KMeans(n_clusters=k,init="k-means++",n_init=10,max_iter=300)
    km.fit(data)
    centers = km.cluster_centers_
    Tree = BallTree(data,leaf_size=40)
    _, ind = Tree.query(centers,k=1)
    ind = ind [:,0]
    center_new = data[ind]
    Tree1 = BallTree(center_new,leaf_size=40)
    dist, _ = Tree1.query(data,k=1)
    dist = dist[:,0] ** 2
    print("------------------------Lloyd--------------------------", (dist * W).sum())
    return (dist * W).sum()



def CheckCost(X, centers):
    pd = pairwise_distances(X, centers)
    pd = pd ** 2
    pd = np.min(pd, axis=1)
    # TreeL1 = BallTree(centers, leaf_size=40)
    # #print(len(centers))
    # distL, _ = TreeL1.query(X, k=1)
    # distL = distL[:,0] ** 2
    # cost_now = distL.sum()
    #print(cost_now)
    return pd.sum()

def Projection1(X, centers):
    TreeL1 = BallTree(X, leaf_size=40)
    _, indL1 = TreeL1.query(centers, k=1)
    indL1 = indL1[:, 0]
    init_temp = X[indL1]
    return indL1


def LSpp_cluster(data, k):
    data = data.copy()
    W = np.ones(data.shape[0])

    # 1) 用 KMeans++ 做初始化，再投影到最近数据点
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=2).fit(data)
    centers0 = km.cluster_centers_
    init_id = Projection1(data, centers0)

    # 2) 调 LS++ 主算法
    cost, centers_final = fast_local_search(
        data.copy(),
        init_id.copy(),
        1/100, 400, 2, k,
        W.copy()
    )

    # 3) 最近邻分配标签
    nbrs = NearestNeighbors(n_neighbors=1).fit(centers_final)
    _, labels = nbrs.kneighbors(data)
    labels = labels[:, 0]

    return centers_final, labels, cost