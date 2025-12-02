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
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from sklearn.cluster import KMeans
#data, _ =make_blobs(n_samples=500000,n_features=2,cluster_std=0.1)

#read data
MAX_PROCESSOR = 16
var_dict = {}

r_tot_c = 0
#print(data.shape, W.sum())


def Projection1(X, centers):
    TreeL1 = BallTree(X, leaf_size=40)
    _, indL1 = TreeL1.query(centers, k=1)
    indL1 = indL1[:, 0]
    init_temp = X[indL1]
    return indL1

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

        
def kmeans_plus(data,k):
    init = []
    prob = 0
    id_range = [i for i in range(0,data.shape[0])]
    for i in range(0,k):
        if(i==0):
            rid = random.sample(range(0,data.shape[0]),1)[0]
            init.append(rid)
        else:
            nextpoint = np.random.choice(id_range,p=prob,size=1,replace=False)[0]
            init.append(nextpoint)
        
        init_id = np.array(init.copy(),dtype=int)
        init_np = data[init_id]
        Tree = BallTree(init_np.copy(), leaf_size=40)
        dist, _ = Tree.query(data.copy(),k=1)
        dist = dist[:,0] ** 2
        prob = dist.copy() / (dist.copy()).sum()
    return init

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

def fast_local_search2(data, init, k, ybxl, r, t, c, alpha, epsilon, W, single, multi, triple):
    # print("-----------------------------------Start Fast Local Search------------------------------------")
    data_shape = (data.shape[0], data.shape[1])
    data_g = RawArray('d', data_shape[0] * data_shape[1])
    data_np = np.frombuffer(data_g).reshape(data_shape)
    w_shape = (W.shape[0])
    w_g = RawArray('d', w_shape)
    w_np = np.frombuffer(w_g).reshape(w_shape)
    np.copyto(data_np, data)
    np.copyto(w_np, W)
    inf = 1000000000000000000000000000000000000000005
    groups = 5
    sample_size = math.ceil(3 * k * math.log2(k / epsilon) / epsilon)
    sample_size = min(data.shape[0], sample_size)
    sample_size = min(math.floor(0.4*data.shape[0]), sample_size)
    cost_glob = inf
    id_range = [i for i in range(0, data.shape[0])]
    fail = 0
    swap_count = np.zeros(data.shape[0])
    # Parallelization
    pool1 = Pool(processes=MAX_PROCESSOR, initializer=init_worker, initargs=(data_g, data_shape, w_g, w_shape))
    count_tot = 0
    init_ff = -1
    while (1):
        init_np = data.copy()[init]
        Tree = BallTree(init_np.copy(), leaf_size=40)
        dist, ind = Tree.query(data.copy(), k=t)
        ind = np.array(ind, dtype=int)
        dist = dist[:, 0] ** 2
        dist = dist * W
        prob = dist.copy() / dist.copy().sum()
        id_range_f = [i for i in range(0, data.shape[0])]
        cost_now = dist.sum()

        # groups = math.ceil(c*alpha/(alpha-1))

        # groups = 1
        # sample_size = data.shape[0]
        # print("Check Size", "Sample Size", sample_size, "Group Size", groups)
        tswap = time.time()
        for i in range(0, r):
            tswap1 = time.time()
            # print("Count",i)
            # print(init)
            # nextpoint = np.random.choice(id_range,1,replace=False,p=prob)[0]
            prob_boost = prob * 1.5
            p1 = np.random.rand(data.shape[0])
            pdiff = prob_boost - p1
            id_range = np.argwhere(pdiff >= 0)
            id_range = id_range[:, 0]
            if (len(id_range) < 1 or len(id_range) > 2):
                continue
            next_points = generate_candidate_set(list(id_range))
            # print("Check Swap Size", len(next_points))
            init_f_np = 0
            init_f = 0
            id_out = []
            id_in = []
            Min = inf
            sample_groups = []
            for s in range(0, groups):
                id_sample = random.sample(range(0, data.shape[0]), sample_size)
                id_sample = np.array(id_sample, dtype=int)
                data_s = (data.copy())[id_sample]
                w_s = (W.copy())[id_sample]
                sample_groups.append([data_s, w_s])

            # Plan A
            pair = []
            tpair = time.time()
            for f in range(0, len(next_points)):
                if (len(next_points[f]) == 1):
                    pair = pair + [[[single[n1]], next_points[f]] for n1 in range(0, len(single))]
                elif (len(next_points[f]) == 2):
                    pair = pair + [[multi[n1], next_points[f]] for n1 in range(0, len(multi))]
                else:
                    pair = pair + [[triple[n1], next_points[f]] for n1 in range(0, len(triple))]
            tpair1 = time.time()
            # print("Pair Construction Time", tpair1 - tpair)
            r_tot = list(pool1.imap(swap_t, [("LOCAL_SEARCH", sample_groups, pair[b], init.copy()) for b in
                                             range(0, len(pair))]))

            # Plan B
            # pair = [[next_points[b]] for b in range(0, len(next_points))]
            # r_tot = list(pool1.imap(swap_t,[("LOCAL_SEARCH1", groups, sample_size, pair[b], init.copy()) for b in range(0,len(pair))]))

            # r_tot = []
            # for qq in range(0,len(pair)):
            #     r_tot.append(list(SWAP_SEARCH1(data.copy(),sample_groups,pair[qq],init.copy())))

            for f in range(0, len(r_tot)):
                if (r_tot[f][0] < Min):
                    Min = r_tot[f][0]
                    id_out = r_tot[f][1]
                    id_in = r_tot[f][2]
            init_f = init.copy()
            for n1 in range(0, len(id_in)):
                init_f[id_out[n1]] = id_in[n1]
            init_f_np = (data.copy())[init_f]

            # print("One Round Swap Length", len(next_points),"Takes Time", tswap2 - tswap1)
            Tree2 = BallTree(init_f_np, leaf_size=50)
            dist1, _ = Tree2.query(data.copy(), k=1)
            dist1 = dist1[:, 0] ** 2
            dist1 = dist1 * W
            cost_next = dist1.sum()
            if (cost_next < cost_now * (1 - 1 / (100 * k))):
                for m in range(0, len(id_out)):
                    swap_count[id_out[m]] -= 1
                cost_now = cost_next
                prob = dist1.copy() / dist1.copy().sum()
                init = init_f.copy()
                #print("Round", i, "Has A Swap", cost_now)

        tswap1 = time.time()
        # print("Swap Takes Time", tswap1 - tswap)
        tlloyd = time.time()
        ct = 0
        # Lloyd Type Search
        # print("Enter the Lloyd")
        init_L = data[init]
        TreeL = BallTree(init_L.copy(), leaf_size=40)
        _, indL = TreeL.query(data.copy(), k=1)
        indL = indL[:, 0]
        while (1):
            # Create new centers
            init_L_new = init_L.copy()
            for i1 in range(0, k):
                id_i = np.argwhere(indL == i1)
                id_i = id_i[:, 0]
                W_repeat = np.tile(W[id_i], (data.shape[1], 1))
                W_repeat = W_repeat.transpose()
                init_L_new[i1] = np.sum(data[id_i] * W_repeat, axis=0) / W[id_i].sum()
            # Find the nearest data points to approximate new centers
            # print(init_L_new)
            TreeL1 = BallTree(data.copy(), leaf_size=40)
            _, indL1 = TreeL1.query(init_L_new, k=1)
            indL1 = indL1[:, 0]
            # print("Check For New Centers", indL1)
            init_temp = data[indL1]
            # Recalculate the cost
            TreeL = BallTree(init_temp.copy(), leaf_size=40)
            distL, indL = TreeL.query(data.copy(), k=1)
            indL = indL[:, 0]
            distL = distL[:, 0] ** 2
            distL = distL * W
            if (distL.sum() < cost_now):
                cost_now = distL.sum()
                init = indL1.copy()
                init_L = init_temp.copy()
                # print("Lloyd Has A Swap", cost_now)
            else:
                # print("No Better", distL.sum())
                break

        tlloyd1 = time.time()
        # print("Lloyd Takes Time", tlloyd1 - tlloyd)

        tneighbor = time.time()
        # print("Enter the Nearest Neighbot Search")
        # print("Check For Init", init)
        init_new_np = data[init]
        k_list = [10,30,50]
        k_id = 0
        dist_f = 0

        # TreeS = BallTree(init_new_np, leaf_size=40)
        # distS, indS = TreeS.query(data.copy(), k=k)
        # distS = distS ** 2

        while (1):
            if (k_id >= len(k_list)):
                break
            Tree2 = BallTree(data.copy(), leaf_size=40)
            _, ind2 = Tree2.query(init_new_np, k=k_list[k_id])
            # print(ind2)
            Min = 10000000000000000000000000000000005
            init_f = 0
            init_f_id = 0
            id_out = -1
            id_in = -1
            pair = [[i2, ind2[i2][j], ] for i2 in range(0, k) for j in range(0, ind2.shape[1])]
            r_tot = list(pool1.imap(swap_t, [("NEIGHBOR_SEARCH", pair[b], init.copy()) for b in range(0, len(pair))]))
            for i2 in range(0, len(r_tot)):
                if (r_tot[i2][0] < Min):
                    id_in = r_tot[i2][2]
                    id_out = r_tot[i2][1]
                    Min = r_tot[i2][0]
                    dist_f = r_tot[i2][3]
            init_f_id = init.copy()
            init_f_id[id_out] = id_in
            init_f = data[init_f_id]

            if (Min < cost_now):
                swap_count[id_out] -= 1
                cost_now = Min
                init_new_np = init_f.copy()
                init = init_f_id.copy()
                #print("Round", ct, "Has A Swap", cost_now, "Neighbor", k_list[k_id])
                ct += 1
            else:
                k_id += 1
        # print("Neighbor Search Takes Time", tneighbor1 - tneighbor)
        swap_count[init] += 1

        # cs = ['red','orange','yellow','green','cyan','blue','purple','pink','magenta','brown']
        Treep = BallTree(init_f, leaf_size=40)
        _, indp = Treep.query(data.copy(), k=1)
        indp = indp[:, 0]
        # fig = plt.figure()
        # ax = Axes3D(fig)
        # ax.set_xlim(-0.2,0.6)
        # ax.set_ylim(-0.4,0.4)
        # ax.set_zlim(0,0.8)
        # for i3 in range(0,k):
        #     indi3 = np.argwhere(indp==i3)
        #     indi3 = indi3[:,0]
        #     ax.scatter((data[indi3])[:,0],(data[indi3])[:,1],(data[indi3])[:,2], c=cs[i3])
        # plt.show()

        # print("Centers",init_f)

        # mutation

        if (cost_now < cost_glob):
            init_ff = init.copy()
            for i3 in range(0, len(init)):
                swap_count[init[i3]] += 1
            cost_glob = cost_now
        else:
            fail += 1

        score_id = np.argsort(-swap_count)[0:k]
        score_id_set = set(score_id)
        org_set = set(init.copy())
        score_id_set = score_id_set.difference(org_set)
        Treeh = BallTree(init_new_np.copy(), leaf_size=40)
        _, indh = Treeh.query(data, k=1)
        indh = indh[:, 0]
        std_max = -1
        std_min = inf
        ind_std = -1
        id_min = - 1
        id_max = -1
        for i3 in range(0, k):
            id3 = np.argwhere(indh == i3)
            id3 = id3[:, 0]
            data_m = data[id3]
            std = np.var(data_m, axis=0)
            std_t = std.sum()
            # print("STD",std_t)
            if (std_t > std_max):
                std_max = std_t
                ind_std = id3.copy()
                id_max = i3
            if (std_t < std_min):
                std_min = std_t
                id_min = i3

        # fig1 = plt.figure()
        # ax1 = Axes3D(fig1)
        # ax1.set_xlim(-0.2,0.6)
        # ax1.set_ylim(-0.4,0.4)
        # ax1.set_zlim(0,0.8)
        # ax1.scatter((data[ind_std])[:,0],(data[ind_std])[:,1],(data[ind_std])[:,2], c=cs[0])
        # plt.show()

        if (len(score_id_set) < 5 or count_tot <= 2):
            # print("Random Mutation")

            # Plan C

            init_ex = init.copy()
            init_ex = list(init_ex)
            t_s1 = math.ceil(k * math.log2(k / 0.5) / epsilon)
            t_s1 = min(math.floor(2 * k), t_s1)
            rd_s1 = random.sample(range(0, data.shape[0]), t_s1)
            # init_ex = init_ex + list(rd_s1)
            init_ex = list(rd_s1)
            l_up = len(init_ex + list(init.copy()))
            init_ex = np.array(init_ex, dtype=int)
            init_npe = data[init_ex]

            Tree_e = BallTree(init_npe.copy(), leaf_size=40)
            dist_e, ind_e = Tree_e.query(data.copy(), k=1)
            dist_e = dist_e[:, 0] ** 2
            dist_e = dist_e * W
            ind_e = ind_e[:, 0]
            cost_ex = (dist_e * W).sum()

            s_temp = set(list(init.copy()) + list(init_ex.copy()))
            s_temp_list = np.array(list(s_temp), dtype=int)

            score_f = np.zeros(l_up)
            # dist_ex = dist_f.copy() * W
            # prob_ex = dist_ex.copy() / (dist_ex.copy()).sum()
            # #print("CHECK",id_range.shape,prob_ex.shape)
            # for i in range(0,k):
            #     nextpoint = np.random.choice(id_range_f,p=prob_ex,size=1,replace=False)[0]
            #     init_ex.append(nextpoint)

            #     init_ide = np.array(init_ex.copy(),dtype=int)
            #     init_npe = data[init_ide]
            #     Tree_e = BallTree(init_npe.copy(), leaf_size=40)
            #     dist_e, ind_e = Tree_e.query(data.copy(),k=1)
            #     dist_e = dist_e[:,0] ** 2
            #     dist_e = dist_e * W
            #     ind_e = ind_e[:,0]
            #     prob_ex = dist.copy() / (dist.copy()).sum()
            # cost_ex = dist_e.sum()

            # print("CHECK for list",init,init_ex)
            # Calculate the weights for mutation

            # print("CHECK", cost_ex)
            for i5 in range(0, len(init_ex)):
                if (i5 < k):
                    id5 = (np.argwhere(indp == i5))[:, 0]
                    id6 = (np.argwhere(ind_e == i5))[:, 0]
                    idi5 = (np.argwhere(s_temp_list == init[i5]))[:, 0]
                    cost5 = (dist_f[id5] * W[id5]).sum()
                    cost6 = (dist_e[id6] * W[id6]).sum()
                    idi6 = (np.argwhere(s_temp_list == init_ex[i5]))[:, 0]
                    score_f[idi5] += 1 * cost5 / cost_now
                    score_f[idi6] += 3 * cost6 / cost_ex
                else:
                    id6 = (np.argwhere(ind_e == i5))[:, 0]
                    cost6 = (dist_e[id6] * W[id6]).sum()
                    idi6 = (np.argwhere(s_temp_list == init_ex[i5]))[:, 0]
                    score_f[idi6] += 3 * cost6 / cost_ex
            scores3 = score_f.copy()
            # normalization
            id_score = np.argsort(-scores3)
            init_ex = np.array(init_ex, dtype=int)
            # print("OLD INIT",scores1)
            # print("CHECK", s_temp_list)
            init = (s_temp_list.copy())[id_score[0:k]]

        else:
            # print("Score Mutation")
            swap_id = (list(score_id_set))[0]
            init[id_min] = swap_id
        count_tot += 1
        if (fail >= 5):
            # print("CHECK", init_ff)
            init_ff = np.array(init_ff, dtype=int)
            #print("---------------------Final Clustering Cost---------------------", cost_glob)
            return cost_glob,init_ff
            break

def MLSP_cluster(data, k):
    data, W = np.unique(data, axis=0, return_counts=True)

    single = [i for i in range(k)]
    multi  = [[i, j] for i in range(k) for j in range(i+1, k)]
    triple = [[i, j, j1] for i in range(k) for j in range(i+1, k) for j1 in range(j+1, k)]

    init_id = np.array(kmeans_plus(data, k), dtype=int)

    cost, centers_idx = fast_local_search2(
        data, init_id, k,
        1/100, 400, 2, 10, 2, 0.5,
        W, single, multi, triple
    )

    centers = data[centers_idx]
    Tree_final = BallTree(centers, leaf_size=40)
    _, labels = Tree_final.query(data, k=1)
    labels = labels[:, 0]

    return centers, labels, cost