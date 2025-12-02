from mytest3 import entropy_model
from util3 import densityQuery,fringeScore,structuralEntorpy
import pickle
import numpy as np
from time import time
import math
import os
import copy
import sys
import time
import pickle
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
import glob
from FSEO import SummaryOutliers
import numpy as np
import networkx as nx
import pandas as pd
import sklearn
from scipy.spatial import distance
from sklearn.cluster import KMeans

from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
from sklearn.preprocessing import MinMaxScaler

# from BDQLSH import projection_overlap
#from kmeans import KMEANS

from drug.lib.coding_tree import PartitionTree
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
from sklearn.metrics import normalized_mutual_info_score
import argparse
from drug.treeGeneration import *
from pathlib import Path
import argparse
import warnings
from generate_distributions3 import generate_datasets
from mydata2.generate_dataset import generate_smile,generate_parabola,make_spiral,make_nested_squares
def main(dataset,data,label):
    datas=[]
    #n_node=300
    #获取模型的结构熵结果
    #dataset="drug"
    #data,label=generate_smile(75)
    k=len(np.unique(label))
    
    #dataset='wine'
    #iris = load_wine()
    #data = iris.data[:,:2]
    #label = iris.target
    #k = len(np.unique(label))
    
    n_node=len(data)
    generate_datasets(data,label)
    datas.append([dataset,data,label,k])
    entropy_model(n_node)
    #取出模型结构熵结果
    #with open(f'C:/Users/10998/Desktop/N-clusters/entropy/cmps/{n_node}/0_pred.txt', 'r', encoding='utf-8') as f:
     #   entropy_results = f.read()
    entropy_results=np.loadtxt(f'C:/Users/10998/Desktop/N-clusters/entropy/cmps/{n_node}/0_pred.txt')
    #entropy_results.reshape(-1,1)
    if isinstance(entropy_results, list):
        entropy_results = np.array(entropy_results)
    print("se shape ",type(entropy_results))
    entropy_results=entropy_results.reshape(-1,1)
    print(entropy_results.shape)
    parser = argparse.ArgumentParser(description='encoding k-means')
    parser.add_argument('--dataset', '-d', type=str, default=False, help='dataset')
    parser.add_argument('--datadir', '-dr', type=str, default=False, help='datasetdir')
    parser.add_argument('--KNN_start', '-ks', type=int, default=4, help='minimun k of KNN')
    parser.add_argument('--KNN_end', '-ke', type=int, default=4, help='maximun k of KNN')
    parser.add_argument('--KNN_step', '-kp', type=int, default=1, help='k step')
    parser.add_argument('--tree_depth_start', '-tds', type=int, default=2, help='minimun depth of encoding tree')
    parser.add_argument('--tree_depth_end', '-tde', type=int, default=0, help='maximun depth of encoding tree')
    parser.add_argument('--tree_depth_step', '-tdp', type=int, default=1, help='depth step')
    parser.add_argument('--gap_start', '-gs', type=float, default=1.0, help='minimun gap')
    parser.add_argument('--gap_end', '-ge', type=float, default=1.0, help='maximun gap')
    parser.add_argument('--gap_number', '-gn', type=int, default=1, help='gap number')
    args = parser.parse_args()

    for dataset, data, label, k in datas:
        sample_indices_, assignment, weights_ = SummaryOutliers(data, 200, 1.0, 0.4)
        data = data[sample_indices_]
        label=label[sample_indices_]
        entropy_results=entropy_results[sample_indices_]
        k=len(np.unique(label))
        print('-'*120)
        print(dataset)
        print(data.shape)
        ne_start = args.KNN_start
        ne_end = args.KNN_end + 1
        ne_step = args.KNN_step
        ne_list = list(range(ne_start, ne_end, ne_step))
        tree_depth_start = args.tree_depth_start
        tree_depth_end = args.tree_depth_end + 1
        if tree_depth_end == 1:
            tree_depth_end = math.ceil(math.log(len(data), 2))
        tree_depth_step = args.tree_depth_step
        tree_depth_list = list(range(tree_depth_start, tree_depth_end, tree_depth_step))
        gap_range_start = args.gap_start
        gap_range_end = args.gap_end
        gap_range_number = args.gap_number
        gap_range = np.linspace(gap_range_start, gap_range_end, gap_range_number)

        #
        result = pd.DataFrame(columns=['dataset', 'KNN', 'tree_depth', 'gap', 'ACC', 'NMI', 'note'])

        kmeans = KMeans(init='k-means++', n_clusters=k).fit(data)
        print('k-means++ NMI',round(normalized_mutual_info_score(label,kmeans.labels_), 4))
        print('k-means++ ACC:', round(acc(label,kmeans.labels_),4))
        opt = kmeans.inertia_

        maximun_acc = 0
        average_acc =0
        reg_average_acc =0
        average_nmi =0
        reg_average_nmi =0
        reg_maximun_acc = 0
        maximun_acc_index = []
        reg_maximun_acc_index = []
        gap_acc_max = 0
        reg_gap_acc_max = 0
        ne_acc_max = 0
        reg_ne_acc_max = 0
        tree_depth_acc_max = 0
        reg_tree_depth_acc_max = 0
        ACC_max_range = []
        reg_ACC_max_range = []
        maximun_nmi = 0
        reg_maximun_nmi = 0
        maximun_nmi_index = []
        reg_maximun_nmi_index = []
        gap_nmi_max = 0
        reg_gap_nmi_max = 0
        ne_nmi_max = 0
        reg_ne_nmi_max = 0
        tree_depth_nmi_max = 0
        reg_tree_depth_nmi_max = 0
        NMI_max_range = []
        reg_NMI_max_range = []

        all = len(ne_list) * len(tree_depth_list) * gap_range_number
        ne_list=[3,4,5,6,7,8,9,10]
        tree_depth_list=[2,3,4]
        gap_range=[0.1,0.2,0.3,0.4,0.5]
        ne_list=[3]
        tree_depth_list=[2]
        gap_range=[0.1]
        print("Data shape is",data.shape)
        start_time = time.time()
        for p in range(len(ne_list)):
            res_time = 0
            ne = ne_list[p]
            print(ne)
            for q in range(len(tree_depth_list)):
                tree_depth = tree_depth_list[q]
                for l in range(len(gap_range)):
                    #print("gap is ",l)
                    gap = gap_range[l]
                    s_time = time.time()
                    
                    data_entory=entropy_results
                    #data_entory.reshape(-1,1)
                        #print("no")
                    #print("se shape ",se.shape)
                    data_ent = np.sum(data_entory, axis=1).reshape(-1, 1)

                    # min_gap = find_min_gap(data_ent)
                    data_ent_uniform = make_uniform(np.min(data_ent), np.max(data_ent), 19, data_ent)

                    data_ = np.hstack((data,data_entory))

                    
                    data_ = MinMaxScaler().fit_transform(data_)
                    # data_[:,2] = data_[:,2]*2

                    kmeans_2 = KMeans(init='k-means++', n_clusters=k).fit(data_entory)
                    # a = kmeans_2.labels_
                    ACC_2 = round(acc(label, kmeans_2.labels_), 4)
                    NMI_2 = round(normalized_mutual_info_score(label, kmeans_2.labels_), 4)
                    note = 'only entroy_list'
                    # print('KNN:', ne, '\ttree_depth:', tree_depth, '\tACC:', ACC_2, '\tNMI:', NMI_2, '\t' + note)
                    new = pd.DataFrame(
                        [[dataset, ne, tree_depth, gap, ACC_2, NMI_2, note]],
                        columns=['dataset', 'KNN', 'tree_depth', 'gap', 'ACC', 'NMI', 'note'])
                    result = pd.concat([result, new])
                    #print("yes")
                    reg_kmeans_2 = RegularizedKMeans(n_clusters=k).fit(data_entory)
                    # a = kmeans_2.labels_
                    reg_ACC_2 = round(acc(label, reg_kmeans_2.labels_), 4)
                    reg_NMI_2 = round(normalized_mutual_info_score(label, reg_kmeans_2.labels_), 4)
                    note = 'reg only entroy_list'
                    # print('KNN:', ne, '\ttree_depth:', tree_depth, '\tACC:', ACC_2, '\tNMI:', NMI_2, '\t' + note)
                    new = pd.DataFrame(
                        [[dataset, ne, tree_depth, gap, reg_ACC_2, reg_NMI_2, note]],
                        columns=['dataset', 'KNN', 'tree_depth', 'gap', 'ACC', 'NMI', 'note'])
                    result = pd.concat([result, new])

                    # kmeans_3 = KMeans(init='k-means++', n_clusters=k).fit((data_entory - np.min(data_entory,axis=0))/(np.max(data_entory,axis=0)-np.min(data_entory,axis=0)))
                    kmeans_3 = KMeans(init='k-means++', n_clusters=k).fit(MinMaxScaler().fit_transform(data_entory))
                    ACC_3 = round(acc(label, kmeans_3.labels_), 4)
                    NMI_3 = round(normalized_mutual_info_score(label, kmeans_3.labels_), 4)
                    note = 'only normalized entroy_list'
                    # print('KNN:', ne, '\ttree_depth:', tree_depth, '\tACC:', ACC_3, '\tNMI:', NMI_3, '\t' + note)
                    new = pd.DataFrame(
                        [[dataset, ne, tree_depth, gap, ACC_3, NMI_3, note]],
                        columns=['dataset', 'KNN', 'tree_depth', 'gap', 'ACC', 'NMI', 'note'])
                    result = pd.concat([result, new])
                    #print("yes")
                    reg_kmeans_3 = RegularizedKMeans(n_clusters=k).fit(MinMaxScaler().fit_transform(data_entory))
                    reg_ACC_3 = round(acc(label, reg_kmeans_3.labels_), 4)
                    reg_NMI_3 = round(normalized_mutual_info_score(label, reg_kmeans_3.labels_), 4)
                    note = 'reg only normalized entroy_list'
                    # print('KNN:', ne, '\ttree_depth:', tree_depth, '\tACC:', ACC_3, '\tNMI:', NMI_3, '\t' + note)
                    new = pd.DataFrame(
                        [[dataset, ne, tree_depth, gap, reg_ACC_3, reg_NMI_3, note]],
                        columns=['dataset', 'KNN', 'tree_depth', 'gap', 'ACC', 'NMI', 'note'])
                    result = pd.concat([result, new])


                    kmeans_4 = KMeans(init='k-means++', n_clusters=k).fit(data_ent)
                    ACC_4 = round(acc(label, kmeans_4.labels_), 4)
                    NMI_4 = round(normalized_mutual_info_score(label, kmeans_4.labels_), 4)
                    note = 'only entroy'
                    # print('KNN:', ne, '\ttree_depth:', tree_depth, '\tACC:', ACC_4, '\tNMI_1=4:', NMI, '\t' + note)
                    new = pd.DataFrame(
                        [[dataset, ne, tree_depth, gap, ACC_4, NMI_4, note]],
                        columns=['dataset', 'KNN', 'tree_depth', 'gap', 'ACC', 'NMI', 'note'])
                    result = pd.concat([result, new])
                    #print("yes")
                    reg_kmeans_4 = RegularizedKMeans(n_clusters=k).fit(data_ent)
                    reg_ACC_4 = round(acc(label, reg_kmeans_4.labels_), 4)
                    reg_NMI_4 = round(normalized_mutual_info_score(label, reg_kmeans_4.labels_), 4)
                    note = 'reg only entroy'
                    # print('KNN:', ne, '\ttree_depth:', tree_depth, '\tACC:', ACC_4, '\tNMI_1=4:', NMI, '\t' + note)
                    new = pd.DataFrame(
                        [[dataset, ne, tree_depth, gap, reg_ACC_4, reg_NMI_4, note]],
                        columns=['dataset', 'KNN', 'tree_depth', 'gap', 'ACC', 'NMI', 'note'])
                    result = pd.concat([result, new])

                    # kmeans_5 = KMeans(init='k-means++', n_clusters=k).fit((data_ent - np.min(data_ent,axis=0))/(np.max(data_ent,axis=0)-np.min(data_ent,axis=0)))
                    kmeans_5 = KMeans(init='k-means++', n_clusters=k).fit(MinMaxScaler().fit_transform(data_ent))
                    ACC_5 = round(acc(label, kmeans_5.labels_), 4)
                    NMI_5 = round(normalized_mutual_info_score(label, kmeans_5.labels_), 4)
                    # np.savetxt("rings_ent.txt",data_ent,fmt="%f")
                    note = 'only normalized entroy'
                    # print('KNN:', ne, '\ttree_depth:', tree_depth, '\tACC:', ACC_5, '\tNMI:', NMI_5, '\t' + note)
                    new = pd.DataFrame(
                        [[dataset, ne, tree_depth, gap, ACC_5, NMI_5, note]],
                        columns=['dataset', 'KNN', 'tree_depth', 'gap', 'ACC', 'NMI', 'note'])
                    result = pd.concat([result, new])
                    #print("yes")
                    reg_kmeans_5 = RegularizedKMeans(n_clusters=k).fit(MinMaxScaler().fit_transform(data_ent))
                    reg_ACC_5 = round(acc(label, reg_kmeans_5.labels_), 4)
                    reg_NMI_5 = round(normalized_mutual_info_score(label, reg_kmeans_5.labels_), 4)
                    # np.savetxt("rings_ent.txt",data_ent,fmt="%f")
                    note = 'reg only normalized entroy'
                    # print('KNN:', ne, '\ttree_depth:', tree_depth, '\tACC:', ACC_5, '\tNMI:', NMI_5, '\t' + note)
                    new = pd.DataFrame(
                        [[dataset, ne, tree_depth, gap, reg_ACC_5, reg_NMI_5, note]],
                        columns=['dataset', 'KNN', 'tree_depth', 'gap', 'ACC', 'NMI', 'note'])
                    result = pd.concat([result, new])


                    kmeans_6 = KMeans(init='k-means++', n_clusters=k).fit(data_)
                    ACC_6 = round(acc(label, kmeans_6.labels_), 4)
                    NMI_6 = round(normalized_mutual_info_score(label, kmeans_6.labels_), 4)
                    note = 'normalized data with entroy_list'
                    # print('KNN:', ne, '\ttree_depth:', tree_depth, '\tACC:', ACC_6, '\tNMI_6:', NMI, '\t' + note)
                    new = pd.DataFrame(
                        [[dataset, ne, tree_depth, gap, ACC_6, NMI_6, note]],
                        columns=['dataset', 'KNN', 'tree_depth', 'gap', 'ACC', 'NMI', 'note'])
                    result = pd.concat([result, new])
                    #print("yes")
                    reg_kmeans_6= RegularizedKMeans(n_clusters=k).fit(data_)
                    reg_ACC_6 =round(acc(label,reg_kmeans_6.labels_),4)
                    reg_NMI_6 =round(normalized_mutual_info_score(label, reg_kmeans_6.labels_), 4)
                    note ='reg normalized data with entroy_list'
                    new = pd.DataFrame(
                        [[dataset, ne, tree_depth, gap, reg_ACC_6, reg_NMI_6, note]],
                        columns=['dataset', 'KNN', 'tree_depth', 'gap', 'ACC', 'NMI', 'note'])
                    result = pd.concat([result, new])

                    data_[:, -1] = data_[:, -1] * 2

                    kmeans_7 = KMeans(init='k-means++', n_clusters=k).fit(data_)
                    ACC_7 = round(acc(label, kmeans_7.labels_), 4)
                    NMI_7 = round(normalized_mutual_info_score(label, kmeans_7.labels_), 4)
                    note = 'normalized data with twice entroy_list'
                    # print('KNN:', ne, '\ttree_depth:', tree_depth, '\tACC:', ACC_7, '\tNMI:', NMI_7, '\t' + note)
                    new = pd.DataFrame(
                        [[dataset, ne, tree_depth, gap, ACC_7, NMI_7, note]],
                        columns=['dataset', 'KNN', 'tree_depth', 'gap', 'ACC', 'NMI', 'note'])
                    result = pd.concat([result, new])
                    #print("yes")
                    reg_kmeans_7 = RegularizedKMeans(n_clusters=k).fit(data_)
                    reg_ACC_7 = round(acc(label, reg_kmeans_7.labels_), 4)
                    reg_NMI_7 = round(normalized_mutual_info_score(label, reg_kmeans_7.labels_), 4)
                    note = 'reg normalized data with twice entroy_list'
                    # print('KNN:', ne, '\ttree_depth:', tree_depth, '\tACC:', ACC_7, '\tNMI:', NMI_7, '\t' + note)
                    new = pd.DataFrame(
                        [[dataset, ne, tree_depth, gap, reg_ACC_7, reg_NMI_7, note]],
                        columns=['dataset', 'KNN', 'tree_depth', 'gap', 'ACC', 'NMI', 'note'])
                    result = pd.concat([result, new])
                    
                    data_ = np.hstack((data, data_ent))
                    

                    kmeans_9 = KMeans(init='k-means++', n_clusters=k).fit(MinMaxScaler().fit_transform(data_))
                    ACC_9 = round(acc(label, kmeans_9.labels_), 4)
                    NMI_9 = round(normalized_mutual_info_score(label, kmeans_9.labels_), 4)
                    note = 'normalized data with entroy'
                    # print('KNN:', ne, '\ttree_depth:', tree_depth, '\tACC:', ACC_9, '\tNMI_9:', NMI, '\t' + note)
                    new = pd.DataFrame(
                        [[dataset, ne, tree_depth, gap, ACC_9, NMI_9, note]],
                        columns=['dataset', 'KNN', 'tree_depth', 'gap', 'ACC', 'NMI', 'note'])
                    result = pd.concat([result, new])
                    #print("yes")
                    reg_kmeans_9 = RegularizedKMeans(n_clusters=k).fit(MinMaxScaler().fit_transform(data_))
                    reg_ACC_9 = round(acc(label, reg_kmeans_9.labels_), 4)
                    reg_NMI_9 = round(normalized_mutual_info_score(label, reg_kmeans_9.labels_), 4)
                    note = 'reg normalized data with entroy'
                    # print('KNN:', ne, '\ttree_depth:', tree_depth, '\tACC:', ACC_9, '\tNMI_9:', NMI, '\t' + note)
                    new = pd.DataFrame(
                        [[dataset, ne, tree_depth, gap, reg_ACC_9, reg_NMI_9, note]],
                        columns=['dataset', 'KNN', 'tree_depth', 'gap', 'ACC', 'NMI', 'note'])
                    result = pd.concat([result, new])
                    

                    data_ent_ = MinMaxScaler().fit_transform(data_ent) + 1e-15
                    data_ = np.hstack((data, np.log(data_ent_)))
                    
                    data_ = MinMaxScaler().fit_transform(data_)
                    kmeans_11 = KMeans(init='k-means++', n_clusters=k).fit(data_)
                    # kmeans_ = KMEANS().kmeans_plusplus(k,(data_ - np.min(data_, axis=0)) / (np.max(data_, axis=0) - np.min(data_, axis=0)))
                    ACC_11 = round(acc(label, kmeans_11.labels_), 4)
                    NMI_11 = round(normalized_mutual_info_score(label, kmeans_11.labels_), 4)
                    note = 'normalized data with log entroy'
                    # print('KNN:', ne, '\ttree_depth:', tree_depth, '\tACC:', ACC_11, '\tNMI:', NMI_11, '\t' + note)
                    new = pd.DataFrame(
                        [[dataset, ne, tree_depth, gap, ACC_11, NMI_11, note]],
                        columns=['dataset', 'KNN', 'tree_depth', 'gap', 'ACC', 'NMI', 'note'])
                    result = pd.concat([result, new])
                    #print("yes")
                    reg_kmeans_11 = RegularizedKMeans(n_clusters=k).fit(data_)
                    # kmeans_ = KMEANS().kmeans_plusplus(k,(data_ - np.min(data_, axis=0)) / (np.max(data_, axis=0) - np.min(data_, axis=0)))
                    reg_ACC_11 = round(acc(label, reg_kmeans_11.labels_), 4)
                    reg_NMI_11 = round(normalized_mutual_info_score(label, reg_kmeans_11.labels_), 4)
                    note = 'reg normalized data with log entroy'
                    # print('KNN:', ne, '\ttree_depth:', tree_depth, '\tACC:', ACC_11, '\tNMI:', NMI_11, '\t' + note)
                    new = pd.DataFrame(
                        [[dataset, ne, tree_depth, gap, reg_ACC_11, reg_NMI_11, note]],
                        columns=['dataset', 'KNN', 'tree_depth', 'gap', 'ACC', 'NMI', 'note'])
                    result = pd.concat([result, new])

                    

                    data_ = np.hstack((data, data_ent_uniform))
                    # data_ = (data_ - np.min(data_, axis=0)) / (np.max(data_, axis=0) - np.min(data_, axis=0))
                    # data_[np.isnan(data_)] = 0
                    #print("yes")
                    kmeans_12 = KMeans(init='k-means++', n_clusters=k).fit(MinMaxScaler().fit_transform(data_))
                    ACC_12 = round(acc(label, kmeans_12.labels_), 4)
                    NMI_12 = round(normalized_mutual_info_score(label, kmeans_12.labels_), 4)
                    note = 'normalized data with uniform entroy'
                    # print('KNN:', ne, '\ttree_depth:', tree_depth, '\tACC:', ACC_12, '\tNMI:', NMI_12, '\t' + note)
                    new = pd.DataFrame(
                        [[dataset, ne, tree_depth, gap, ACC_12, NMI_12, note]],
                        columns=['dataset', 'KNN', 'tree_depth', 'gap', 'ACC', 'NMI', 'note'])
                    result = pd.concat([result, new])
                    #print("yes")
                    reg_kmeans_12 = RegularizedKMeans(n_clusters=k).fit(MinMaxScaler().fit_transform(data_))
                    reg_ACC_12 = round(acc(label, reg_kmeans_12.labels_), 4)
                    reg_NMI_12 = round(normalized_mutual_info_score(label, reg_kmeans_12.labels_), 4)
                    note = 'reg normalized data with uniform entroy'
                    # print('KNN:', ne, '\ttree_depth:', tree_depth, '\tACC:', ACC_12, '\tNMI:', NMI_12, '\t' + note)
                    new = pd.DataFrame(
                        [[dataset, ne, tree_depth, gap, reg_ACC_12, reg_NMI_12, note]],
                        columns=['dataset', 'KNN', 'tree_depth', 'gap', 'ACC', 'NMI', 'note'])
                    result = pd.concat([result, new])
                    #print("yes")
                    ACC_list = np.array([ACC_2, ACC_3, ACC_4, ACC_5, ACC_6, ACC_7, ACC_9, ACC_11, ACC_12])
                    reg_ACC_list = np.array([reg_ACC_2, reg_ACC_3, reg_ACC_4, reg_ACC_5, reg_ACC_6, reg_ACC_7, reg_ACC_9, reg_ACC_11, reg_ACC_12])
                    ACC_max = np.max(ACC_list)
                    reg_ACC_max = np.max(reg_ACC_list)
                    ACC_max_range.append(ACC_max)
                    reg_ACC_max_range.append(reg_ACC_max)
                    ACC_max_index = np.argmax(ACC_list)
                    reg_ACC_max_index = np.argmax(reg_ACC_list)
                    NMI_list = np.array(
                        [NMI_2, NMI_3, NMI_4, NMI_5, NMI_6, NMI_7, NMI_9, NMI_11, NMI_12])
                    reg_NMI_list = np.array(
                        [reg_NMI_2, reg_NMI_3, reg_NMI_4, reg_NMI_5, reg_NMI_6, reg_NMI_7, reg_NMI_9, reg_NMI_11, reg_NMI_12])
                    NMI_max = np.max(NMI_list)
                    reg_NMI_max = np.max(reg_NMI_list)
                    NMI_max_range.append(NMI_max)
                    reg_NMI_max_range.append(reg_NMI_max)
                    NMI_max_index = np.argmax(NMI_list)
                    reg_NMI_max_index = np.argmax(reg_NMI_list)
                    if NMI_max>maximun_nmi:
                        maximun_nmi = NMI_max
                        maximun_nmi_index = NMI_max_index
                        gap_nmi_max = gap
                        ne_nmi_max = ne
                        tree_depth_nmi_max = tree_depth
                    if reg_NMI_max>reg_maximun_nmi:
                        reg_maximun_nmi = reg_NMI_max
                        reg_maximun_nmi_index = reg_NMI_max_index
                        reg_gap_nmi_max = gap
                        reg_ne_nmi_max = ne
                        reg_tree_depth_nmi_max = tree_depth
                    if ACC_max>maximun_acc:
                        maximun_acc = ACC_max
                        maximun_acc_index = ACC_max_index
                        gap_acc_max = gap
                        ne_acc_max = ne
                        tree_depth_acc_max = tree_depth
                    if reg_ACC_max>reg_maximun_acc:
                        reg_maximun_acc = reg_ACC_max
                        reg_maximun_acc_index = reg_ACC_max_index
                        reg_gap_acc_max = gap
                        reg_ne_acc_max = ne
                        reg_tree_depth_acc_max = tree_depth
                    now = p * len(tree_depth_list) * gap_range_number + q * gap_range_number + l + 1
                    res_time = (time.time() - start_time) * (all / now - 1)
                    print("\r", "已测试完成KNN={},depth={},gap={}\t进度[{}]{:.5f}%({}/{}.耗时{:.2f}s,还需{:.2f}s)".format(
                        ne, tree_depth, gap,
                        '|' + '■|' * int(now / all * 10 + 1) + ' |' * (
                                10 - int(now / all * 10 + 1)), now / all * 100, now,
                        all, time.time() - start_time, res_time), end='',
                          flush=True)
        print("\n最大ACC:", maximun_acc, "\t方法:", maximun_acc_index, "\tKNN:", ne_acc_max, "\ttree_depth:",
              tree_depth_acc_max, "\tgap:", gap_acc_max)
        print("最大NMI:", maximun_nmi, "\t方法:", maximun_nmi_index, "\tKNN:", ne_nmi_max, "\ttree_depth:",
              tree_depth_nmi_max, "\tgap:", gap_nmi_max)
        print("\n最大reg_ACC:", reg_maximun_acc, "\t方法:", reg_maximun_acc_index, "\tKNN:", reg_ne_acc_max, "\ttree_depth:",
              reg_tree_depth_acc_max, "\tgap:", reg_gap_acc_max)
        print("最大reg_NMI:", reg_maximun_nmi, "\t方法:", reg_maximun_nmi_index, "\tKNN:", reg_ne_nmi_max, "\ttree_depth:",
              reg_tree_depth_nmi_max, "\tgap:", reg_gap_nmi_max)   
    return 

data_dir = "C:/Users/10998/Desktop/N-clusters/entropy/mydata2/transformed_dataset_csv"  # 根据你的实际路径修改
datasets = []
if __name__ == '__main__':
    #给出数据集
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".txt"):
            file_path = os.path.join(data_dir, file_name)
            df = pd.read_csv(file_path, header=None, sep=r'\s+|,', engine='python')
            data = df.iloc[:, :-1].to_numpy()  # 所有特征列
            if data.shape[1]>2:
                data=data[:,:2]
            label = df.iloc[:, -1].to_numpy()  # 最后一列是标签
            main(file_name,data,label)