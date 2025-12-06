import pickle
import numpy as np

class DataLoader(object):
    def __init__(self, file_path, batch_size,problem="tsp"):
        self.file_path = file_path
        self.batch_size = batch_size
        if problem == "pdp" or problem == "cvrptw":
            #self.n_ranges = 16
            self.n_ranges = 1            
        else:
            #self.n_ranges = 40
            self.n_ranges = 1
        self.problem = problem
        # self.epoch_size = 10000 * self.n_ranges

    def load_data(self,n_nodes=100):
        self.dataset = []
        loading_datasets = []
        loading_datasets.append(n_nodes)
        with open(self.file_path + "/" +'train_merged_' + str(n_nodes) + ".pkl", "rb") as f:
            self.dataset.append(pickle.load(f))
        print ("load datasets wtih nodes " + ", ".join([str(_) for _ in loading_datasets]))
        self.batch_index = 0
        self.dataset_index=0
        dataset = self.dataset[self.dataset_index]
        data_len = dataset["node_feat"].shape[0]
        return data_len

    def next_batch(self):
        if self.problem == "tsp":
            assert self.batch_index < 125 * 40
        elif self.problem == "cvrp":
            assert self.batch_index < 30 * 40
        elif self.problem == "pdp" or self.problem == "cvrptw":
            assert self.batch_index < 60 * 16
        #dataset_index = self.batch_index % self.n_ranges
        dataset = self.dataset[self.dataset_index]
        n_nodes = dataset["node_feat"].shape[1]
        
        # 获取当前数据集的总样本数
        data_len = dataset["node_feat"].shape[0]
        # === 修改开始 ===
        # 优先使用 self.batch_size (我们在 __init__ 里传入的)
        if self.batch_size is not None:
            batch_size = self.batch_size
        else:
            # 如果没有设置，则回退到旧代码的逻辑 (为了兼容旧代码)
            if self.problem == "tsp":
                batch_size = min(16,data_len // 5000) 
                #batch_size = data_len // 125
            elif self.problem == "cvrp":
                batch_size = data_len // 30
            elif self.problem == "pdp" or self.problem == "cvrptw":
                batch_size = data_len // 60
            else:
                # 其他情况的默认处理，防止报错
                batch_size = min(16,data_len // 5000) 
        # === 修改结束 ===
        
        batch_index_inside_dataset = self.batch_index // self.n_ranges
        node_feat = dataset["node_feat"][batch_index_inside_dataset * batch_size : (batch_index_inside_dataset + 1) * batch_size] # b x 100 x 2
        edge_feat = dataset["edge_feat"][batch_index_inside_dataset * batch_size : (batch_index_inside_dataset + 1) * batch_size] # b x 1,0000 x 2
        edge_index = dataset["edge_index"][batch_index_inside_dataset * batch_size : (batch_index_inside_dataset + 1) * batch_size] # b x 100 x 10
        inverse_edge_index = dataset["inverse_edge_index"][batch_index_inside_dataset * batch_size : (batch_index_inside_dataset + 1) * batch_size] # b x 100 x 10

        if self.problem == "pdp" or self.problem == "cvrptw":
            label1 = dataset["label1"][batch_index_inside_dataset * batch_size : (batch_index_inside_dataset + 1) * batch_size] # b x 1000
            label2 = dataset["label2"][batch_index_inside_dataset * batch_size : (batch_index_inside_dataset + 1) * batch_size] # b x 1000
            self.batch_index += 1
            return (node_feat, edge_feat, label1, label2, edge_index, inverse_edge_index)
        elif self.problem == "kmeans":
            label = dataset["density_feat"][batch_index_inside_dataset * batch_size : (batch_index_inside_dataset + 1) * batch_size] # b x 1000
            self.batch_index += 1
            return (node_feat, edge_feat, label, edge_index, inverse_edge_index)
        else:
            label = dataset["density_feat"][batch_index_inside_dataset * batch_size : (batch_index_inside_dataset + 1) * batch_size] # b x 1000
            self.batch_index += 1
            return (node_feat, edge_feat, label, edge_index, inverse_edge_index)


