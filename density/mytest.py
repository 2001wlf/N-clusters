import os
import argparse
import numpy as np
from utils.data_loader import DataLoader
import glob
from tqdm import trange
from net.sgcn_model import SparseGCNModel
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.autograd import Variable
import pickle
def entropy_model(my_node):
    #可以考虑先将生成的结构上或者密度值先通过一个文件存储下来，最后让升维的代码读取然后直接做对比/
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--eval_interval', type=int, default=1, help='')
    parser.add_argument('--eval_batch_size', type=int, default=1, help='')
    parser.add_argument('--eval_file_path', default='C:/Users/10998/Desktop/N-clusters/density/mydata/val', help='')
    parser.add_argument('--model_path', type=str, default='C:/Users/10998/Desktop/N-clusters/density/saved_density/exp1/5.pt', help='')
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
        print(dataset["node_feat"].shape[0])
        dataset_norms = []
        os.makedirs("cmps/{}/".format(n_node),exist_ok=True)
        for eval_batch in trange(1 // args.eval_batch_size): #总共1000个数据集，一行是一个，20个数据集一起评测，返回第一个的结果
            node_feat = dataset["node_feat"][eval_batch * args.eval_batch_size:(eval_batch + 1) * args.eval_batch_size]
            edge_feat = dataset["edge_feat"][eval_batch * args.eval_batch_size:(eval_batch + 1) * args.eval_batch_size]
            edge_index = dataset["edge_index"][eval_batch * args.eval_batch_size:(eval_batch + 1) * args.eval_batch_size]
            inverse_edge_index = dataset["inverse_edge_index"][eval_batch * args.eval_batch_size:(eval_batch + 1) * args.eval_batch_size]
            label = dataset["density_feat"][eval_batch * args.eval_batch_size:(eval_batch + 1) * args.eval_batch_size]
            #print("node: ",node_feat[0]);
            #print("density: ",label[0])
            with torch.no_grad():
                node_feat = Variable(torch.FloatTensor(node_feat).type(torch.cuda.FloatTensor), requires_grad=False) # B x 100 x 2
                edge_feat = Variable(torch.FloatTensor(edge_feat).type(torch.cuda.FloatTensor), requires_grad=False).view(args.eval_batch_size, -1, 1) # B x 1000 x 2
                label = Variable(torch.FloatTensor(label).type(torch.cuda.FloatTensor), requires_grad=False).view(args.eval_batch_size, -1) # B x 1000
                edge_index = Variable(torch.FloatTensor(edge_index).type(torch.cuda.FloatTensor), requires_grad=False).view(args.eval_batch_size, -1) # B x 1000
                inverse_edge_index = Variable(torch.FloatTensor(inverse_edge_index).type(torch.cuda.FloatTensor), requires_grad=False).view(args.eval_batch_size, -1) # B x 1000
                n_nodes = node_feat.size(1)
                y_edges, loss_nodes, y_nodes = net.forward(node_feat, edge_feat, edge_index, inverse_edge_index, label, edge_cw, n_edges)
                loss_nodes = loss_nodes.mean()

                y_edges = y_edges.detach().cpu().numpy()
                label = label.cpu().numpy()
                np.savetxt("cmps/{}/{}_label.txt".format(n_node,eval_batch),label[0],fmt='%.4f')
                np.savetxt("cmps/{}/{}_pred.txt".format(n_node,eval_batch),list(y_nodes[0].flatten().cpu()),fmt='%.4f')
    print("eval: n = 100 | {}".format(loss_nodes))