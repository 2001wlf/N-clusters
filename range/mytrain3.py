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

parser = argparse.ArgumentParser(description='')
parser.add_argument('--n_nodes', type=int, default=100, help='')


parser.add_argument('--file_path', default='mydata2/train', help='')
parser.add_argument('--eval_file_path', default='mydata2/val', help='')
parser.add_argument('--n_epoch', type=int, default=5, help='')
parser.add_argument('--eval_interval', type=int, default=1, help='')
parser.add_argument('--eval_batch_size', type=int, default=20, help='')
parser.add_argument('--n_hidden', type=int, default=128, help='')
parser.add_argument('--n_gcn_layers', type=int, default=30, help='')
parser.add_argument('--n_mlp_layers', type=int, default=3, help='')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='')
parser.add_argument('--save_interval', type=int, default=1, help='')
parser.add_argument('--save_dir', type=str, default="saved/radius_centralpoint_uniform/", help='')
parser.add_argument('--load_pt', type=str, default="", help='')
args = parser.parse_args()

n_edges = 20
n_edges = min(n_edges, args.n_nodes - 1)
net = SparseGCNModel()
net.cuda()
dataLoader = DataLoader(file_path=args.file_path,n_nodes=args.n_nodes,
                        batch_size=None)

edge_cw = None
optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

os.makedirs(args.save_dir, exist_ok=True)

epoch = 0
if args.load_pt:
    saved = torch.load(args.load_pt)
    epoch = saved["epoch"]
    net.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
while epoch < args.n_epoch:
    statistics = {"loss_train": [],
                  "loss_test": []}
    rank_train = [[] for _ in range(20)]
    Norms_train = [[] for _ in range(20)]
    net.train()
    dataset_index = epoch % 10
    dataLoader.load_data(dataset_index)
    for batch in trange(5000):
        node_feat, edge_feat, label, edge_index, inverse_edge_index = dataLoader.next_batch()
        batch_size = node_feat.shape[0]
        node_feat = Variable(torch.FloatTensor(node_feat).type(torch.cuda.FloatTensor), requires_grad=False)
        edge_feat = Variable(torch.FloatTensor(edge_feat).type(torch.cuda.FloatTensor), requires_grad=False).view(batch_size, -1, 1)
        label = Variable(torch.FloatTensor(label).type(torch.cuda.FloatTensor), requires_grad=False).view(batch_size, -1)
        edge_index = Variable(torch.LongTensor(edge_index).type(torch.cuda.LongTensor), requires_grad=False).view(batch_size, -1)
        inverse_edge_index = Variable(torch.LongTensor(inverse_edge_index).type(torch.cuda.LongTensor), requires_grad=False).view(batch_size, -1)
        if type(edge_cw) != torch.Tensor:
            edge_labels = label.cpu().numpy().flatten()
            edge_cw = compute_class_weight("balanced", classes=np.unique(edge_labels), y=edge_labels)
            edge_cw = torch.Tensor(edge_cw).type(torch.cuda.FloatTensor)
        y_edges, loss_nodes, y_nodes = net.forward(node_feat, edge_feat, edge_index, inverse_edge_index, label, edge_cw, n_edges)
        loss_nodes = loss_nodes.mean()

        n_nodes = node_feat.size(1)
        
        loss = loss_nodes

        loss.backward()
        statistics["loss_train"].append(loss.detach().cpu().numpy())
        optimizer.step()
        optimizer.zero_grad()
        y_edges = y_edges.detach().cpu().numpy()
        label = label.cpu().numpy()

    print ("Epoch {} loss {:.7f} rank:".format(epoch, np.mean(statistics["loss_train"])))

    if epoch % args.eval_interval == 0:
        eval_results = []
        for n_node in [args.n_nodes]:
            dataset = pickle.load(open(args.eval_file_path + "/" + str(n_node) + ".pkl", "rb"))
            dataset_rank = []
            dataset_norms = []
            for eval_batch in trange(1000 // args.eval_batch_size):
                node_feat = dataset["node_feat"][eval_batch * args.eval_batch_size:(eval_batch + 1) * args.eval_batch_size]
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
                    y_edges, loss_nodes, y_nodes = net.forward(node_feat, edge_feat, edge_index, inverse_edge_index, label, edge_cw, n_edges)
                    loss_nodes = loss_nodes.mean()

                    y_edges = y_edges.detach().cpu().numpy()
                    label = label.cpu().numpy()
        print("eval: n = 100 | {}".format(loss_nodes))     
        # print ("n=100 %.3f %d, n=200 %.3f %d, n=500 %.3f %d" % (tuple(eval_results)))

    epoch += 1
    if epoch % args.save_interval == 0:
        torch.save({"epoch": epoch, "model": net.state_dict(), "optimizer": optimizer.state_dict()}, args.save_dir + "/" + str(epoch) + ".pt")
