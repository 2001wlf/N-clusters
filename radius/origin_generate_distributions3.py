from util3 import densityQuery,fringeScore,structuralEntorpy,raidusQuery,multiprocess
import pickle
import numpy as np
from time import time
from pathlib import Path
import argparse
from sklearn.cluster import KMeans
import warnings
from mydata2.generate_dataset import generate_smile
warnings.filterwarnings("ignore",category=FutureWarning)

def clustered_xys(args, center_depot=False, max_xy=1):
    uniform_frac = 0.5 if args.mixed else 0.0
    n_uniform = int((args.n_nodes - args.n_c) * uniform_frac)
    n_clustered = args.n_nodes - args.n_c - n_uniform
    uniform_locs = np.random.uniform(0, max_xy, size=(n_uniform, 2))

    assert args.n_c < args.n_nodes
    centers = np.random.uniform(0.2, max_xy - 0.2, size=(args.n_c, 2))

    n_clustered_samples = 0
    all_clustered_locs = []
    while n_clustered_samples < n_clustered:
        center_locs = centers[np.random.randint(len(centers), size=2 * (n_clustered - n_clustered_samples))]
        cluster_locs = np.random.normal(center_locs, args.std_cluster)
        cluster_locs = cluster_locs[(cluster_locs >= 0).all(axis=1) & (cluster_locs < max_xy).all(axis=1)]
        all_clustered_locs.append(cluster_locs)
        n_clustered_samples += len(cluster_locs)
    cluster_locs = np.concatenate(all_clustered_locs)[:n_clustered]
    xys = np.vstack((centers, uniform_locs, cluster_locs))
    if center_depot:
        depot = np.mean(xys, axis=0, keepdims=True)
    else:
        min_x, min_y = np.clip(xys.min(axis=0) - 0.1, 0, max_xy)
        max_x, max_y = np.clip(xys.max(axis=0) + 0.1, 0, max_xy)
        depot = np.array([[np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y)]])
    return np.vstack((depot, xys))


def generate_gaussian_mixture_tsp(graph_size, num_modes=0, cdist=0):
    '''
    Adaptation from AAAI-2022 "Learning to Solve Travelling Salesman Problem with Hardness-Adaptive Curriculum".
    '''

    def gaussian_mixture(graph_size=100, num_modes=0, cdist=1):
        '''
        GMM create one instance of TSP-100, using cdist
        '''
        from sklearn.preprocessing import MinMaxScaler
        nums = np.random.multinomial(graph_size, np.ones(num_modes) / num_modes)
        xy = []
        for num in nums:
            center = np.random.uniform(0, cdist, size=(1, 2))
            nxy = np.random.multivariate_normal(mean=center.squeeze(), cov=np.eye(2, 2), size=(num,))
            xy.extend(nxy)
        xy = np.array(xy)
        xy = MinMaxScaler().fit_transform(xy)
        return xy

    if num_modes == 0:  # (0, 0) - uniform
        return np.random.uniform(0, 1, [graph_size, 2])
    else:
        return np.array(gaussian_mixture(graph_size=graph_size, num_modes=num_modes, cdist=cdist))

def my_generate_problem(xys,k):
    #density = densityQuery(xys)
    #fringe = fringeScore(xys)
    #SE = structuralEntorpy(xys)
    #SE = np.multiply(SE, 10)
    kmeans= KMeans(init='k-means++', n_clusters=k).fit(xys)
    Radius=raidusQuery(xys,kmeans.labels_)
    return xys,Radius
def generate_problem(args, init=None):
    if init:
        xys, demands, capacity, pkwargs = init
    else:
        if args.dist == "uniform":
            xys = clustered_xys(args) if args.n_c else np.random.uniform(0, 1, size=(1 + args.n_nodes, 2))
            demands = np.random.randint(args.min_demand, args.max_demand, size=1 + args.n_nodes)
            demands[0] = 0
        elif args.dist == "gm":
            training_set = [(0, 0), (3, 10), (3, 30), (3, 50), (5, 10), (5, 30), (5, 50), (7, 10), (7, 30), (7, 50)]
            if args.partition != "train":
                num_modes, cdist = training_set[args.id % len(training_set)]
            else:
                # num_modes, cdist = training_set[args.id // 50]
                num_modes, cdist = training_set[args.id % len(training_set)]
            xys = generate_gaussian_mixture_tsp(1 + args.n_nodes, num_modes, cdist)
            demands = np.random.randint(args.min_demand, args.max_demand, size=1 + args.n_nodes)
            demands[0] = 0

    #density = densityQuery(xys)
    #fringe = fringeScore(xys)
    #SE = structuralEntorpy(xys)
    #SE = np.multiply(SE, 10)
    density=[]
    fringe=[]
    SE=[]
    kmeans= KMeans(init='k-means++', n_clusters=args.n_clusters).fit(xys)
    Radius=raidusQuery(xys,kmeans.labels_)
    return xys, density, fringe, SE,Radius

def generate_i(gen_args):
    i, seed, args, init = gen_args
    np.random.seed(seed)
    start_time = time()
    # print(f'Generating problem {i}...')
    args.id = i
    p, q, f, s,r = generate_problem(args, init)

    total_time = time() - start_time
    # print(f'Problem {i} took {total_time:.4f} seconds')
    return p, q, f, s,r

def generate_datasets():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=Path)
    parser.add_argument('--partition', type=str, choices=['train', 'val', 'test'])
    parser.add_argument('--n_nodes', type=int,default=1000)
    parser.add_argument('--n_c', type=int, default=0, help='Number of city clusters in the problem instance')
    parser.add_argument('--mixed', action='store_true')
    parser.add_argument('--std_cluster', type=float, default=0.07, help='Standard deviation for normal distribution of city clusters')
    parser.add_argument('--ptype', type=str, default='CVRP', choices=['CVRP', 'CVRPTW', 'VRPMPD'])
    parser.add_argument('--n_instances', type=int, default=None)
    parser.add_argument('--n_clusters', type=int, default=10)
    parser.add_argument('--n_lkh_trials', type=int, default=100)
    parser.add_argument('--min_demand', type=int, default=1, help='Inclusive')
    parser.add_argument('--max_demand', type=int, default=10, help='Exclusive')
    parser.add_argument('--capacity', type=int, default=50)
    parser.add_argument('--service_time', type=float, default=0.02)
    parser.add_argument('--max_window_width', type=float, default=1.5)
    parser.add_argument('--pickup_every', type=int, default=2)
    parser.add_argument('--n_cpus', type=int, default=40)
    parser.add_argument('--n_process', type=int, default=None)
    parser.add_argument('--n_threads_per_process', type=int, default=None)
    parser.add_argument('--solver', type=str, choices=['LKH', 'HGS'], default='LKH')
    parser.add_argument('--naive_init', action='store_true')
    parser.add_argument('--full_solver_init', action='store_true')
    parser.add_argument('--dist', type=str, choices=['uniform', 'gm'], default='gm')  # (0, 0) + {3, 5, 7} * {10, 30, 50}
    args = parser.parse_args()
    args.partition='val'
    args.save_dir="mydata2"
    #args.save_dir.mkdir(parents=True, exist_ok=True)
    #现在是测试用
    #args.n_instances=1;
    args.n_instances = args.n_instances or (1000 if args.partition == 'train' else 1000)

    ref_path = ref_problems = None
    #args.n_nodes=x.shape[0]
    save_path = "{}/{}/{}.pkl".format(args.save_dir,args.partition,args.n_nodes)
    #n_nodes=args.n_nodes
    n_nodes = args.n_nodes + 1
    n_neighbours = 20
    n_samples = args.n_instances
    
    # print(f'Generating to {save_path}', flush=True)
    # print(f'Generating {args.n_instances} {args.ptype} instances from {"uniform distribution" if args.n_c == 0 else f"mixed distribution with {args.n_c} city clusters" if args.mixed else f"clustered distribution with {args.n_c} city_clusters"}, each with {args.n_clusters} radial sections to run LKH subsolver on', flush=True)

    results = multiprocess(generate_i, list(zip(
        range(0, args.n_instances),
        np.random.randint(np.iinfo(np.int32).max, size=args.n_instances),
        [args] * args.n_instances,
        [None] * args.n_instances,
    )), cpus=args.n_process or (args.n_cpus - 1) // args.n_clusters + 1)
    #读取数据集
    #笑脸
    #print(x.shape)
    #k=len(np.unique(label))
    #x,Radius = my_generate_problem(x,k)
    x,q,frige,s,Radius = zip(*results)
    x = np.array(x)

    dist = x.reshape(n_samples, n_nodes, 1, 2) - x.reshape(n_samples, 1, n_nodes, 2)
    dist = np.sqrt((dist ** 2).sum(-1))
    edge_index = np.argsort(dist, -1)[:, :, 1:1 + n_neighbours]
    edge_feat = dist[np.arange(n_samples).reshape(-1, 1, 1), np.arange(n_nodes).reshape(1, -1, 1), edge_index]

    dist = x.reshape(n_samples, n_nodes, 1, 2) - x.reshape(n_samples, 1, n_nodes, 2)
    dist = np.sqrt((dist ** 2).sum(-1)) # 10000 x 100 x 100
    edge_index = np.argsort(dist, -1)[:, :, 1:1 + n_neighbours]
    inverse_edge_index = -np.ones(shape=[n_samples, n_nodes, n_nodes], dtype="int")
    inverse_edge_index[np.arange(n_samples).reshape(-1, 1, 1), edge_index, np.arange(n_nodes).reshape(1, -1, 1)] = np.arange(n_neighbours).reshape(1, 1, -1) + np.arange(n_nodes).reshape(1, -1, 1) * n_neighbours
    inverse_edge_index = inverse_edge_index[np.arange(n_samples).reshape(-1, 1, 1), np.arange(n_nodes).reshape(1, -1, 1), edge_index]

    # print(len(xys),len(xys[0]),len(xys[0][0]))
    # print(len(density),len(density[0]),density[0][0])
    # pkwargs = {k: np.array([pk[k] for pk in pkwargs]) for k in pkwargs[0]}

    x=x.reshape(n_samples,n_nodes,2);
    print(x.shape);
    print(edge_feat.shape);
    print(edge_index.shape)
    print(inverse_edge_index.shape);
    print(np.array(Radius).reshape(-1,n_nodes,1).shape)
    feat = {"node_feat": x, # n_samples x n_nodes x 2
            "edge_feat":edge_feat, # n_samples x n_nodes x n_neighbours
            "edge_index":edge_index, # n_samples x n_nodes x n_neighbours
            "inverse_edge_index":inverse_edge_index, # n_samples x n_nodes x n_neighbours
            "density_feat":np.array(Radius).reshape(-1,n_nodes,1) # n_samples x n_nodes x 1
            }
    with open(save_path, "wb") as f:
        pickle.dump(feat, f)
if __name__ == "__main__":
    generate_datasets()
    print("hello wolld")