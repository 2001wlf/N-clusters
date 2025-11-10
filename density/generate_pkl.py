import argparse
import numpy as np
import pickle
from pathlib import Path
from util import densityQuery
import sys
import os
sys.path.append('../')
from density.mydata.generate_dataset import generate_smile, make_spiral, generate_parabola, make_nested_squares
def compute_features(x, density, n_neighbours=20):
    """
    Compute node features (coordinates), edge features (distances),
    edge indices (nearest neighbours), inverse edge indices, and density features.
    x: array of shape (n_nodes, dim)
    density: array of shape (n_nodes,)
    returns: dict of feature arrays ready to pickle
    """
    n_nodes, dim = x.shape
    # Add sample dimension
    x_batch = x[np.newaxis, ...]  # shape: (1, n_nodes, dim)
    # Pairwise distances: (1, n_nodes, n_nodes)
    diff = x_batch.reshape(1, n_nodes, 1, dim) - x_batch.reshape(1, 1, n_nodes, dim)
    dist = np.sqrt((diff ** 2).sum(-1))

    # Nearest neighbours
    edge_index = np.argsort(dist, axis=-1)[:, :, 1:1 + n_neighbours]  # skip self (index 0)
    edge_feat = dist[np.arange(1)[:, None, None], np.arange(n_nodes)[None, :, None], edge_index]

    # Inverse edge index mapping
    inverse_edge_index = -np.ones((1, n_nodes, n_nodes), dtype=int)
    # flat neighbour index: neighbour rank + node_id * n_neighbours
    flat_idx = (np.arange(n_nodes) * n_neighbours)[None, :, None] + np.arange(n_neighbours)[None, None, :]
    inverse_edge_index[np.arange(1)[:, None, None], edge_index, np.arange(n_nodes)[None, :, None]] = flat_idx
    # reduce to shape (1, n_nodes, n_neighbours)
    inverse_edge_index = inverse_edge_index[np.arange(1)[:, None, None], np.arange(n_nodes)[None, :, None], edge_index]

    # Density feature: (1, n_nodes, 1)
    density_feat = density.reshape(1, n_nodes, 1)

    return {
        'node_feat': x_batch,
        'edge_feat': edge_feat,
        'edge_index': edge_index,
        'inverse_edge_index': inverse_edge_index,
        'density_feat': density_feat
    }


def main():
    print("yes")
    parser = argparse.ArgumentParser(
        description='Compute graph features from a user-provided dataset and save to .pkl'
    )
    parser.add_argument(
        '--input_file',
        help='Path to CSV file (no header) containing node coordinates; first row is depot, subsequent rows are customers'
    )
    parser.add_argument(
        '--output_file',
        help='Path where the resulting pickle file will be saved'
    )
    parser.add_argument(
        '--n_neighbours', type=int, default=20,
        help='Number of nearest neighbours to use for edge features'
    )
    args = parser.parse_args()
    print("yes")
    # --- Load data ---
    #xys = np.loadtxt(args.input_file, delimiter=',')
    #if xys.ndim != 2:
     #   raise ValueError(f"Expected 2D array in {args.input_file}, got shape {xys.shape}")
    #暂时使用直接生成的数据集
    data, label = generate_smile(300)
    dataset= 'smile'
    
    #data, label = make_spiral(300)
    #dataset= 'spiral'
    
    #data, label = generate_parabola(300)
    #dataset= 'parabola'
    
    #data, label = make_nested_squares(300)
    #dataset= 'nested_squares'
    
    xys=data
    # --- Compute density ---
    density = densityQuery(xys)

    # --- Compute features ---
    features = compute_features(xys, density, n_neighbours=args.n_neighbours)

    # --- Save to pickle ---
    args.output_file=f'test/{dataset}.pkl'
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(features, f)

    print(f"Saved features for {xys.shape[0]} nodes to {out_path}")


if __name__ == '__main__':
    #print("yes")
    main()
