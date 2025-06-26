import pickle
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def draw_dataset(data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Assuming data is a 2D array with shape (n_samples, n_features)
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2] if data.shape[1] > 2 else np.zeros_like(x)  # Use zeros if no z-dimension

    ax.scatter(x, y, z)
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    plt.show()
def read_data(n_node,k):
    mydata = pickle.load(open("radius/mydata2/train/" + str(n_node) + ".pkl", "rb"))
    node_feat = mydata["node_feat"]
    edge_feat = mydata["edge_feat"]
    edge_index = mydata["edge_index"]
    inverse_edge_index = mydata["inverse_edge_index"]
    density_feat = mydata["density_feat"]
    for i in range(k):
        now_data=node_feat[i]
        draw_dataset(now_data)
    return node_feat, edge_feat, edge_index, inverse_edge_index, density_feat
if __name__ == "__main__":
    n_node = 100
    k = 3
    node_feat, edge_feat, edge_index, inverse_edge_index, density_feat = read_data(n_node, k)
    print("Node Feature Shape:", node_feat.shape)
    print("Edge Feature Shape:", edge_feat.shape)
    print("Edge Index Shape:", edge_index.shape)
    print("Inverse Edge Index Shape:", inverse_edge_index.shape)
    print("Density Feature Shape:", density_feat.shape)