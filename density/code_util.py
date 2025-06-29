import numpy as np
from sklearn.datasets import make_moons, make_circles
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
def draw_picture(centers,true_data,pred_label, dataset_name):

    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(pred_label)
    N=true_data.shape[0]
    for label in unique_labels:
        indices = np.where(pred_label == label)[0]
        plt.scatter(true_data[indices, 0], true_data[indices, 1], label=f'Cluster {label}')
    plt.scatter(true_data[centers, 0], true_data[centers, 1], color='black', marker='x', s=100, label='Centers')
    plt.title(f'Clustering Results for {dataset_name}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid()
    plt.savefig(f'cmps/{N}/{dataset_name}_clustering_results.png')
    #plt.show()
def generate_subgaussian(n_node, lower=-3, upper=3):
        """
        Generate data from a truncated normal distribution (sub-Gaussian).
        """
        data = np.random.normal(0, 1, size=(n_node, 2))
        mask = (data < lower) | (data > upper)
        row_mask = mask.any(axis=1)
        while row_mask.any():
            new = np.random.normal(0, 1, size=(row_mask.sum(), 2))
            data[row_mask] = new
            mask = (data < lower) | (data > upper)
            row_mask = mask.any(axis=1)
        #return MinMaxScaler().fit_transform(data)
        return data
def generate_exponential(n_node, scale=1.0):
    """
    Generate data from an exponential distribution.
    """
    data = np.random.exponential(scale, size=(n_node, 2))
    #return MinMaxScaler().fit_transform(data)
    return data
def generate_student_t(n_node, df=3):
    """
    Generate data from a Student's t-distribution with df degrees of freedom.
    """
    data = np.random.standard_t(df, size=(n_node, 2))
    #return MinMaxScaler().fit_transform(data)
    return data
def generate_gaussian_mixture(n_node,m):
    """
    Generate a mixture of Gaussians.
    - centers: list of 2-dim means
    - covariances: list of 2x2 covariance matrices
    - weights: mixing proportions (defaults to uniform)
    """
    # number of clusters
    m = 3
    # randomly generate cluster centers in [0, 1]^2
    centers = np.random.uniform(0, 1, size=(m, 2))
    # use small isotropic covariance for each cluster
    covariances = [np.eye(2) * 0.05 for _ in range(m)]
    # uniform mixing weights
    weights = np.ones(m) / m
    m = len(centers)
    if weights is None:
        weights = np.ones(m) / m
    labels = np.random.choice(m, size=n_node, p=weights)
    data = np.zeros((n_node, 2))
    for i in range(m):
        idx = labels == i
        data[idx] = np.random.multivariate_normal(centers[i], covariances[i], size=idx.sum())
    return data, labels
def generate_uniform(n_node, low=0.0, high=1.0):
    """
    Generate data uniformly in [low, high]^2.
    """
    return np.random.uniform(low, high, size=(n_node, 2))
def generate_lognormal(n_node, mean=0.0, sigma=1.0):
    """
    Generate data from a log-normal distribution.
    """
    return np.random.lognormal(mean, sigma, size=(n_node, 2))
def generate_gamma(n_node, shape=2.0, scale=1.0):
    """
    Generate data from a Gamma distribution.
    """
    return np.random.gamma(shape, scale, size=(n_node, 2))
def generate_laplace(n_node, loc=0.0, scale=1.0):
    """
    Generate data from a Laplace (double exponential) distribution.
    """
    return np.random.laplace(loc, scale, size=(n_node, 2))
def generate_cauchy(n_node, loc=0.0, scale=1.0):
    """
    Generate data from a Cauchy distribution.
    """
    return np.random.standard_cauchy(size=(n_node, 2)) * scale + loc
def generate_ring(n_node, radius=1.0, noise=0.05):
    """
    Generate points roughly on a circle of given radius.
    """
    theta = np.random.uniform(0, 2*np.pi, size=n_node)
    r = np.random.normal(radius, noise, size=n_node)
    return np.vstack([r*np.cos(theta), r*np.sin(theta)]).T

def genrate_moss_and_circles(n_node):
    X_moons, y_moons = make_moons(n_samples=n_node, noise=0.1)
    X_circles, y_circles = make_circles(n_samples=n_node, noise=0.05, factor=0.5)
    X = np.vstack([X_moons, X_circles])
    y = np.hstack([y_moons, y_circles + 2])
    return X,y
def generate_beta(n_node, a=2.0, b=5.0):
    """
    Generate data from a Beta distribution (each维度独立).
    """
    return np.column_stack([
        np.random.beta(a, b, size=n_node),
        np.random.beta(a, b, size=n_node),
    ])