import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_array
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

class RegularizedKMeans(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=3, regularization_strength=0.1, max_iter=300, tol=1e-4, random_state=None):
        """
        Regularized KMeans clustering algorithm with LASSO-style regularization.

        Parameters:
        - n_clusters: int, number of clusters.
        - regularization_strength: float, regularization parameter for feature selection.
        - max_iter: int, maximum number of iterations for k-means.
        - tol: float, convergence tolerance.
        - random_state: int or None, random state for reproducibility.
        """
        self.n_clusters = n_clusters
        self.regularization_strength = regularization_strength
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, y=None):
        """
        Fit the Regularized KMeans model to the data.

        Parameters:
        - X: ndarray, shape (n_samples, n_features), input data.
        - y: Not used, placeholder for compatibility.

        Returns:
        - self: Fitted estimator.
        """
        X = check_array(X)  # Ensure the input data is a valid array
        self.n_samples_, self.n_features_ = X.shape
        self.feature_weights_ = np.ones(self.n_features_)  # Initialize feature weights
        self.cluster_centers_ = np.zeros((self.n_clusters, self.n_features_))  # Initialize cluster centers

        # Normalize the input data
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

        # KMeans iterations with regularization
        for iteration in range(self.max_iter):
            # Weight the features by the current weights
            X_weighted = X * self.feature_weights_

            # Apply standard KMeans clustering
            kmeans = KMeans(
                init='k-means++',
                n_clusters=self.n_clusters,
            ).fit(X_weighted)

            new_cluster_centers = kmeans.cluster_centers_

            # Update feature weights using L1-regularization (LASSO-style)
            distances = np.linalg.norm(new_cluster_centers, axis=0)
            self.feature_weights_ = np.maximum(0, 1 - self.regularization_strength / distances)

            # Convergence check
            if np.allclose(self.cluster_centers_, new_cluster_centers, atol=self.tol):
                break

            self.cluster_centers_ = new_cluster_centers

        self.labels_ = kmeans.labels_
        return self

    def transform(self, X):
        """
        Transform the data to cluster-distance space.

        Parameters:
        - X: ndarray, shape (n_samples, n_features), input data.

        Returns:
        - distances: ndarray, shape (n_samples, n_clusters), distances to cluster centers.
        """
        X = check_array(X)
        X_weighted = X * self.feature_weights_
        return np.linalg.norm(X_weighted[:, np.newaxis] - self.cluster_centers_, axis=2)

    def predict(self, X):
        """
        Predict the closest cluster for each sample.

        Parameters:
        - X: ndarray, shape (n_samples, n_features), input data.

        Returns:
        - labels: ndarray, shape (n_samples,), cluster indices.
        """
        distances = self.transform(X)
        return np.argmin(distances, axis=1)