import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int = 8, tol: float = 1e-6, max_iter: int = 100, init = "random"):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """

        if k <= 0:
            raise ValueError("Number of clusters (k) must be greater than 0.")
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.centroids = None

    def _initialize_cetroids(self, X):
        # Initializes centroids randomly
        if self.init == "random":
            indices = np.random.choice(X.shape[0], self.k, replace=False)  
            centroids = np.array(X[indices, :])
            return centroids

    # KMeans++ Initialization
        elif self.init == "kmeans++":
            centroids = [X[np.random.choice(X.shape[0]), :]] # First centroid is random

            for _ in range(1, self.k):
                # Compute distances from all points to the closest centroid
                distances = np.min(cdist(X, np.array(centroids)), axis=1)  
                prob = distances / distances.sum()  # Compute selection probability

                # Select next centroid with weighted probability
                next_centroid_idx = np.random.choice(X.shape[0], p=prob)  # Select valid index
                next_centroid = X[next_centroid_idx, :]
                centroids.append(next_centroid)

            return np.array(centroids)  # Ensure final output is a numpy array

        else:
            raise ValueError("Invalid initialization method. Use 'random' or 'kmeans++'.")


    def fit(self, X: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        """Compute k-means clustering on dataset X."""
        if X.shape[0] < self.k:
            raise ValueError("Number of clusters (k) cannot be greater than the number of observations.")
        
        self.centroids = self._initialize_cetroids(X)
        
        for _ in range(self.max_iter):
            distances = cdist(X, np.array(self.centroids))  # Compute distances to centroids
            labels = np.argmin(distances, axis=1)  # Assign closest centroid
            
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.k)])
            
            if np.linalg.norm(np.array(self.centroids) - new_centroids) < self.tol:
                break  # Converged
            
            self.centroids = new_centroids

        return self


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        """Predict the closest cluster each sample in X belongs to."""
        if self.centroids is None:
            raise ValueError("Model must be fitted before predicting.")
        distances = cdist(X, np.array(self.centroids))
        return np.argmin(distances, axis=1)


    def get_error(self, X) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """
        """Compute the total intra-cluster variance (sum of squared distances to centroids)."""
        labels = self.predict(X)
        return np.sum([np.linalg.norm(X[labels == i] - self.centroids[i]) ** 2 for i in range(self.k)])

    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        return self.centroids if self.centroids is not None else None
    

