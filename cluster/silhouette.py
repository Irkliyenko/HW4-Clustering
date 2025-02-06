import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score as sklearn_silhouette



class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """
        pass
    
    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """

        if len(set(y)) < 2:
            raise ValueError("Silhouette score requires at least 2 clusters.")
        
        # Compute distances between all points
        distances = cdist(X, X)
        silhouette_values = []

        for i in range(X.shape[0]):
            same_cluster = y == y[i]

            a = np.mean(distances[i, same_cluster])  # Intra-cluster distance
            b = np.min([np.mean(distances[i, y == j]) for j in set(y) if j != y[i]])  # Nearest-cluster distance

            silhouette_values.append((b - a) / max(a, b))

        return np.array(silhouette_values) 

    def compare_with_sklearn(self, X, labels):
        """Compare custom silhouette score with sklearn's implementation."""
        custom_score = np.mean(self.score(X, labels))
        sklearn_score = sklearn_silhouette(X, labels)
        return custom_score, sklearn_score

