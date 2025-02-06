import pytest 
import numpy as np

from cluster import (KMeans, make_clusters)

def test_recognition_of_number_cluster():
    """Test if KMeans correctly recognizes the number of clusters."""
    n = 5
    clusters, _ = make_clusters(k=n, scale=1)

    k = KMeans(k=n)
    k.fit(clusters)
    centroids = k.get_centroids()

    assert len(centroids) == n, "KMeans don't recognize number of centroids"


def test_zero_clusters():
    """Test if KMeans raises an error when k=0."""
    with pytest.raises(ValueError, match=r"Number of clusters \(k\) must be greater than 0."):
        k = KMeans(k=0)









