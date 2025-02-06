import pytest
import numpy as np
from cluster import (KMeans, 
                     Silhouette,
                     make_clusters)

def test_silhouette_score_test():

    clusters, _ = make_clusters(k=3, scale=1)
    k = KMeans(k=3)
    s = Silhouette()
    k.fit(clusters)
    pred = k.predict(clusters)
    score = s.compare_with_sklearn(clusters, pred)

    assert np.isclose(score[0], score[1], atol=1e-2), "Custom silhouette score is different from Sklearn"
