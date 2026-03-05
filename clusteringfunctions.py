import numpy as np
import pandas as pd
from model import ClusteringFunction
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import normalize

class Dbscan(ClusteringFunction):
    def __init__(self):
        super().__init__("DBSCAN")
    
    def run(self, tokens: list[str], embeddings: np.ndarray, distances: np.ndarray) -> pd.DataFrame:
        model = DBSCAN(eps=0.45, min_samples=2, metric="precomputed")
        pred = model.fit_predict(distances)
        df = pd.DataFrame({"Token": tokens, "Cluster": pred})
        return pd.pivot_table(df, values="Token", index="Cluster", aggfunc=", ".join)


class Agglomerative(ClusteringFunction):
    def __init__(self):
        super().__init__("Agglomerative")
    
    def run(self, tokens:list[str], embeddings: np.ndarray, distances: np.ndarray) -> pd.DataFrame:
        model = AgglomerativeClustering(
            n_clusters=9,
            metric="precomputed",
            linkage="average"
        )
        pred = model.fit_predict(distances)
        df = pd.DataFrame({"Token": tokens, "Cluster": pred})
        return pd.pivot_table(df, values="Token", index="Cluster", aggfunc=", ".join)