import numpy as np
from model import PreprocessFunction
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder as CE
from sklearn.metrics.pairwise import cosine_distances

class SentenceEmbeddings(PreprocessFunction):
    def __init__(self):
        super().__init__("Sentence Embeddings")
    
    def run(self, sentences: list[str]) -> tuple[np.ndarray, np.ndarray]:
        model = SentenceTransformer("all-mpnet-base-v2")
        embeddings = model.encode(sentences, convert_to_numpy=True)
        distances = cosine_distances(embeddings)
        return (embeddings, distances)
    
class CrossEncoder(PreprocessFunction):
    def __init__(self):
        super().__init__("CrossEncoder")
    
    def run(self, sentences: list[str]) -> tuple[np.ndarray, np.ndarray]:
        model = CE("cross-encoder/stsb-distilroberta-base")
        model2 = SentenceTransformer("all-mpnet-base-v2")
        embeddings = model2.encode(sentences, convert_to_numpy=True)
        pairs = [[s1, s2] for s1 in sentences for s2 in sentences]
        pairwise_distances = model.predict(pairs)
        distances = np.reshape(pairwise_distances, (len(sentences), len(sentences)))
        return (embeddings, distances)