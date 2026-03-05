import numpy as np
from model import PreprocessFunction
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances

class SentenceEmbeddings(PreprocessFunction):
    def __init__(self):
        super().__init__("Sentence Embeddings")
    
    def run(self, sentences: list[str]) -> tuple[np.ndarray, np.ndarray]:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(sentences, convert_to_numpy=True)
        distances = cosine_distances(embeddings)
        return (embeddings, distances)