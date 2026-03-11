import numpy as np
import pandas as pd
from model import PreprocessFunction
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder as CE
from sklearn.metrics.pairwise import cosine_distances
from inputfunctions import getTokens

tokens = getTokens()

classes = [
    'Fabrication',
    'Physical and Embedded computing',
    'Visual computing and Digital Design',
    'Data and computing',
    'Business Management',
    'Entrepreneurship',
    'Design'
]

pairs = [[t, c] for t in tokens for c in classes]

model = CE("cross-encoder/stsb-distilroberta-base")

flat_distances = model.predict(pairs)

distances = np.reshape(flat_distances, (len(tokens), 7))

output = pd.DataFrame(distances, index=tokens, columns=classes)

output['Cluster'] = output.idxmax(axis = 1)
output['Token'] = output.index

output.to_csv("./distribution.csv")

clusters = pd.pivot_table(output, values='Token', index='Cluster', aggfunc=", ".join)

clusters.to_csv("./clusters.csv")