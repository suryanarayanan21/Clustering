import numpy as np
import csv
import pandas as pd
from model import PreprocessFunction
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder as CE
from sklearn.metrics.pairwise import cosine_distances
from inputfunctions import getTokens

tokens = getTokens()

with open("./TokenDataset.csv", newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    strings = [f"{row['Tokens']}: {row['Description (Gemini)']}" for row in reader]

classes = [
    'Fabrication',
    'Physical and Embedded computing',
    'Visual computing and Digital Design',
    'Data and computing',
    'Business Management',
    'Entrepreneurship',
    'Design'
]

model = SentenceTransformer("all-mpnet-base-v2")
s = model.encode(strings, convert_to_numpy = True)
cl = model.encode(classes, convert_to_numpy = True)

# pairs = [[t, c] for t in strings for c in classes]


# flat_distances = model.predict(pairs)

# distances = np.reshape(flat_distances, (len(tokens), 7))

distances = cosine_distances(s, cl)

output = pd.DataFrame(distances, index=tokens, columns=classes)

output['Cluster'] = output.idxmax(axis = 1)
output['Token'] = output.index

output.to_csv("./distribution.csv")

clusters = pd.pivot_table(output, values='Token', index='Cluster', aggfunc=", ".join)

clusters.to_csv("./clusters.csv")