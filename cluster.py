'''
Modular tool that takes a set of inputs, pre-processing functions, and clustering functions,
and returns an output file with clusters for each combination of configurations
'''
import pandas as pd
from model import ClusteringFunction, PreprocessFunction, InputFunction
from clusteringfunctions import Dbscan, Agglomerative
from inputfunctions import GeminiDescriptions, ChatGPTDescriptions, getTokens
from preprocessfunctions import SentenceEmbeddings, CrossEncoder

clustering_functions: list[ClusteringFunction] = [Dbscan(), Agglomerative()]
preprocess_functions: list[PreprocessFunction] = [SentenceEmbeddings(), CrossEncoder()]
input_functions: list[InputFunction] = [GeminiDescriptions(), ChatGPTDescriptions()]

tokens = getTokens()
inputs = [(x.name, x.run()) for x in input_functions]
print("Obtained Inputs")
embeddings = [(i[0], p.name, p.run(i[1])) for i in inputs for p in preprocess_functions]
print("Created Embeddings")
outputs = [(e[0], e[1], c.name, c.run(tokens, embeddings=e[2][0], distances=e[2][1])) for e in embeddings for c in clustering_functions]
print("Performed clustering")

for o in outputs:
    o[3].to_csv(f"./outputs/{o[0]}_{o[1]}_{o[2]}.csv")

print("Wrote outputs")