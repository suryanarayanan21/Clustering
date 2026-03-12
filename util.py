import pandas as pd
from scipy.special import softmax

distribution = pd.read_csv("./distribution2.csv", index_col=0)
tokens = distribution["Token"]

distribution = distribution.drop(columns=["Token", "Cluster"])

classes = distribution.columns

probs = softmax(distribution.to_numpy(), axis = 1)
probs_df = pd.DataFrame(probs, index=tokens, columns=classes)
probs_df['Cluster'] = probs_df.idxmax(axis=1)
probs_df.to_csv("./distribution_softmax.csv")

distribution['Cluster'] = distribution.idxmax(axis = 1)
distribution['Token'] = distribution.index

output = pd.pivot_table(distribution, values='Token', index='Cluster', aggfunc=", ".join)

output.to_csv("./clusters2.csv")