import pandas as pd

distribution = pd.read_csv("./distribution2.csv", index_col=0)
distribution = distribution.drop(columns=["Token", "Cluster"])
distribution['Cluster'] = distribution.idxmax(axis = 1)
distribution['Token'] = distribution.index

output = pd.pivot_table(distribution, values='Token', index='Cluster', aggfunc=", ".join)

output.to_csv("./clusters2.csv")