import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.special import softmax

distribution = pd.read_csv("./distribution.csv")
tokens = distribution["Token"]

distribution = distribution.drop(distribution.columns[0], axis=1)
distribution = distribution.drop(columns=["Token", "Cluster"])

classes = distribution.columns

angles = np.linspace(0, 2 * np.pi, len(classes), endpoint=False) + np.pi / 2
vectors = np.column_stack((np.cos(angles), np.sin(angles)))

probs = softmax(distribution.to_numpy(), axis = 1)
probs_df = pd.DataFrame(probs, index=tokens, columns=classes)

points = distribution.to_numpy().dot(vectors)

fig, ax = plt.subplots(figsize=(20, 20))

polypoints = np.vstack((vectors, vectors[0]))

scaling_factor = 1

ax.plot([x[0] for x in polypoints], [x[1] for x in polypoints], color='black', linestyle='-', linewidth=1.5)

ax.scatter([x[0] for x in vectors], [x[1] for x in vectors], c="black")

for i, label in enumerate(classes):
    ax.annotate(label, (vectors[i][0], vectors[i][1]), textcoords="offset points", xytext=(0,10), ha='center')

ax.scatter([scaling_factor*x[0] for x in points], [scaling_factor*x[1] for x in points], c="blue")

for i, label in enumerate(tokens):
    ax.annotate(label, (scaling_factor*points[i][0], scaling_factor*points[i][1]), textcoords="offset points", xytext=(0,10), ha='center')

fig.savefig("./visual.jpg")