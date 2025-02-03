import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd
from matplotlib.ticker import MaxNLocator
import os
from scipy.stats import kstest
from matplotlib.colors import ListedColormap


sns.set(style='whitegrid')


data_path = '/content/final_optimized_data.npy'
y_path = '/content/y_train.npy'


if not os.path.exists(data_path):
   raise FileNotFoundError(f"The file {data_path} does not exist.")
if not os.path.exists(y_path):
   raise FileNotFoundError(f"The file {y_path} does not exist.")


data = np.load(data_path)
y = np.load(y_path)


print(f"Data shape: {data.shape}")
print(f"Label shape: {y.shape}")


n_samples = data.shape[0]
num_rows = data.shape[1]
d = data.shape[2]


channel_labels = [
   "FP1","FPZ","FP2","AF3","AF4","F7","F5","F3","F1","FZ","F2","F4","F6","F8",
   "FT7","FC5","FC3","FC1","FCZ","FC2","FC4","FC6","FT8","T7","C5","C3","C1","CZ",
   "C2","C4","C6","T8","TP7","CP5","CP3","CP1","CPZ","CP2","CP4","CP6","TP8","P7",
   "P5","P3","P1","PZ","P2","P4","P6","P8","PO7","PO5","PO3","POZ","PO4","PO6","PO8",
   "CB1","O1","OZ","O2","CB2"
]


assert len(channel_labels) == num_rows, "Number of channel labels must match num_rows (62)."


phd_results = []
principal_directions_matrix = np.zeros((num_rows, d))


for row_idx in range(num_rows):
   X = data[:, row_idx, :]
   Sx = (X.T @ X) / n_samples
   y_weighted_X = y[:, np.newaxis] * X
   yxx = (X.T @ y_weighted_X) / n_samples


   try:
       eigenvals, eigenvecs = eigh(yxx, Sx)
   except np.linalg.LinAlgError as e:
       print(f"Eigen decomposition failed for row {row_idx}: {e}")
       continue


   idx = np.argsort(np.abs(eigenvals))[::-1]
   eigenvals = eigenvals[idx]
   eigenvecs = eigenvecs[:, idx]


   principal_direction = eigenvecs[:, 0]
   phd_results.append({
       'row_index': row_idx,
       'eigenvalues': eigenvals,
       'eigenvectors': eigenvecs,
       'principal_direction': principal_direction
   })
   principal_directions_matrix[row_idx, :] = principal_direction


print("PHD computation completed.\n")


feature_indices = np.arange(d)
weights = 1/(feature_indices+1)


plt.figure(figsize=(10,6))
plt.plot(feature_indices, weights, marker='o')
plt.title("Exponential Weighting Function")
plt.xlabel("Feature Index")
plt.ylabel("Weight")
plt.grid(True)
plt.tight_layout()
plt.show()


weighted_principal_directions = principal_directions_matrix * weights


correlation_matrix = np.corrcoef(weighted_principal_directions)
np.fill_diagonal(correlation_matrix, 0)


plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm',
           xticklabels=channel_labels, yticklabels=channel_labels,
           vmin=-1, vmax=1, cbar=True)
plt.title("Correlation Matrix of Weighted Principal Hessian Directions")
plt.xlabel("Channels")
plt.ylabel("Channels")
plt.tight_layout()
plt.show()


def compute_modularity(G, communities):
   if G.number_of_edges() == 0:
       return 0.0
   return nx.algorithms.community.quality.modularity(G, communities, weight=None)


def get_communities(G):
   return list(nx.connected_components(G))


def build_graph(correlation_matrix, threshold=0.0):
   G = nx.Graph()
   G.add_nodes_from(channel_labels)
   for i in range(num_rows):
       for j in range(i+1, num_rows):
           corr = correlation_matrix[i, j]
           if corr >= threshold:
               G.add_edge(channel_labels[i], channel_labels[j], weight=corr)
   return G


thresholds = np.arange(0.00, 1.00, 0.001)
modularity_per_threshold = []


for t in thresholds:
   G_thres = build_graph(correlation_matrix, threshold=t)
   communities = get_communities(G_thres)
   Q = compute_modularity(G_thres, communities)
   modularity_per_threshold.append(Q)
   print(f"Threshold: {t:.3f}, Modularity: {Q:.4f}")


max_modularity_idx = np.argmax(modularity_per_threshold)
optimal_threshold = thresholds[max_modularity_idx]
optimal_modularity = modularity_per_threshold[max_modularity_idx]


print(f"\nOptimal Threshold: {optimal_threshold:.2f} with Modularity: {optimal_modularity:.4f}")


G_optimal = build_graph(correlation_matrix, threshold=optimal_threshold)
communities_optimal = get_communities(G_optimal)
Q_optimal = compute_modularity(G_optimal, communities_optimal)
num_components_optimal = len(communities_optimal)


print(f"Number of connected components at optimal threshold ({optimal_threshold:.2f}): {num_components_optimal}")
print(f"Communities at optimal threshold ({optimal_threshold:.2f}):")
for idx, community in enumerate(communities_optimal, 1):
   print(f"  Community {idx}: {sorted(community)}")


plt.figure(figsize=(12, 10))
pos = nx.spring_layout(G_optimal, seed=42)
edges = G_optimal.edges(data=True)
weights_edge = [edge[2]['weight'] for edge in edges]
nx.draw_networkx_nodes(G_optimal, pos, node_size=700, node_color='lightblue')
nx.draw_networkx_edges(G_optimal, pos, width=weights_edge, edge_color='grey')
nx.draw_networkx_labels(G_optimal, pos, font_size=12, font_family='sans-serif')
plt.title(f"Graph at Optimal Threshold = {optimal_threshold:.2f} with Modularity = {optimal_modularity:.4f}")
plt.axis('off')
plt.tight_layout()
plt.show()


optimal_adj_matrix = nx.to_numpy_array(G_optimal, nodelist=channel_labels, weight='weight')
df_optimal_adj = pd.DataFrame(optimal_adj_matrix, index=channel_labels, columns=channel_labels)
output_csv_path = f'optimal_adj_matrix_threshold_{optimal_threshold:.2f}.csv'
df_optimal_adj.to_csv(output_csv_path)
print(f"Optimal adjacency matrix saved to {output_csv_path}")


plt.figure(figsize=(10, 6))
plt.plot(thresholds, modularity_per_threshold, marker='o', linestyle='-')
plt.title("Modularity vs. Correlation Threshold (No Negative Edges Allowed)")
plt.xlabel("Correlation Threshold")
plt.ylabel("Modularity")
plt.xlim(0, 1)
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=False))
plt.grid(True)
plt.tight_layout()
plt.show()


threshold_75 = 0.29
G_75 = build_graph(correlation_matrix, threshold=threshold_75)
communities_75 = get_communities(G_75)
Q_75 = compute_modularity(G_75, communities_75)
num_components_75 = len(communities_75)
print(f"\nModularity after threshold={threshold_75:.2f}: {Q_75:.4f}")
print(f"Number of connected components after threshold={threshold_75:.2f}: {num_components_75}")


print("Computing KS normality tests...")


ks_matrix = np.zeros((num_rows, d))
for r in range(num_rows):
   for f in range(d):
       x = data[r, :, f]
       _, p = kstest(x, 'norm')
       ks_matrix[r, f] = p


significant_mask = ks_matrix < 0.05
cmap = ListedColormap(["green", "red"])
color_data = significant_mask.astype(int)


plt.figure(figsize=(20, 10))
sns.heatmap(color_data, cmap=cmap, cbar=False,
           xticklabels=False, yticklabels=False)
plt.title("KS Normality Scores (Red < 0.05, Green ≥ 0.05)")
plt.xlabel("Features")
plt.ylabel("Channels")
plt.tight_layout()
plt.show()


print("All computations and plots completed successfully.")

