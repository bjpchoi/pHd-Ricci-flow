import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from matplotlib.ticker import MaxNLocator
import os
import pandas as pd


sns.set(style='whitegrid')


channel_labels = ["AF3", "F7", "F3", "FC5", "T7", "P7",
                 "O1", "O2", "P8", "T8", "FC6", "F4",
                 "F8", "AF4"]


data_path = '/content/drive/MyDrive/train_n_final.npy'
if not os.path.exists(data_path):
   raise FileNotFoundError(f"The file {data_path} does not exist in the current directory.")


data = np.load(data_path)
print(f"Data shape: {data.shape}")


n_samples = data.shape[0]
half = n_samples // 2
y = np.concatenate([np.ones(half), -1*np.ones(half)])


num_rows = data.shape[1]
d = data.shape[2]


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
   channel_name = channel_labels[row_idx]
   print(f"Channel {channel_name} principal direction (first 5): {principal_direction[:5]}")
   print(f"Top 5 eigenvalues by abs: {eigenvals[:5]}\n")


print("PHD computation completed.\n")


correlation_matrix = np.corrcoef(principal_directions_matrix)
np.fill_diagonal(correlation_matrix, 0)


plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm',
           xticklabels=channel_labels, yticklabels=channel_labels,
           vmin=-1, vmax=1)
plt.title("Correlation Matrix of Primary Principal Hessian Directions")
plt.xlabel("Channels")
plt.ylabel("Channels")
plt.tight_layout()
plt.show()


def compute_modularity(G, communities):
   if G.number_of_edges() == 0:
       return 0.0
   return nx.algorithms.community.quality.modularity(G, communities)


def get_communities(G):
   return list(nx.connected_components(G))


def build_graph(correlation_matrix, threshold=0.0):
   G = nx.Graph()
   G.add_nodes_from(channel_labels)
   num_rows = len(channel_labels)
   for i in range(num_rows):
       for j in range(i+1, num_rows):
           corr = correlation_matrix[i, j]
           if corr >= threshold:
               G.add_edge(channel_labels[i], channel_labels[j], weight=corr)
   return G


thresholds = np.arange(0.00, 1.00, 0.01)
modularity_per_threshold = []


for t in thresholds:
   G_thres = build_graph(correlation_matrix, threshold=t)
   communities = get_communities(G_thres)
   Q = compute_modularity(G_thres, communities)
   modularity_per_threshold.append(Q)
   print(f"Threshold: {t:.2f}, Modularity: {Q:.4f}")


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
weights = [edge[2]['weight'] for edge in edges]
nx.draw_networkx_nodes(G_optimal, pos, node_size=700, node_color='lightblue')
nx.draw_networkx_edges(G_optimal, pos, width=weights, edge_color='grey')
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
plt.xticks([0.0, 1.0], ['0', '1'])
plt.xlim(0, 1)
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=False))
plt.grid(True)
plt.tight_layout()
plt.show()


threshold_f = 0.76
correlation_matrix_f = correlation_matrix.copy()
correlation_matrix_f[correlation_matrix_f < threshold_f] = 0
np.fill_diagonal(correlation_matrix_f, 0)
mask = correlation_matrix_f == 0


plt.figure(figsize=(12, 10))
cmap_f = sns.color_palette("YlGnBu", as_cmap=True)
sns.heatmap(correlation_matrix_f,
           annot=True,
           fmt=".2f",
           cmap=cmap_f,
           xticklabels=channel_labels,
           yticklabels=channel_labels,
           vmin=0.76,
           vmax=1.0,
           mask=mask,
           cbar_kws={"shrink": .8},
           linewidths=.5,
           linecolor='gray')
plt.title(f"Heatmap of Correlation Matrix (Threshold ≥ {threshold_f})", fontsize=16)
plt.xlabel("Channels", fontsize=14)
plt.ylabel("Channels", fontsize=14)
plt.tight_layout()
plt.show()


