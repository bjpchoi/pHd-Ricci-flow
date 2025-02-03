"""
!pip install scikit-learn networkx POT GraphRicciCurvature tqdm scipy

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import networkx as nx
import ot
import time
import logging
from tqdm import tqdm
import random
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from scipy.stats import ttest_1samp


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


N = 14
num_samples = 240
expected_num_features = 3584
iterations = 20
alpha = 0.75
method = "OTD"
cut_ratios = np.arange(0.0, 0.97+(1/91), 1/91)
channel_labels = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]


df = pd.read_csv("/content/drive/MyDrive/num_train.csv", header=None)
df = df.iloc[1:, :]
data = df.iloc[:, 1:].values
data = data.astype(float)


if data.shape[1] != expected_num_features:
   raise ValueError(f"Data must have {expected_num_features} features. Found {data.shape[1]}.")


if data.shape[0] != num_samples:
   raise ValueError(f"Expected {num_samples} samples, found {data.shape[0]}.")


scaler = StandardScaler()
data = scaler.fit_transform(data)
data_reshaped = data.reshape(num_samples, N, 256)


correlations_across_samples = np.zeros((N, N, num_samples))
for s in range(num_samples):
   corr = np.corrcoef(data_reshaped[s], rowvar=True)
   correlations_across_samples[:, :, s] = corr


mean_corr = np.mean(correlations_across_samples, axis=2)
np.fill_diagonal(mean_corr, 0.0)


dist_matrix = 1 - mean_corr
np.fill_diagonal(dist_matrix, 0.0)


G_initial = nx.Graph()
for ch in channel_labels:
   G_initial.add_node(ch)


for i in range(N):
   for j in range(i+1, N):
       G_initial.add_edge(channel_labels[i], channel_labels[j], weight=dist_matrix[i, j])


def run_ricci_flow(G, iterations=10, alpha=0.5, method="OTD"):
   ricci = OllivierRicci(G, alpha=alpha, method=method, base=np.e, exp_power=2, verbose="ERROR")
   ricci.compute_ricci_flow(iterations=iterations)
   return ricci.G


print("Running Ricci Flow on Mean-Correlation-Based Graph:")
G_ricci = run_ricci_flow(G_initial, iterations=iterations, alpha=alpha, method=method)


curvature_matrix = np.zeros((N, N))
node_index = {ch: idx for idx, ch in enumerate(channel_labels)}
for u, v, data_attr in G_ricci.edges(data=True):
   curv = data_attr.get('ricciCurvature', 0.0)
   i, j = node_index[u], node_index[v]
   curvature_matrix[i, j] = curv
   curvature_matrix[j, i] = curv


def compute_modularity(G, communities):
   if G.number_of_edges() == 0:
       return 0.0
   G_transformed = G.copy()
   for u, v, data in G_transformed.edges(data=True):
       data['weight'] = 1.0
   return nx.algorithms.community.quality.modularity(G_transformed, communities, weight='weight')


def get_communities(G):
   return list(nx.connected_components(G))


def cut_graph(G, cut_ratio=0.75):
   if G.number_of_edges() == 0:
       return G, []
   all_weights = sorted([(G[u][v]['weight'], u, v) for u, v in G.edges()], key=lambda x: x[0])
   num_to_cut = int(len(all_weights) * cut_ratio)
   if num_to_cut == 0:
       return G, []
   to_cut = all_weights[-num_to_cut:]
   cut_edges = []
   for weight, u, v in to_cut:
       G.remove_edge(u, v)
       cut_edges.append((u, v))
   return G, cut_edges


modularity_per_ratio = []
for r in cut_ratios:
   G_copy = G_ricci.copy()
   G_cut, cut_edges = cut_graph(G_copy, cut_ratio=r)
   communities = get_communities(G_cut)
   Q = compute_modularity(G_cut, communities)
   modularity_per_ratio.append(Q)


cut_ratio_75 = 0.75
G_copy_75 = G_ricci.copy()
G_cut_75, _ = cut_graph(G_copy_75, cut_ratio=cut_ratio_75)
final_adj_75 = nx.to_numpy_array(G_cut_75, nodelist=channel_labels, weight='weight')


communities_75 = get_communities(G_cut_75)
Q_75 = compute_modularity(G_cut_75, communities_75)
num_components_75 = len(communities_75)
print(f"Modularity after {int(cut_ratio_75*100)}% cut: {Q_75:.4f}")
print(f"Number of connected components after {int(cut_ratio_75*100)}% cut: {num_components_75}")


max_modularity = max(modularity_per_ratio)
optimal_indices = [i for i, Q in enumerate(modularity_per_ratio) if Q == max_modularity]
optimal_index = optimal_indices[0]
optimal_cut_ratio = cut_ratios[optimal_index]


print(f"\nOptimal Cut Ratio: {optimal_cut_ratio*100:.2f}% with Modularity: {max_modularity:.4f}")


G_optimal = G_ricci.copy()
G_optimal, _ = cut_graph(G_optimal, cut_ratio=optimal_cut_ratio)
optimal_communities = get_communities(G_optimal)


print(f"Number of connected components after optimal cut: {len(optimal_communities)}")
print("Connected Components (Communities) after Optimal Cut:")
for idx, community in enumerate(optimal_communities, 1):
   sorted_community = sorted(community)
   print(f"Community {idx}: {sorted_community}")


pos_initial = nx.spring_layout(G_initial, seed=42)
pos_ricci = nx.spring_layout(G_ricci, seed=42)
G_after_75 = nx.from_numpy_array(final_adj_75, create_using=nx.Graph())
mapping = {i: ch for i, ch in enumerate(channel_labels)}
G_after_75 = nx.relabel_nodes(G_after_75, mapping)
pos_75 = nx.spring_layout(G_after_75, seed=42)


plt.figure(figsize=(8,6))
im = plt.imshow(mean_corr, cmap='seismic', interpolation='nearest')
plt.colorbar(im)
plt.title("Mean Correlation Matrix Between EEG Channels")
plt.xlabel("EEG Channel")
plt.ylabel("EEG Channel")
plt.xticks(ticks=np.arange(N), labels=channel_labels, rotation=45)
plt.yticks(ticks=np.arange(N), labels=channel_labels)
plt.tight_layout()
plt.show()


plt.figure(figsize=(8,6))
im = plt.imshow(nx.to_numpy_array(G_initial, nodelist=channel_labels, weight='weight'), cmap='seismic', interpolation='nearest')
plt.colorbar(im)
plt.title("Adjacency Matrix Before Ricci Flow (1 - Mean Correlation)")
plt.xlabel("EEG Channel")
plt.ylabel("EEG Channel")
plt.xticks(ticks=np.arange(N), labels=channel_labels, rotation=45)
plt.yticks(ticks=np.arange(N), labels=channel_labels)
plt.tight_layout()
plt.show()


plt.figure(figsize=(8,6))
im = plt.imshow(nx.to_numpy_array(G_ricci, nodelist=channel_labels, weight='weight'), cmap='jet', interpolation='nearest')
plt.colorbar(im)
plt.title("Adjacency Matrix After Ricci Flow")
plt.xlabel("EEG Channel")
plt.ylabel("EEG Channel")
plt.xticks(ticks=np.arange(N), labels=channel_labels, rotation=45)
plt.yticks(ticks=np.arange(N), labels=channel_labels)
plt.tight_layout()
plt.show()


plt.figure(figsize=(8,6))
im = plt.imshow(final_adj_75, cmap='jet', interpolation='nearest')
plt.colorbar(im)
plt.title("Adjacency Matrix After 75% Cut (Jet Colormap)")
plt.xlabel("EEG Channel")
plt.ylabel("EEG Channel")
plt.xticks(ticks=np.arange(N), labels=channel_labels, rotation=45)
plt.yticks(ticks=np.arange(N), labels=channel_labels)
plt.tight_layout()
plt.show()


plt.figure(figsize=(8,6))
plt.plot(np.array(cut_ratios)*100, modularity_per_ratio, marker='o', linestyle='-')
plt.title("Modularity vs Cut Ratio")
plt.xlabel("Cut Ratio (%)")
plt.ylabel("Modularity (Q)")
plt.grid(True)
plt.tight_layout()
plt.show()


final_adj_df = pd.DataFrame(final_adj_75, index=channel_labels, columns=channel_labels)
final_adj_df.to_csv("adjacency_after_75_percent_cut.csv")
print("Final adjacency matrix after 75% cut saved to 'adjacency_after_75_percent_cut.csv'.")

