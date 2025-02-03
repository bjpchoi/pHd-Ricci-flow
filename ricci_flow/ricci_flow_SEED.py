"""
!pip install scikit-learn networkx POT GraphRicciCurvature tqdm scipy

"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import ot
import time
import logging
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from GraphRicciCurvature.OllivierRicci import OllivierRicci


# ============================
# CONFIGURATION & PARAMETERS
# ============================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


channel_labels = [
   "FP1","FPZ","FP2","AF3","AF4","F7","F5","F3","F1","FZ","F2","F4","F6","F8","FT7","FC5","FC3","FC1","FCZ","FC2","FC4","FC6","FT8","T7","C5","C3","C1","CZ","C2","C4","C6","T8","TP7","CP5","CP3","CP1","CPZ","CP2","CP4","CP6","TP8","P7","P5","P3","P1","PZ","P2","P4","P6","P8","PO7","PO5","PO3","POZ","PO4","PO6","PO8","CB1","O1","OZ","O2","CB2"
]
N = len(channel_labels)  
iterations = 20       
alpha = 0.75          
method = "OTD"        # Use OTD for Wasserstein-based curvature


# Define cut ratios for modularity analysis
cut_ratios = np.arange(0.0, 0.97+(1/2016), 1/2016)


# ============================
# LOAD 62x62 CORRELATION MATRIX
# ============================
avg_corr_matrix = np.load('average_correlation_matrix.npy')
np.fill_diagonal(avg_corr_matrix, 0.0)


# ============================
# CREATE GRAPH FROM CORRELATION MATRIX
# ============================
dist_matrix = 1 - avg_corr_matrix
np.fill_diagonal(dist_matrix, 0.0)


G_initial = nx.Graph()
for ch in channel_labels:
   G_initial.add_node(ch)


for i in range(N):
   for j in range(i+1, N):
       G_initial.add_edge(channel_labels[i], channel_labels[j], weight=dist_matrix[i, j])


# ============================
# DEFINE RICCI FLOW FUNCTION
# ============================
def run_ricci_flow(G, iterations=10, alpha=0.5, method="OTD"):
   ricci = OllivierRicci(G, alpha=alpha, method=method, base=np.e, exp_power=2, verbose="ERROR")
   ricci.compute_ricci_flow(iterations=iterations)
   return ricci.G


print("Running Ricci Flow on the 62x62 Correlation-Based Graph:")
G_ricci = run_ricci_flow(G_initial, iterations=iterations, alpha=alpha, method=method)


# ============================
# COMMUNITY DETECTION FUNCTIONS
# ============================
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
   all_weights = sorted((G[u][v]['weight'], u, v) for u, v in G.edges())
   num_to_cut = int(len(all_weights) * cut_ratio)
   if num_to_cut == 0:
       return G, []
   to_cut = all_weights[-num_to_cut:]
   cut_edges = []
   for weight, u, v in to_cut:
       G.remove_edge(u, v)
       cut_edges.append((u, v))
   return G, cut_edges


# ============================
# COMPUTE MODULARITY FOR VARIOUS CUT RATIOS
# ============================
modularity_per_ratio = []
print("Computing modularity for various cut ratios...")
for r in tqdm(cut_ratios, desc="Processing cut ratios"):
   G_copy = G_ricci.copy()
   G_cut, cut_edges = cut_graph(G_copy, cut_ratio=r)
   communities = get_communities(G_cut)
   Q = compute_modularity(G_cut, communities)
   modularity_per_ratio.append(Q)


max_modularity_idx = np.argmax(modularity_per_ratio)
optimal_cut_ratio = cut_ratios[max_modularity_idx]
optimal_modularity = modularity_per_ratio[max_modularity_idx]


print(f"\nOptimal Cut Ratio: {optimal_cut_ratio:.2f} ({int(optimal_cut_ratio*100)}% cut) with Modularity: {optimal_modularity:.4f}")


G_optimal = G_ricci.copy()
G_optimal, cut_edges_optimal = cut_graph(G_optimal, cut_ratio=optimal_cut_ratio)
communities_optimal = get_communities(G_optimal)
Q_optimal = compute_modularity(G_optimal, communities_optimal)
num_components_optimal = len(communities_optimal)


print(f"Number of connected components at optimal cut ratio ({int(optimal_cut_ratio*100)}% cut): {num_components_optimal}")
print(f"Communities at optimal cut ratio ({int(optimal_cut_ratio*100)}% cut):")
for idx, community in enumerate(communities_optimal, 1):
   print(f"  Community {idx}: {sorted(community)}")


# ============================
# VISUALIZATIONS
# ============================


# 1. Plot mean correlation matrix (with zeroed diagonal)
plt.figure(figsize=(8,6))
im = plt.imshow(avg_corr_matrix, cmap='seismic', interpolation='nearest', vmin=-1, vmax=1)
plt.colorbar(im)
plt.title("Mean Correlation Matrix Between EEG Channels (Diagonal Zeroed)")
plt.xlabel("EEG Channel")
plt.ylabel("EEG Channel")
plt.xticks(ticks=np.arange(N), labels=channel_labels, rotation=90)
plt.yticks(ticks=np.arange(N), labels=channel_labels)
plt.tight_layout()
plt.show()


# 2. Plot adjacency before Ricci flow
initial_adj = nx.to_numpy_array(G_initial, nodelist=channel_labels, weight='weight')
plt.figure(figsize=(8,6))
im = plt.imshow(initial_adj, cmap='seismic', interpolation='nearest')
plt.colorbar(im)
plt.title("Adjacency Matrix Before Ricci Flow (1 - Mean Correlation)")
plt.xlabel("EEG Channel")
plt.ylabel("EEG Channel")
plt.xticks(ticks=np.arange(N), labels=channel_labels, rotation=90)
plt.yticks(ticks=np.arange(N), labels=channel_labels)
plt.tight_layout()
plt.show()


# 3. Plot adjacency after Ricci flow
ricci_adj = nx.to_numpy_array(G_ricci, nodelist=channel_labels, weight='weight')
plt.figure(figsize=(8,6))
im = plt.imshow(ricci_adj, cmap='jet', interpolation='nearest')
plt.colorbar(im)
plt.title("Adjacency Matrix After Ricci Flow")
plt.xlabel("EEG Channel")
plt.ylabel("EEG Channel")
plt.xticks(ticks=np.arange(N), labels=channel_labels, rotation=90)
plt.yticks(ticks=np.arange(N), labels=channel_labels)
plt.tight_layout()
plt.show()


# 4. Plot final adjacency after optimal cut
optimal_adj_matrix = nx.to_numpy_array(G_optimal, nodelist=channel_labels, weight='weight')
plt.figure(figsize=(8,6))
im = plt.imshow(optimal_adj_matrix, cmap='jet', interpolation='nearest')
plt.colorbar(im)
plt.title(f"Adjacency Matrix After Optimal Cut ({int(optimal_cut_ratio*100)}% Cut)")
plt.xlabel("EEG Channel")
plt.ylabel("EEG Channel")
plt.xticks(ticks=np.arange(N), labels=channel_labels, rotation=90)
plt.yticks(ticks=np.arange(N), labels=channel_labels)
plt.tight_layout()
plt.show()


# 5. Plot modularity vs cut ratio
plt.figure(figsize=(8,6))
plt.plot(np.array(cut_ratios)*100, modularity_per_ratio, marker='o', linestyle='-')
plt.title("Modularity vs Cut Ratio")
plt.xlabel("Cut Ratio (%)")
plt.ylabel("Modularity (Q)")
plt.grid(True)
plt.tight_layout()
plt.show()


df_optimal_adj = pd.DataFrame(optimal_adj_matrix, index=channel_labels, columns=channel_labels)
output_csv_path = f'adjacency_after_optimal_cut_ratio_{int(optimal_cut_ratio*100)}_percent.csv'
df_optimal_adj.to_csv(output_csv_path)
print(f"Optimal adjacency matrix saved to '{output_csv_path}'.")


plt.figure(figsize=(12, 10))
pos_optimal = nx.spring_layout(G_optimal, seed=42)  
edges_optimal = G_optimal.edges(data=True)
weights_optimal = [edge[2]['weight'] for edge in edges_optimal]
nx.draw_networkx_nodes(G_optimal, pos_optimal, node_size=700, node_color='lightblue')
nx.draw_networkx_edges(G_optimal, pos_optimal, width=weights_optimal, edge_color='grey')
nx.draw_networkx_labels(G_optimal, pos_optimal, font_size=10, font_family='sans-serif')
plt.title(f"Graph at Optimal Cut Ratio = {int(optimal_cut_ratio*100)}% with Modularity = {optimal_modularity:.4f}")
plt.axis('off')
plt.tight_layout()
plt.show()
