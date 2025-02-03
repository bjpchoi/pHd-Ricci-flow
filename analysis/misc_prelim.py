"""
Miscellaneous scripts; unformatted and mostly unused. 

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings


warnings.filterwarnings('ignore')


train_csv_path = "/content/drive/MyDrive/num_train.csv"
test_csv_path = "/content/drive/MyDrive/num_test.csv"


df_train = pd.read_csv(train_csv_path, header=None)
df_train = df_train.iloc[1:, :]
train_data = df_train.iloc[:, 1:].values.astype(float)


df_test = pd.read_csv(test_csv_path, header=None)
df_test = df_test.iloc[1:, :]
test_data = df_test.iloc[:, 1:].values.astype(float)


data = np.vstack((train_data, test_data))


if data.shape[1] != 3584:
   raise ValueError(f"Data must have 3584 features, but got {data.shape[1]}")


labels = np.array([1] * 120 + [0] * 120 + [1] * 30 + [0] * 30)


X_train, X_test, y_train, y_test = train_test_split(
   data, labels, test_size=0.2, random_state=0, stratify=labels
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = LogisticRegression(max_iter=1000, random_state=0)
model.fit(X_train, y_train)


y_test_pred = model.predict(X_test)
y_test_proba = model.predict_proba(X_test)


per_sample_accuracy = (y_test_pred == y_test).astype(int)


epsilon = 1e-15
y_test_proba_clipped = np.clip(y_test_proba, epsilon, 1 - epsilon)
per_sample_log_loss = -(
   y_test * np.log(y_test_proba_clipped[:, 1]) +
   (1 - y_test) * np.log(y_test_proba_clipped[:, 0])
)


mean_accuracy = per_sample_accuracy.mean()
mean_log_loss = per_sample_log_loss.mean()


n_samples = len(y_test)
se_log_loss = per_sample_log_loss.std(ddof=1) / np.sqrt(n_samples)


z_score = 1


log_loss_ci_lower = mean_log_loss - z_score * se_log_loss
log_loss_ci_upper = mean_log_loss + z_score * se_log_loss


print(f"Test Accuracy: {mean_accuracy * 100:.2f}%")
print("Test Classification Report:")
print(classification_report(y_test, y_test_pred, digits=4))
print(f"Test Log Loss: {mean_log_loss:.4f}")
print(f"Log Loss SEM: {log_loss_ci_lower:.4f} - {log_loss_ci_upper:.4f}")


coefficients = model.coef_[0]
importance = np.abs(coefficients)
ranked_indices = np.argsort(importance)[::-1]
top_k = 10


print("\nTop 10 Most Important Features:")
for i in range(top_k):
   print(f"Feature {ranked_indices[i]}: Coefficient = {coefficients[ranked_indices[i]]:.4f}")


plt.figure(figsize=(10, 6))
plt.bar(range(top_k), coefficients[ranked_indices[:top_k]], color='skyblue', edgecolor='black')
plt.xticks(range(top_k), [f"Feature {ranked_indices[i]}" for i in range(top_k)], rotation=45)
plt.title("Top 10 Most Important Features in Logistic Regression (Temporal Domain)")
plt.xlabel("Feature Index")
plt.ylabel("Coefficient Value")
plt.tight_layout()
plt.show()


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


train_csv_path = "/content/drive/MyDrive/num_train.csv"
test_csv_path = "/content/drive/MyDrive/num_test.csv"


def load_and_preprocess(csv_path):
   df = pd.read_csv(csv_path, header=None)
   df = df.iloc[1:, 1:]
   data = df.values.astype(float)
   num_samples = data.shape[0]
   processed_data = []
   for idx in range(num_samples):
       matrix = data[idx].reshape(14, 256)
       fft_matrix = np.fft.rfft(matrix, axis=1)
       power_spectrum = np.abs(fft_matrix) ** 2
       power_spectrum = power_spectrum[:, :128]
       modified_matrix = power_spectrum.copy()
       modified_matrix[2] = modified_matrix[2] - modified_matrix[13]
       modified_matrix[1] = modified_matrix[1] - modified_matrix[3]
       modified_matrix[5] = (modified_matrix[5] + modified_matrix[6]) / 2
       modified_matrix[7] = (modified_matrix[7] + modified_matrix[8]) / 2
       rows_to_keep = [0, 1, 2, 4, 5, 7, 9, 10, 11, 13]
       modified_matrix = modified_matrix[rows_to_keep, :]
       flattened = modified_matrix.flatten()
       processed_data.append(flattened)
   return np.array(processed_data)


print("Loading and preprocessing training data...")
train_data = load_and_preprocess(train_csv_path)
print(f"Processed training data shape: {train_data.shape}")


print("Loading and preprocessing test data...")
test_data = load_and_preprocess(test_csv_path)
print(f"Processed test data shape: {test_data.shape}")


y_train = np.array([1] * 120 + [0] * 120)
y_test = np.array([1] * 30 + [0] * 30)


if train_data.shape[0] != 240:
   raise ValueError(f"Expected 240 training samples, but got {train_data.shape[0]}")
if test_data.shape[0] != 60:
   raise ValueError(f"Expected 60 test samples, but got {test_data.shape[0]}")


scaler = StandardScaler()
print("Normalizing training data...")
X_train = scaler.fit_transform(train_data)
print("Normalizing test data...")
X_test = scaler.transform(test_data)


model = LogisticRegression(max_iter=1000)
print("Training Logistic Regression model...")
model.fit(X_train, y_train)


print("Predicting on test data...")
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
print("\nTest Classification Report:")
print(classification_report(y_test, y_test_pred, digits=4))


coefficients = model.coef_[0]
importance = np.abs(coefficients)
ranked_indices = np.argsort(importance)[::-1]
top_k = 10


print("\nTop 10 Most Important Features:")
for i in range(top_k):
   feature_idx = ranked_indices[i]
   print(f"Feature {feature_idx}: Coefficient = {coefficients[feature_idx]:.4f}")


plt.figure(figsize=(12, 6))
plt.bar(range(top_k), coefficients[ranked_indices[:top_k]], color='skyblue')
plt.xticks(range(top_k), [f"Feature {ranked_indices[i]}" for i in range(top_k)], rotation=45)
plt.title("Top 10 Most Important Features in Logistic Regression")
plt.xlabel("Feature Index")
plt.ylabel("Coefficient Value")
plt.tight_layout()
plt.show()


plt.figure(figsize=(6, 6))
plt.bar(['Test Accuracy'], [test_accuracy * 100], color='green')
plt.ylim(0, 100)
plt.ylabel('Accuracy (%)')
plt.title('Logistic Regression Test Accuracy')
for index, value in enumerate([test_accuracy * 100]):
   plt.text(index, value + 1, f"{value:.2f}%", ha='center')
plt.show()


y_test_proba = model.predict_proba(X_test)


epsilon = 1e-15
y_test_proba_clipped = np.clip(y_test_proba, epsilon, 1 - epsilon)


per_sample_log_loss = -(
   y_test * np.log(y_test_proba_clipped[:, 1]) +
   (1 - y_test) * np.log(y_test_proba_clipped[:, 0])
)


mean_log_loss = per_sample_log_loss.mean()


n_samples = len(y_test)
se_log_loss = per_sample_log_loss.std(ddof=1) / np.sqrt(n_samples)


z_score = 1
log_loss_ci_lower = mean_log_loss - z_score * se_log_loss
log_loss_ci_upper = mean_log_loss + z_score * se_log_loss


print("\n--- Log Loss Metrics ---")
print(f"Test Log Loss: {mean_log_loss:.4f}")
print(f"Standard Error (SE): {se_log_loss:.4f}")
print(f"Log Loss CI ({z_score}σ): [{log_loss_ci_lower:.4f}, {log_loss_ci_upper:.4f}]")


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from math import log, sqrt


fs = 500.0
max_freq = 75.0
num_bins = 75


train_data_path = "/content/drive/MyDrive/xtrain.npy"
test_data_path = "/content/X_test.npy"
y_train_path = "/content/y_train (1).npy"
y_test_path = "/content/y_test (1).npy"


print("Loading data...")
X_train = np.load(train_data_path)
X_test = np.load(test_data_path)
y_train = np.load(y_train_path)
y_test = np.load(y_test_path)


print("Data loaded.")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")


X_combined = np.concatenate((X_train, X_test), axis=0)
y_combined = np.concatenate((y_train, y_test), axis=0)
print(f"X_combined shape: {X_combined.shape}, y_combined shape: {y_combined.shape}")


X_full_train, X_full_test, y_full_train, y_full_test = train_test_split(
   X_combined, y_combined, test_size=0.1, shuffle=True, random_state=42
)


print(f"X_full_train shape: {X_full_train.shape}, y_full_train: {y_full_train.shape}")
print(f"X_full_test shape: {X_full_test.shape}, y_full_test: {y_full_test.shape}")


print("Flattening the time-series data...")
n_train_samples = X_full_train.shape[0]
n_test_samples  = X_full_test.shape[0]
n_channels = X_full_train.shape[1]
n_timepoints = X_full_train.shape[2]


X_train_flat = X_full_train.reshape(n_train_samples, n_channels * n_timepoints)
X_test_flat  = X_full_test.reshape(n_test_samples,  n_channels * n_timepoints)


print(f"X_train_flat shape: {X_train_flat.shape}")
print(f"X_test_flat shape: {X_test_flat.shape}")


print("Normalizing the data using StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_flat)
X_test_scaled  = scaler.transform(X_test_flat)
print("Data normalization complete.")


print("Training Logistic Regression model...")
clf = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='auto', n_jobs=-1, verbose=1)
clf.fit(X_train_scaled, y_full_train)
print("Model training complete.")


print("Evaluating the model on test data...")
y_pred = clf.predict(X_test_scaled)


acc = accuracy_score(y_full_test, y_pred)
report = classification_report(y_full_test, y_pred, digits=4)


print(f"\nTest Accuracy: {acc * 100:.2f}%")
print("Classification Report:")
print(report)


plt.figure(figsize=(6, 6))
plt.bar(['Test Accuracy'], [acc * 100], color='green')
plt.ylim(0, 100)
plt.ylabel('Accuracy (%)')
plt.title('Logistic Regression Test Accuracy')
plt.text(0, acc * 100 + 1, f"{acc * 100:.2f}%", ha='center', va='bottom')
plt.show()


print("\n### Computing Standard Error of the Mean for Individual Log Losses ###")


y_test_proba = clf.predict_proba(X_test_scaled)


individual_log_losses = []
for i in range(len(y_full_test)):
   true_label = y_full_test[i]
   p_correct = y_test_proba[i, true_label]
   p_correct = max(p_correct, 1e-15)
   loss_i = -log(p_correct)
   individual_log_losses.append(loss_i)


individual_log_losses = np.array(individual_log_losses)


mean_log_loss = np.mean(individual_log_losses)
std_log_loss = np.std(individual_log_losses, ddof=1)
sem_log_loss = std_log_loss / sqrt(len(individual_log_losses))


print(f"Mean of Individual Log Losses: {mean_log_loss:.4f}")
print(f"Standard Deviation of Individual Log Losses: {std_log_loss:.4f}")
print(f"Standard Error of the Mean (SEM) of Log Losses: {sem_log_loss:.4f}")


print("\nDone.")


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from math import log, sqrt


fs = 500.0
max_freq = 75.0
num_bins = 75


train_data_path = "/content/drive/MyDrive/xtrain.npy"
test_data_path = "/content/X_test.npy"
y_train_path = "/content/y_train (1).npy"
y_test_path = "/content/y_test (1).npy"


preprocessed_train_path = "/content/drive/MyDrive/X_train_preprocessed.npy"
preprocessed_test_path = "/content/drive/MyDrive/X_test_preprocessed.npy"


channel_names = [
   "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ",
   "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2",
   "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4",
   "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6",
   "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7",
   "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "CB1", "O1", "OZ", "O2", "CB2"
]


avg_groups = [
   ["AF3", "F1"],
   ["F5", "FC1"],
   ["C3", "F8", "FZ"],
   ["OZ", "P2", "P4", "PO6"],
   ["CP2", "FT7"],
   ["C6", "CB2", "P3", "PZ"],
   ["P7", "PO8"]
]


diff_groups = [
   ["CPZ", "O2"]
]


print("Loading data...")
X_train = np.load(train_data_path)
X_test = np.load(test_data_path)
y_train = np.load(y_train_path)
y_test = np.load(y_test_path)


print("Data loaded.")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")


N = X_train.shape[-1]
freqs = np.fft.rfftfreq(N, d=1.0/fs)


f_max_idx = np.where(freqs <= max_freq)[0][-1]


print(f"Maximum frequency index for {max_freq} Hz: {f_max_idx}")
print(f"Frequency resolution: {freqs[1]-freqs[0]:.4f} Hz")
print(f"Bin edges: {np.linspace(0, max_freq, num_bins + 1)}")


def modify_channels(power_spectrum, channel_names):
   name_to_idx = {name: i for i, name in enumerate(channel_names)}
   channels_to_remove = set()
   new_channels_data = []
   new_channels_names = []
   for group in avg_groups:
       idxs = [name_to_idx[ch] for ch in group]
       avg_data = np.mean(power_spectrum[idxs, :], axis=0)
       new_channels_data.append(avg_data)
       new_channels_names.append("_".join(group) + "_avg")
       for ch in group:
           channels_to_remove.add(name_to_idx[ch])
   for group in diff_groups:
       if len(group) != 2:
           raise ValueError("Each group in diff_groups must contain exactly two channels for differencing.")
       idxs = [name_to_idx[ch] for ch in group]
       diff_data = power_spectrum[idxs[0], :] - power_spectrum[idxs[1], :]
       new_channels_data.append(diff_data)
       new_channels_names.append("_".join(group) + "_diff")
       for ch in group:
           channels_to_remove.add(name_to_idx[ch])
   original_indices = np.arange(power_spectrum.shape[0])
   keep_indices = [i for i in original_indices if i not in channels_to_remove]
   if new_channels_data:
       final_data = np.concatenate([power_spectrum[keep_indices, :]] + [np.array(new_channels_data)], axis=0)
   else:
       final_data = power_spectrum[keep_indices, :]
   final_names = [channel_names[i] for i in keep_indices] + new_channels_names
   return final_data, final_names


def bin_power_spectrum(power_spectrum, freqs, bin_edges):
   n_channels, n_freqs = power_spectrum.shape
   binned_power = np.zeros((n_channels, num_bins))
   bin_indices = []
   for i in range(num_bins):
       if i == num_bins - 1:
           idx = np.where((freqs >= bin_edges[i]) & (freqs <= bin_edges[i+1]))[0]
       else:
           idx = np.where((freqs >= bin_edges[i]) & (freqs < bin_edges[i+1]))[0]
       bin_indices.append(idx)
   for b in range(num_bins):
       idx = bin_indices[b]
       if len(idx) == 0:
           binned_power[:, b] = 0
       else:
           binned_power[:, b] = power_spectrum[:, idx].mean(axis=1)
   return binned_power


def preprocess_data_binned(X, channel_names, freqs, f_max_idx, bin_edges):
   n_samples, n_channels, n_time = X.shape
   X_processed = []
   final_channel_names = None
   for i in range(n_samples):
       if (i+1) % 1000 == 0 or (i+1) == n_samples:
           print(f"Processing sample {i+1}/{n_samples}")
       fft_data = np.fft.rfft(X[i], axis=1)
       power_spectrum = np.abs(fft_data)**2
       power_spectrum = power_spectrum[:, :f_max_idx+1]
       mod_power_spectrum, mod_channel_names = modify_channels(power_spectrum, channel_names)
       if final_channel_names is None:
           final_channel_names = mod_channel_names
       binned_power = bin_power_spectrum(mod_power_spectrum, freqs[:f_max_idx+1], bin_edges)
       flattened = binned_power.flatten()
       X_processed.append(flattened)
   return np.array(X_processed), final_channel_names


print("Preprocessing training data with channel modification and binning...")
X_train_processed, final_channel_names = preprocess_data_binned(X_train, channel_names, freqs, f_max_idx, np.linspace(0, max_freq, num_bins + 1))
print("Preprocessing test data with channel modification and binning...")
X_test_processed, _ = preprocess_data_binned(X_test, channel_names, freqs, f_max_idx, np.linspace(0, max_freq, num_bins + 1))


print("Preprocessed shapes:")
print(f"X_train_processed: {X_train_processed.shape}")
print(f"X_test_processed: {X_test_processed.shape}")


print("Saving preprocessed data for future use...")
np.save(preprocessed_train_path, X_train_processed)
np.save(preprocessed_test_path, X_test_processed)
print(f"Preprocessed data saved as:\n{preprocessed_train_path}\n{preprocessed_test_path}")


scaler = StandardScaler()
print("Fitting scaler on training data and transforming...")
X_train_scaled = scaler.fit_transform(X_train_processed)
print("Transforming test data...")
X_test_scaled = scaler.transform(X_test_processed)


print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='auto')
model.fit(X_train_scaled, y_train)


print("Predicting on test data...")
y_test_pred = model.predict(X_test_scaled)


test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, digits=4))


coefficients = model.coef_[0]
importance = np.abs(coefficients)
ranked_indices = np.argsort(importance)[::-1]
top_k = 10


print("\nTop 10 Most Important Features:")
for i in range(top_k):
   idx = ranked_indices[i]
   print(f"Feature {idx}: Coefficient = {coefficients[idx]:.4f}")


plt.figure(figsize=(12, 6))
plt.bar(range(top_k), coefficients[ranked_indices[:top_k]], color='skyblue')
plt.xticks(range(top_k), [f"F{ranked_indices[i]}" for i in range(top_k)], rotation=45)
plt.title("Top 10 Most Important Features in Logistic Regression")
plt.xlabel("Feature Index")
plt.ylabel("Coefficient Value")
plt.tight_layout()
plt.show()


plt.figure(figsize=(6, 6))
plt.bar(['Test Accuracy'], [test_accuracy * 100], color='green')
plt.ylim(0, 100)
plt.ylabel('Accuracy (%)')
plt.title('Logistic Regression Test Accuracy')
plt.text(0, test_accuracy * 100 + 1, f"{test_accuracy * 100:.2f}%", ha='center')
plt.show()


print("\n### Computing Standard Error of the Mean for Individual Log Losses ###")


y_test_proba = model.predict_proba(X_test_scaled)


individual_log_losses = []
for i in range(len(y_test)):
   true_label = y_test[i]
   p_correct = y_test_proba[i, true_label]
   loss_i = -log(p_correct)
   individual_log_losses.append(loss_i)


individual_log_losses = np.array(individual_log_losses)
mean_log_loss = np.mean(individual_log_losses)
std_log_loss = np.std(individual_log_losses, ddof=1)
sem_log_loss = std_log_loss / sqrt(len(individual_log_losses))


print(f"Mean of Individual Log Losses: {mean_log_loss:.4f}")
print(f"Standard Deviation of Individual Log Losses: {std_log_loss:.4f}")
print(f"Standard Error of the Mean (SEM) of Log Losses: {sem_log_loss:.4f}")


"""
GCN training routine -- not included in manuscript.

"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import random


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if torch.cuda.is_available():
   torch.cuda.manual_seed(RANDOM_SEED)


num_train_samples = 240
num_test_samples = 60
N = 14
temporal_feature_dim = 256
spectral_feature_dim = 128
total_temporal_features = N * temporal_feature_dim


train_temporal_path = "/content/drive/MyDrive/num_train.csv"
test_temporal_path = "/content/drive/MyDrive/num_test.csv"


train_numerical_path = "/content/drive/MyDrive/numerical_train.csv"
test_numerical_path = "/content/drive/MyDrive/numerical_test.csv"


temporal_graph_path = "/content/temporal graph.csv"
frequency_graph_path = "/content/frequency graph.csv"


train_transformed_path = "train_transformed.npy"
test_transformed_path = "test_transformed.npy"


model_path = "/content/trained_model.pth"
prediction_csv_path = "/content/test_predictions.csv"


dropout_rate = 0.4
learning_rate = 0.01
num_epochs = 30
batch_size = 20
weight_decay = 5e-4


df_train = pd.read_csv(train_temporal_path, header=None)
df_train = df_train.iloc[1:, :]
train_data = df_train.iloc[:, 1:].values
train_data = train_data.astype(float)


if train_data.shape[0] != num_train_samples:
   raise ValueError(f"Expected {num_train_samples} training samples, got {train_data.shape[0]}")
if train_data.shape[1] != total_temporal_features:
   raise ValueError(f"Training data must have {total_temporal_features} features, got {train_data.shape[1]}")


scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
train_temporal_matrices = train_data.reshape(num_train_samples, N, temporal_feature_dim)


df_test = pd.read_csv(test_temporal_path, header=None)
df_test = df_test.iloc[1:, :]
test_data = df_test.iloc[:, 1:].values
test_data = test_data.astype(float)


if test_data.shape[0] != num_test_samples:
   raise ValueError(f"Expected {num_test_samples} test samples, got {test_data.shape[0]}")
if test_data.shape[1] != total_temporal_features:
   raise ValueError(f"Test data must have {total_temporal_features} features, got {test_data.shape[1]}")


test_data = scaler.transform(test_data)
test_temporal_matrices = test_data.reshape(num_test_samples, N, temporal_feature_dim)


pca_per_channel = []
n_components = 128


for ch in range(N):
   X_ch_train = train_temporal_matrices[:, ch, :]
   pca_ch = PCA(n_components=n_components, random_state=RANDOM_SEED)
   pca_ch.fit(X_ch_train)
   pca_per_channel.append(pca_ch)


train_temporal_128 = np.zeros((num_train_samples, N, n_components))
for ch in range(N):
   X_ch_train = train_temporal_matrices[:, ch, :]
   train_temporal_128[:, ch, :] = pca_per_channel[ch].transform(X_ch_train)


test_temporal_128 = np.zeros((num_test_samples, N, n_components))
for ch in range(N):
   X_ch_test = test_temporal_matrices[:, ch, :]
   test_temporal_128[:, ch, :] = pca_per_channel[ch].transform(X_ch_test)


def row_fft_power_spectrum(row_256):
   fft_vals = np.fft.rfft(row_256, n=256)
   power = np.abs(fft_vals)**2
   return power[:128]


train_file = train_numerical_path
test_file = test_numerical_path


train_data_num = np.loadtxt(train_file, delimiter=',', skiprows=1, encoding='utf-8-sig')
test_data_num = np.loadtxt(test_file, delimiter=',', skiprows=1, encoding='utf-8-sig')


train_labels_num = train_data_num[:, 0]
train_features_num = train_data_num[:, 1:]
test_labels_num = test_data_num[:, 0]
test_features_num = test_data_num[:, 1:]


train_mean = np.mean(train_features_num)
train_std = np.std(train_features_num)


test_mean = np.mean(test_features_num)
test_std = np.std(test_features_num)


if train_std == 0:
   train_features_norm = train_features_num
else:
   train_features_norm = (train_features_num - train_mean) / train_std


if test_std == 0:
   test_features_norm = test_features_num
else:
   test_features_norm = (test_features_num - test_mean) / test_std


train_matrices_num = train_features_norm.reshape(train_features_norm.shape[0], 14, 256)
test_matrices_num = test_features_norm.reshape(test_features_norm.shape[0], 14, 256)


train_transformed = np.zeros((train_matrices_num.shape[0], 14, 128))
for i in range(train_matrices_num.shape[0]):
   for j in range(14):
       train_transformed[i, j, :] = row_fft_power_spectrum(train_matrices_num[i, j, :])


test_transformed = np.zeros((test_matrices_num.shape[0], 14, 128))
for i in range(test_matrices_num.shape[0]):
   for j in range(14):
       test_transformed[i, j, :] = row_fft_power_spectrum(test_matrices_num[i, j, :])


np.save('train_labels.npy', train_labels_num)
np.save(train_transformed_path, train_transformed)
np.save('test_labels.npy', test_labels_num)
np.save(test_transformed_path, test_transformed)


train_spectral_128 = np.load(train_transformed_path)
test_spectral_128 = np.load(test_transformed_path)


if train_spectral_128.shape[0] != train_temporal_128.shape[0]:
   raise ValueError("Mismatch in training samples count between temporal and spectral.")
if test_spectral_128.shape[0] != test_temporal_128.shape[0]:
   raise ValueError("Mismatch in test samples count between temporal and spectral.")


temporal_graph_df = pd.read_csv(temporal_graph_path, index_col=0)
frequency_graph_df = pd.read_csv(frequency_graph_path, index_col=0)


if temporal_graph_df.shape[0] != temporal_graph_df.shape[1]:
   raise ValueError("Temporal graph is not square.")
if frequency_graph_df.shape[0] != frequency_graph_df.shape[1]:
   raise ValueError("Frequency graph is not square.")


temporal_graph_df = temporal_graph_df.astype(float)
frequency_graph_df = frequency_graph_df.astype(float)


temporal_adj = temporal_graph_df.values
frequency_adj = frequency_graph_df.values


num_nodes = N
temporal_edges_list = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if temporal_adj[i, j] != 0]
frequency_edges_list = [(num_nodes + i, num_nodes + j) for i in range(num_nodes) for j in range(num_nodes) if frequency_adj[i, j] != 0]
multiplex_edges_list = [(i, num_nodes + i) for i in range(num_nodes)]


all_static_edges = temporal_edges_list + frequency_edges_list + multiplex_edges_list
edge_index = torch.tensor(all_static_edges, dtype=torch.long).t().contiguous()


train_data_list = []
for i in range(train_temporal_128.shape[0]):
   node_features = np.concatenate([train_temporal_128[i], train_spectral_128[i]], axis=0)
   x = torch.tensor(node_features, dtype=torch.float)
   label = 1 if i < num_train_samples // 2 else 0
   y = torch.tensor([label], dtype=torch.long)
   data = Data(x=x, edge_index=edge_index, y=y)
   train_data_list.append(data)


train_loader = DataLoader(train_data_list, batch_size=batch_size, shuffle=True)


class GCN(nn.Module):
   def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
       super(GCN, self).__init__()
       self.conv1 = GCNConv(input_dim, hidden_dim)
       self.conv2 = GCNConv(hidden_dim, hidden_dim)
       self.fc = nn.Linear(hidden_dim, output_dim)
       self.dropout = nn.Dropout(p=dropout)
  
   def forward(self, x, edge_index, batch):
       x = self.conv1(x, edge_index)
       x = F.relu(x)
       x = self.dropout(x)
       x = self.conv2(x, edge_index)
       x = F.relu(x)
       x = self.dropout(x)
       x = global_mean_pool(x, batch)
       x = self.fc(x)
       return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(input_dim=128, hidden_dim=64, output_dim=2, dropout=dropout_rate).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def train_model(model, loader, optimizer):
   model.train()
   total_loss = 0
   for data in loader:
       data = data.to(device)
       optimizer.zero_grad()
       out = model(data.x, data.edge_index, data.batch)
       loss = F.nll_loss(out, data.y.view(-1))
       loss.backward()
       optimizer.step()
       total_loss += loss.item() * data.num_graphs
   return total_loss / len(loader.dataset)


def evaluate_model(model, loader):
   model.eval()
   correct = 0
   total = 0
   with torch.no_grad():
       for data in loader:
           data = data.to(device)
           out = model(data.x, data.edge_index, data.batch)
           pred = out.argmax(dim=1)
           correct += (pred == data.y.view(-1)).sum().item()
           total += data.num_graphs
   return correct / total


print("Starting Training...")
for epoch in range(num_epochs):
   loss_val = train_model(model, train_loader, optimizer)
   acc_val = evaluate_model(model, train_loader)
   print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss_val:.4f}, Accuracy: {acc_val:.4f}")


torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")


test_data_list = []
for i in range(test_temporal_128.shape[0]):
   node_features_test = np.concatenate([test_temporal_128[i], test_spectral_128[i]], axis=0)
   x_test = torch.tensor(node_features_test, dtype=torch.float)
   data_test = Data(x=x_test, edge_index=edge_index)
   test_data_list.append(data_test)


test_loader = DataLoader(test_data_list, batch_size=batch_size, shuffle=False)


test_model = GCN(input_dim=128, hidden_dim=64, output_dim=2, dropout=dropout_rate).to(device)
test_model.load_state_dict(torch.load(model_path))
test_model.eval()


def predict(model, loader):
   model.eval()
   all_predictions = []
   with torch.no_grad():
       for data in loader:
           data = data.to(device)
           out = model(data.x, data.edge_index, data.batch)
           pred = out.argmax(dim=1).cpu().numpy()
           all_predictions.extend(pred)
   return all_predictions


test_predictions = predict(test_model, test_loader)


graph_indices = np.arange(1, test_temporal_128.shape[0] + 1)
prediction_df = pd.DataFrame({
   'Graph_Index': graph_indices,
   'Predicted_Label': test_predictions
})


prediction_df.to_csv(prediction_csv_path, index=False)
print(f"Predictions saved to '{prediction_csv_path}'.")


predictions_df = pd.read_csv(prediction_csv_path)


expected_columns = ['Graph_Index', 'Predicted_Label']
if not all(column in predictions_df.columns for column in expected_columns):
   raise ValueError(f"CSV must contain the following columns: {expected_columns}")


predictions_df = predictions_df.sort_values(by='Graph_Index').reset_index(drop=True)


total_test_samples = predictions_df.shape[0]


if total_test_samples < 60:
   raise ValueError(f"Expected at least 60 test samples, but found {total_test_samples}")


true_labels = [1]*30 + [0]*30


predicted_labels = predictions_df['Predicted_Label'].iloc[:60].values


true_labels = pd.Series(true_labels, name='True_Label')


comparison_df = pd.DataFrame({
   'Graph_Index': predictions_df['Graph_Index'].iloc[:60],
   'Predicted_Label': predicted_labels,
   'True_Label': true_labels
})


accuracy = accuracy_score(comparison_df['True_Label'], comparison_df['Predicted_Label'])
print(f"Test Accuracy: {accuracy * 100:.2f}%")


print("\nConfusion Matrix:")
cm = confusion_matrix(comparison_df['True_Label'], comparison_df['Predicted_Label'])
print(cm)


print("\nClassification Report:")
report = classification_report(comparison_df['True_Label'], comparison_df['Predicted_Label'], digits=4)
print(report)


comparison_df.to_csv("/content/comparison_results.csv", index=False)
print("Comparison results saved to '/content/comparison_results.csv'.")


