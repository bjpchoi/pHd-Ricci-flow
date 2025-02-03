"""
Note: needs cleaning.

"""


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


train_csv_path = "/content/drive/MyDrive/num_train.csv"
test_csv_path = "/content/drive/MyDrive/num_test.csv"


def load_and_preprocess_modified(csv_path):
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


def load_and_preprocess_unmodified(csv_path):
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
       flattened = power_spectrum.flatten()
       processed_data.append(flattened)


   return np.array(processed_data)


def moving_average(values, window=5):
   values = np.array(values)
   if window < 2:
       return values
   cumsum = np.cumsum(np.insert(values, 0, 0))
   smoothed = (cumsum[window:] - cumsum[:-window]) / float(window)
   pad = np.full(window-1, smoothed[0])
   return np.concatenate([pad, smoothed])


print("Loading and preprocessing training data (modified)...")
X_train_modified = load_and_preprocess_modified(train_csv_path)
print(f"Modified training data shape: {X_train_modified.shape}")


print("Loading and preprocessing test data (modified)...")
X_test_modified = load_and_preprocess_modified(test_csv_path)
print(f"Modified test data shape: {X_test_modified.shape}")


print("Loading and preprocessing training data (unmodified)...")
X_train_unmodified = load_and_preprocess_unmodified(train_csv_path)
print(f"Unmodified training data shape: {X_train_unmodified.shape}")


print("Loading and preprocessing test data (unmodified)...")
X_test_unmodified = load_and_preprocess_unmodified(test_csv_path)
print(f"Unmodified test data shape: {X_test_unmodified.shape}")


y_train = np.array([1]*120 + [0]*120)
y_test = np.array([1]*30 + [0]*30)


assert X_train_modified.shape[0] == 240, "Expected 240 training samples for modified data"
assert X_test_modified.shape[0] == 60, "Expected 60 test samples for modified data"
assert X_train_unmodified.shape[0] == 240, "Expected 240 training samples for unmodified data"
assert X_test_unmodified.shape[0] == 60, "Expected 60 test samples for unmodified data"


scaler_modified = StandardScaler()
X_train_modified_scaled = scaler_modified.fit_transform(X_train_modified)
X_test_modified_scaled = scaler_modified.transform(X_test_modified)


scaler_unmodified = StandardScaler()
X_train_unmodified_scaled = scaler_unmodified.fit_transform(X_train_unmodified)
X_test_unmodified_scaled = scaler_unmodified.transform(X_test_unmodified)


max_components = 50
n_components_list = range(1, max_components + 1)


test_losses_modified = []
test_losses_unmodified = []


pca_mod_full = PCA(n_components=max_components)
X_train_mod_pca_full = pca_mod_full.fit_transform(X_train_modified_scaled)
X_test_mod_pca_full = pca_mod_full.transform(X_test_modified_scaled)


pca_unmod_full = PCA(n_components=max_components)
X_train_unmod_pca_full = pca_unmod_full.fit_transform(X_train_unmodified_scaled)
X_test_unmod_pca_full = pca_unmod_full.transform(X_test_unmodified_scaled)


for n in n_components_list:
   X_train_mod_pca = X_train_mod_pca_full[:, :n]
   X_test_mod_pca = X_test_mod_pca_full[:, :n]
   model_mod = LogisticRegression(max_iter=1000, solver='lbfgs')
   model_mod.fit(X_train_mod_pca, y_train)
   y_test_mod_pred_proba = model_mod.predict_proba(X_test_mod_pca)
   test_loss_mod = log_loss(y_test, y_test_mod_pred_proba)
   test_losses_modified.append(test_loss_mod)


   X_train_unmod_pca = X_train_unmod_pca_full[:, :n]
   X_test_unmod_pca = X_test_unmod_pca_full[:, :n]
   model_unmod = LogisticRegression(max_iter=1000, solver='lbfgs')
   model_unmod.fit(X_train_unmod_pca, y_train)
   y_test_unmod_pred_proba = model_unmod.predict_proba(X_test_unmod_pca)
   test_loss_unmod = log_loss(y_test, y_test_unmod_pred_proba)
   test_losses_unmodified.append(test_loss_unmod)


window_size = 5
test_losses_modified_smooth = moving_average(test_losses_modified, window=window_size)
test_losses_unmodified_smooth = moving_average(test_losses_unmodified, window=window_size)


plt.figure(figsize=(14, 8))
plt.plot(n_components_list, test_losses_modified, marker='o', linestyle=':', color='blue', alpha=0.3, label='Modified (Raw)')
plt.plot(n_components_list, test_losses_unmodified, marker='s', linestyle=':', color='red', alpha=0.3, label='Unmodified (Raw)')
plt.plot(n_components_list, test_losses_modified_smooth, marker='o', linestyle='-', color='blue', label='Modified (Smoothed)', linewidth=2)
plt.plot(n_components_list, test_losses_unmodified_smooth, marker='s', linestyle='-', color='red', label='Unmodified (Smoothed)', linewidth=2)
plt.title("Log Loss vs Number of Principal Components", fontsize=20, fontweight='bold')
plt.xlabel("Number of Principal Components", fontsize=16)
plt.ylabel("Log Loss", fontsize=16)
plt.xticks(np.arange(0, max_components + 1, 5), fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend(fontsize=14)
plt.tight_layout()
min_mod_idx = np.argmin(test_losses_modified)
min_unmod_idx = np.argmin(test_losses_unmodified)
plt.scatter(n_components_list[min_mod_idx], test_losses_modified[min_mod_idx], color='blue', s=100, edgecolors='k', zorder=5)
plt.scatter(n_components_list[min_unmod_idx], test_losses_unmodified[min_unmod_idx], color='red', s=100, edgecolors='k', zorder=5)
plt.show()


import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import os


fs = 200.0
max_freq = 50.0
num_bins = 50


train_data_path = "/content/drive/MyDrive/xtrain.npy"
test_data_path = "/content/X_test.npy"
y_train_path = "/content/y_train (1).npy"
y_test_path = "/content/y_test (1).npy"


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


frequency_bands = {
   "delta": (1, 3),
   "theta": (4, 7),
   "alpha": (8, 13),
   "beta": (14, 30),
   "gamma": (31, 50)
}


pairs = [
   ("FP1", "FP2"), ("F7", "F8"), ("F3", "F4"), ("FT7", "FT8"), ("FC3", "FC4"), ("T7", "T8"), ("P7", "P8"), ("C3", "C4"), ("TP7", "TP8"),
   ("CP3", "CP4"), ("P3", "P4"), ("O1", "O2"), ("AF3", "AF4"), ("F5", "F6"), ("F7", "F8"), ("FC5", "FC6"), ("FC1", "FC2"), ("C5", "C6"),
   ("C1", "C2"), ("CP5", "CP6"), ("CP1", "CP2"), ("P5", "P6"), ("P1", "P2"), ("PO7", "PO8"), ("PO5", "PO6"), ("PO3", "PO4"), ("CB1", "CB2")
]
pairs = list(dict.fromkeys(pairs))


def bandpass_filter(data, low, high, fs=500, order=4):
   nyquist = 0.5 * fs
   low_cutoff = low / nyquist
   high_cutoff = high / nyquist
   b, a = butter(order, [low_cutoff, high_cutoff], btype="band")
   return filtfilt(b, a, data)


def differential_entropy(signal):
   variance = np.var(signal)
   return 0.5 * np.log(2 * np.pi * np.e * variance) if variance > 0 else 0


def modify_channels(power_spectrum, channel_names, avg_groups, diff_groups):
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


def preprocess_data_binned(X, channel_names, freqs, f_max_idx, bin_edges, do_channel_modification=True):
   n_samples, n_channels, n_time = X.shape
   X_processed = []
   final_channel_names = None
   for i in range(n_samples):
       if (i+1) % 1000 == 0 or (i+1) == n_samples:
           print(f"Processing sample {i+1}/{n_samples}")
       fft_data = np.fft.rfft(X[i], axis=1)
       power_spectrum = np.abs(fft_data)**2
       power_spectrum = power_spectrum[:, :f_max_idx+1]
       if do_channel_modification:
           mod_power_spectrum, mod_channel_names = modify_channels(power_spectrum, channel_names, avg_groups, diff_groups)
       else:
           mod_power_spectrum = power_spectrum.copy()
           mod_channel_names = channel_names
       if final_channel_names is None:
           final_channel_names = mod_channel_names
       binned_power = bin_power_spectrum(mod_power_spectrum, freqs[:f_max_idx+1], bin_edges)
       flattened = binned_power.flatten()
       X_processed.append(flattened)
   return np.array(X_processed), final_channel_names


def compute_rasm_features(X):
   num_samples, num_channels, num_time = X.shape
   ch_to_idx = {ch: i for i, ch in enumerate(channel_names)}
   X_features = []
   for i in range(num_samples):
       if (i+1) % 1000 == 0 or (i+1) == num_samples:
           print(f"Computing RASM for sample {i+1}/{num_samples}")
       channel_band_de = {}
       for ch in channel_names:
           ch_idx = ch_to_idx[ch]
           data_ch = X[i, ch_idx, :]
           de_values = []
           for (low, high) in frequency_bands.values():
               filtered_signal = bandpass_filter(data_ch, low, high, fs=fs, order=4)
               de_values.append(differential_entropy(filtered_signal))
           channel_band_de[ch] = np.array(de_values)
       sample_features = []
       for (left_ch, right_ch) in pairs:
           left_de = channel_band_de[left_ch]
           right_de = channel_band_de[right_ch]
           with np.errstate(divide='ignore', invalid='ignore'):
               rasm = np.where(right_de != 0, left_de / right_de, 0.0)
           sample_features.extend(rasm)
       X_features.append(sample_features)
   return np.array(X_features)


def compute_dasm_rasm_de_features(X):
   num_samples, num_channels, num_time = X.shape
   ch_to_idx = {ch: i for i, ch in enumerate(channel_names)}
   n_bands = len(frequency_bands)
   X_features = []
   for i in range(num_samples):
       if (i+1) % 1000 == 0 or (i+1) == num_samples:
           print(f"Computing DE+DASM+RASM for sample {i+1}/{num_samples}")
       channel_band_de = {}
       for ch in channel_names:
           ch_idx = ch_to_idx[ch]
           data_ch = X[i, ch_idx, :]
           de_values = []
           for (low, high) in frequency_bands.values():
               filtered_signal = bandpass_filter(data_ch, low, high, fs=fs, order=4)
               de_values.append(differential_entropy(filtered_signal))
           channel_band_de[ch] = np.array(de_values)
       all_de_features = []
       for ch in channel_names:
           all_de_features.extend(channel_band_de[ch])
       dasm_rasm_features = []
       for (left_ch, right_ch) in pairs:
           left_de = channel_band_de[left_ch]
           right_de = channel_band_de[right_ch]
           dasm = left_de - right_de
           with np.errstate(divide='ignore', invalid='ignore'):
               rasm = np.where(right_de != 0, left_de / right_de, 0.0)
           dasm_rasm_features.extend(dasm)
           dasm_rasm_features.extend(rasm)
       sample_features = all_de_features + dasm_rasm_features
       X_features.append(sample_features)
   return np.array(X_features)


def moving_average(data, window_size=10):
   data = np.asarray(data, dtype=float)
   result = np.zeros_like(data)
   for i in range(len(data)):
       start_idx = max(0, i - window_size + 1)
       result[i] = np.mean(data[start_idx : i+1])
   return result


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
bin_edges = np.linspace(0, max_freq, num_bins + 1)


print("Preprocessing training data (with modifications)...")
X_train_mod, final_channel_names_mod = preprocess_data_binned(X_train, channel_names, freqs, f_max_idx, bin_edges, do_channel_modification=True)
print("Preprocessing test data (with modifications)...")
X_test_mod, _ = preprocess_data_binned(X_test, channel_names, freqs, f_max_idx, bin_edges, do_channel_modification=True)


print("Preprocessing training data (without modifications)...")
X_train_unmod, final_channel_names_unmod = preprocess_data_binned(X_train, channel_names, freqs, f_max_idx, bin_edges, do_channel_modification=False)
print("Preprocessing test data (without modifications)...")
X_test_unmod, _ = preprocess_data_binned(X_test, channel_names, freqs, f_max_idx, bin_edges, do_channel_modification=False)


print("Shapes after preprocessing:")
print(f"With modifications: X_train_mod: {X_train_mod.shape}, X_test_mod: {X_test_mod.shape}")
print(f"Without modifications: X_train_unmod: {X_train_unmod.shape}, X_test_unmod: {X_test_unmod.shape}")


print("Computing DE+DASM+RASM features...")
X_train_dasm_rasm_de = compute_dasm_rasm_de_features(X_train)
X_test_dasm_rasm_de = compute_dasm_rasm_de_features(X_test)
print("DE+DASM+RASM features computed.")
print(f"X_train_dasm_rasm_de shape: {X_train_dasm_rasm_de.shape}, X_test_dasm_rasm_de shape: {X_test_dasm_rasm_de.shape}")


scaler_mod = StandardScaler()
X_train_mod_scaled = scaler_mod.fit_transform(X_train_mod)
X_test_mod_scaled = scaler_mod.transform(X_test_mod)


scaler_unmod = StandardScaler()
X_train_unmod_scaled = scaler_unmod.fit_transform(X_train_unmod)
X_test_unmod_scaled = scaler_unmod.transform(X_test_unmod)


scaler_dasm_rasm_de = StandardScaler()
X_train_dasm_rasm_de_scaled = scaler_dasm_rasm_de.fit_transform(X_train_dasm_rasm_de)
X_test_dasm_rasm_de_scaled = scaler_dasm_rasm_de.transform(X_test_dasm_rasm_de)


max_components = 50
n_components_list = range(1, max_components + 1)


test_losses_modified = []
test_losses_unmodified = []
test_losses_dasm_rasm_de = []


pca_mod_full = PCA(n_components=max_components)
X_train_mod_pca_full = pca_mod_full.fit_transform(X_train_mod_scaled)
X_test_mod_pca_full = pca_mod_full.transform(X_test_mod_scaled)


pca_unmod_full = PCA(n_components=max_components)
X_train_unmod_pca_full = pca_unmod_full.fit_transform(X_train_unmod_scaled)
X_test_unmod_pca_full = pca_unmod_full.transform(X_test_unmod_scaled)


pca_dasm_rasm_de_full = PCA(n_components=max_components)
X_train_dasm_rasm_de_pca_full = pca_dasm_rasm_de_full.fit_transform(X_train_dasm_rasm_de_scaled)
X_test_dasm_rasm_de_pca_full = pca_dasm_rasm_de_full.transform(X_test_dasm_rasm_de_scaled)


for n in n_components_list:
   X_train_mod_pca = X_train_mod_pca_full[:, :n]
   X_test_mod_pca = X_test_mod_pca_full[:, :n]
   model_mod = LogisticRegression(max_iter=1000, solver='lbfgs')
   model_mod.fit(X_train_mod_pca, y_train)
   y_test_mod_pred_proba = model_mod.predict_proba(X_test_mod_pca)
   test_loss_mod = log_loss(y_test, y_test_mod_pred_proba)
   test_losses_modified.append(test_loss_mod)


   X_train_unmod_pca = X_train_unmod_pca_full[:, :n]
   X_test_unmod_pca = X_test_unmod_pca_full[:, :n]
   model_unmod = LogisticRegression(max_iter=1000, solver='lbfgs')
   model_unmod.fit(X_train_unmod_pca, y_train)
   y_test_unmod_pred_proba = model_unmod.predict_proba(X_test_unmod_pca)
   test_loss_unmod = log_loss(y_test, y_test_unmod_pred_proba)
   test_losses_unmodified.append(test_loss_unmod)


   X_train_dasm_rasm_de_pca = X_train_dasm_rasm_de_pca_full[:, :n]
   X_test_dasm_rasm_de_pca = X_test_dasm_rasm_de_pca_full[:, :n]
   model_dasm_rasm_de = LogisticRegression(max_iter=1000, solver='lbfgs')
   model_dasm_rasm_de.fit(X_train_dasm_rasm_de_pca, y_train)
   y_test_dasm_rasm_de_pred_proba = model_dasm_rasm_de.predict_proba(X_test_dasm_rasm_de_pca)
   test_loss_dasm_rasm_de = log_loss(y_test, y_test_dasm_rasm_de_pred_proba)
   test_losses_dasm_rasm_de.append(test_loss_dasm_rasm_de)


window_size = 10
test_losses_modified_smooth = moving_average(test_losses_modified, window_size)
test_losses_unmodified_smooth = moving_average(test_losses_unmodified, window_size)
test_losses_dasm_rasm_de_smooth = moving_average(test_losses_dasm_rasm_de, window_size)


plt.figure(figsize=(14, 8))
plt.plot(n_components_list, test_losses_modified, marker='o', linestyle=':', color='blue', alpha=0.3, label='Modified (Raw)')
plt.plot(n_components_list, test_losses_unmodified, marker='s', linestyle=':', color='red', alpha=0.3, label='Unmodified (Raw)')
plt.plot(n_components_list, test_losses_dasm_rasm_de, marker='^', linestyle=':', color='green', alpha=0.3, label='DE+DASM+RASM (Raw)')
plt.plot(n_components_list, test_losses_modified_smooth, marker='o', linestyle='-', color='blue', label='Modified (Smoothed)', linewidth=2)
plt.plot(n_components_list, test_losses_unmodified_smooth, marker='s', linestyle='-', color='red', label='Unmodified (Smoothed)', linewidth=2)
plt.plot(n_components_list, test_losses_dasm_rasm_de_smooth, marker='^', linestyle='-', color='green', label='DE+DASM+RASM (Smoothed)', linewidth=2)
plt.title("Log Loss vs Number of Principal Components\n(Modified vs Unmodified vs DE+DASM+RASM)", fontsize=20, fontweight='bold')
plt.xlabel("Number of Principal Components", fontsize=16)
plt.ylabel("Log Loss", fontsize=16)
plt.xticks(np.arange(0, max_components+1, 5), fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend(fontsize=14)
plt.tight_layout()
min_mod_idx = np.argmin(test_losses_modified)
min_unmod_idx = np.argmin(test_losses_unmodified)
min_dasm_rasm_de_idx = np.argmin(test_losses_dasm_rasm_de)
plt.scatter(n_components_list[min_mod_idx], test_losses_modified[min_mod_idx], color='blue', s=100, edgecolors='k', zorder=5)
plt.scatter(n_components_list[min_unmod_idx], test_losses_unmodified[min_unmod_idx], color='red', s=100, edgecolors='k', zorder=5)
plt.scatter(n_components_list[min_dasm_rasm_de_idx], test_losses_dasm_rasm_de[min_dasm_rasm_de_idx], color='green', s=100, edgecolors='k', zorder=5)
plt.show()


print("Done.")


import numpy as np
import matplotlib.pyplot as plt
!pip install mrmr_selection
from mrmr import mrmr_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from scipy.signal import butter, filtfilt
import pandas as pd
from math import log, sqrt


fs = 200.0
max_freq = 50.0
num_bins = 50
K = 50


train_data_path = "/content/drive/MyDrive/xtrain.npy"
test_data_path = "/content/X_test.npy"
y_train_path = "/content/y_train (1).npy"
y_test_path = "/content/y_test (1).npy"


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


def preprocess_data_binned(X, freqs, f_max_idx, bin_edges):
   n_samples, n_channels, n_time = X.shape
   X_processed = []
   for i in range(n_samples):
       if (i+1) % 1000 == 0 or (i+1) == n_samples:
           print(f"Processing sample {i+1}/{n_samples}")
       fft_data = np.fft.rfft(X[i], axis=1)
       power_spectrum = np.abs(fft_data)**2
       power_spectrum = power_spectrum[:, :f_max_idx+1]
       binned_power = bin_power_spectrum(power_spectrum, freqs[:f_max_idx+1], bin_edges)
       flattened = binned_power.flatten()
       X_processed.append(flattened)
   return np.array(X_processed)


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
bin_edges = np.linspace(0, max_freq, num_bins + 1)


print("Preprocessing training data (unmodified)...")
X_train_unmod = preprocess_data_binned(X_train, freqs, f_max_idx, bin_edges)
print("Preprocessing test data (unmodified)...")
X_test_unmod = preprocess_data_binned(X_test, freqs, f_max_idx, bin_edges)
print(f"Unmodified shapes => X_train_unmod: {X_train_unmod.shape}, X_test_unmod: {X_test_unmod.shape}")


scaler = StandardScaler()
X_train_unmod_scaled = scaler.fit_transform(X_train_unmod)
X_test_unmod_scaled = scaler.transform(X_test_unmod)


X_train_df = pd.DataFrame(X_train_unmod_scaled)
selected_features = mrmr_classif(X=X_train_df, y=y_train, K=K)
print(f"\nTop {K} mRMR feature indices:\n{selected_features}")


X_train_mrmr = X_train_unmod_scaled[:, selected_features]
X_test_mrmr = X_test_unmod_scaled[:, selected_features]


lr_model = LogisticRegression(max_iter=1000, solver='lbfgs')
lr_model.fit(X_train_mrmr, y_train)


train_preds = lr_model.predict(X_train_mrmr)
test_preds = lr_model.predict(X_test_mrmr)


train_acc = accuracy_score(y_train, train_preds)
test_acc = accuracy_score(y_test, test_preds)


test_proba = lr_model.predict_proba(X_test_mrmr)
loss = log_loss(y_test, test_proba)


print("\n========== Logistic Regression with mRMR(50) Features ==========")
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy:  {test_acc:.4f}")
print(f"Test Log Loss:  {loss:.4f}")


individual_log_losses = []
for i, true_label in enumerate(y_test):
   p_correct = test_proba[i, true_label]
   p_correct = max(p_correct, 1e-15)
   loss_i = -log(p_correct)
   individual_log_losses.append(loss_i)


individual_log_losses = np.array(individual_log_losses)
mean_log_loss = np.mean(individual_log_losses)
std_log_loss = np.std(individual_log_losses, ddof=1)
sem_log_loss = std_log_loss / sqrt(len(individual_log_losses))
ci_lower = mean_log_loss - 1.96 * sem_log_loss
ci_upper = mean_log_loss + 1.96 * sem_log_loss


print(f"\nMean of Individual Log Losses: {mean_log_loss:.4f}")
print(f"Standard Deviation of Individual Log Losses: {std_log_loss:.4f}")
print(f"Standard Error of the Mean (SEM) of Log Losses: {sem_log_loss:.4f}")
print(f"95% Confidence Interval for Mean Log Loss: [{ci_lower:.4f}, {ci_upper:.4f}]")


