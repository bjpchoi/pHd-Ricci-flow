import numpy as np
import pandas as pd
import os
from math import log, sqrt
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline


fs = 200.0
max_freq = 50.0
num_bins = 50
pca_components = 50


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


def compute_dasm_rasm_de_features(X):
   num_samples, num_channels, num_time = X.shape
   ch_to_idx = {ch: i for i, ch in enumerate(channel_names)}
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
X_train_mod, final_channel_names_mod = preprocess_data_binned(
   X_train, channel_names, freqs, f_max_idx, bin_edges, do_channel_modification=True
)
print("Preprocessing test data (with modifications)...")
X_test_mod, _ = preprocess_data_binned(
   X_test, channel_names, freqs, f_max_idx, bin_edges, do_channel_modification=True
)


print("Preprocessing training data (without modifications)...")
X_train_unmod, final_channel_names_unmod = preprocess_data_binned(
   X_train, channel_names, freqs, f_max_idx, bin_edges, do_channel_modification=False
)
print("Preprocessing test data (without modifications)...")
X_test_unmod, _ = preprocess_data_binned(
   X_test, channel_names, freqs, f_max_idx, bin_edges, do_channel_modification=False
)


print("Computing DE+DASM+RASM features...")
X_train_dasm_rasm_de = compute_dasm_rasm_de_features(X_train)
X_test_dasm_rasm_de = compute_dasm_rasm_de_features(X_test)


print("\nShapes after preprocessing:")
print(f"X_train_mod: {X_train_mod.shape}, X_test_mod: {X_test_mod.shape}")
print(f"X_train_unmod: {X_train_unmod.shape}, X_test_unmod: {X_test_unmod.shape}")
print(f"X_train_dasm_rasm_de: {X_train_dasm_rasm_de.shape}, X_test_dasm_rasm_de: {X_test_dasm_rasm_de.shape}")


scaler_mod = StandardScaler()
X_train_mod_scaled = scaler_mod.fit_transform(X_train_mod)
X_test_mod_scaled = scaler_mod.transform(X_test_mod)


scaler_unmod = StandardScaler()
X_train_unmod_scaled = scaler_unmod.fit_transform(X_train_unmod)
X_test_unmod_scaled = scaler_unmod.transform(X_test_unmod)


scaler_dasm_rasm_de = StandardScaler()
X_train_dasm_rasm_de_scaled = scaler_dasm_rasm_de.fit_transform(X_train_dasm_rasm_de)
X_test_dasm_rasm_de_scaled = scaler_dasm_rasm_de.transform(X_test_dasm_rasm_de)


pca_mod = PCA(n_components=pca_components)
X_train_mod_pca = pca_mod.fit_transform(X_train_mod_scaled)
X_test_mod_pca = pca_mod.transform(X_test_mod_scaled)


pca_unmod = PCA(n_components=pca_components)
X_train_unmod_pca = pca_unmod.fit_transform(X_train_unmod_scaled)
X_test_unmod_pca = pca_unmod.transform(X_test_unmod_scaled)


pca_dasm_rasm_de = PCA(n_components=pca_components)
X_train_dasm_rasm_de_pca = pca_dasm_rasm_de.fit_transform(X_train_dasm_rasm_de_scaled)
X_test_dasm_rasm_de_pca = pca_dasm_rasm_de.transform(X_test_dasm_rasm_de_scaled)


param_grid = {
   'pca__n_components': [10, 30, 50],
   'svm__C': [100, 1000, 5000],
   'svm__gamma': [1e-4, 1e-3],
   'svm__kernel': ['rbf']
}


skf = StratifiedKFold(n_splits=10, shuffle=True)


pipeline_mod = Pipeline([
   ('pca', PCA()),
   ('svm', SVC(probability=True))
])


grid_search_mod = GridSearchCV(
   estimator=pipeline_mod,
   param_grid=param_grid,
   cv=skf,
   scoring='accuracy',
   n_jobs=-1,
   verbose=2
)


grid_search_mod.fit(X_train_mod_scaled, y_train)
print("\n============ GridSearchCV RESULTS: Modified ============")
print("Best Params (Modified):", grid_search_mod.best_params_)
best_model_mod = grid_search_mod.best_estimator_


train_preds_mod = best_model_mod.predict(X_train_mod_scaled)
test_preds_mod = best_model_mod.predict(X_test_mod_scaled)


train_acc_mod = accuracy_score(y_train, train_preds_mod)
test_acc_mod = accuracy_score(y_test, test_preds_mod)
print(f"Modified -> Train Accuracy: {train_acc_mod:.4f}, Test Accuracy: {test_acc_mod:.4f}")


y_proba_mod = best_model_mod.predict_proba(X_test_mod_scaled)
individual_log_losses_mod = []
for i, true_label in enumerate(y_test):
   p_correct = max(y_proba_mod[i, true_label], 1e-15)
   loss_i = -log(p_correct)
   individual_log_losses_mod.append(loss_i)


individual_log_losses_mod = np.array(individual_log_losses_mod)
mean_log_loss_mod = np.mean(individual_log_losses_mod)
std_log_loss_mod = np.std(individual_log_losses_mod, ddof=1)
sem_log_loss_mod = std_log_loss_mod / sqrt(len(individual_log_losses_mod))


print(f"Modified -> Mean Log Loss: {mean_log_loss_mod:.4f}")
print(f"Modified -> Std Dev Log Loss: {std_log_loss_mod:.4f}")
print(f"Modified -> SEM Log Loss: {sem_log_loss_mod:.4f}")


pipeline_unmod = Pipeline([
   ('pca', PCA()),
   ('svm', SVC(probability=True))
])


grid_search_unmod = GridSearchCV(
   estimator=pipeline_unmod,
   param_grid=param_grid,
   cv=skf,
   scoring='accuracy',
   n_jobs=-1,
   verbose=2
)


grid_search_unmod.fit(X_train_unmod_scaled, y_train)
print("\n============ GridSearchCV RESULTS: Unmodified ============")
print("Best Params (Unmodified):", grid_search_unmod.best_params_)
best_model_unmod = grid_search_unmod.best_estimator_


train_preds_unmod = best_model_unmod.predict(X_train_unmod_scaled)
test_preds_unmod = best_model_unmod.predict(X_test_unmod_scaled)


train_acc_unmod = accuracy_score(y_train, train_preds_unmod)
test_acc_unmod = accuracy_score(y_test, test_preds_unmod)
print(f"Unmodified -> Train Accuracy: {train_acc_unmod:.4f}, Test Accuracy: {test_acc_unmod:.4f}")


y_proba_unmod = best_model_unmod.predict_proba(X_test_unmod_scaled)
individual_log_losses_unmod = []
for i, true_label in enumerate(y_test):
   p_correct = max(y_proba_unmod[i, true_label], 1e-15)
   loss_i = -log(p_correct)
   individual_log_losses_unmod.append(loss_i)


individual_log_losses_unmod = np.array(individual_log_losses_unmod)
mean_log_loss_unmod = np.mean(individual_log_losses_unmod)
std_log_loss_unmod = np.std(individual_log_losses_unmod, ddof=1)
sem_log_loss_unmod = std_log_loss_unmod / sqrt(len(individual_log_losses_unmod))


print(f"Unmodified -> Mean Log Loss: {mean_log_loss_unmod:.4f}")
print(f"Unmodified -> Std Dev Log Loss: {std_log_loss_unmod:.4f}")
print(f"Unmodified -> SEM Log Loss: {sem_log_loss_unmod:.4f}")


pipeline_dasm = Pipeline([
   ('pca', PCA()),
   ('svm', SVC(probability=True))
])


grid_search_dasm = GridSearchCV(
   estimator=pipeline_dasm,
   param_grid=param_grid,
   cv=skf,
   scoring='accuracy',
   n_jobs=-1,
   verbose=2
)


grid_search_dasm.fit(X_train_dasm_rasm_de_scaled, y_train)
print("\n============ GridSearchCV RESULTS: DE+DASM+RASM ============")
print("Best Params (DE+DASM+RASM):", grid_search_dasm.best_params_)
best_model_dasm = grid_search_dasm.best_estimator_


train_preds_dasm = best_model_dasm.predict(X_train_dasm_rasm_de_scaled)
test_preds_dasm = best_model_dasm.predict(X_test_dasm_rasm_de_scaled)


train_acc_dasm = accuracy_score(y_train, train_preds_dasm)
test_acc_dasm = accuracy_score(y_test, test_preds_dasm)
print(f"DE+DASM+RASM -> Train Accuracy: {train_acc_dasm:.4f}, Test Accuracy: {test_acc_dasm:.4f}")


y_proba_dasm = best_model_dasm.predict_proba(X_test_dasm_rasm_de_scaled)
individual_log_losses_dasm = []
for i, true_label in enumerate(y_test):
   p_correct = max(y_proba_dasm[i, true_label], 1e-15)
   loss_i = -log(p_correct)
   individual_log_losses_dasm.append(loss_i)


individual_log_losses_dasm = np.array(individual_log_losses_dasm)
mean_log_loss_dasm = np.mean(individual_log_losses_dasm)
std_log_loss_dasm = np.std(individual_log_losses_dasm, ddof=1)
sem_log_loss_dasm = std_log_loss_dasm / sqrt(len(individual_log_losses_dasm))


print(f"DE+DASM+RASM -> Mean Log Loss: {mean_log_loss_dasm:.4f}")
print(f"DE+DASM+RASM -> Std Dev Log Loss: {std_log_loss_dasm:.4f}")
print(f"DE+DASM+RASM -> SEM Log Loss: {sem_log_loss_dasm:.4f}")


print("\nDone.")
