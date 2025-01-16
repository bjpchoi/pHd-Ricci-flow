from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt


file_path = '/content/drive/MyDrive/2 1.mat'


data = loadmat(file_path)


metadata_keys = ['__header__', '__version__', '__globals__']
variables = [key for key in data.keys() if key not in metadata_keys]


for var_name in variables:
   var_data = data[var_name]


   print(f"Variable: {var_name}")
   print(f"Type: {type(var_data)}")


   if isinstance(var_data, np.ndarray):
       print(f"Shape: {var_data.shape}")


       if var_data.ndim >= 2:
           print("Preview (first few rows):")
           print(var_data[:5])
       elif var_data.ndim == 1:
           print("Preview (first few elements):")
           print(var_data[:5])
   else:
       print(f"Data: {var_data}")


   if var_name == 'djc_eeg1' and isinstance(var_data, np.ndarray):
       plt.figure(figsize=(10, 6))
       plt.plot(var_data[0, :-1])
       plt.title(f"First Channel of {var_name}")
       plt.xlabel("Time Points")
       plt.ylabel("Amplitude")
       plt.grid()
       plt.show()


   print("\n" + "-" * 50 + "\n")


#########


mat_files_dir = '/content/drive/MyDrive/'


files = [f"{i} {j}.mat" for i in range(1, 16) for j in range(1, 4)]


all_data = []
all_labels = []


positive_suffixes = ['eeg1', 'eeg6', 'eeg9', 'eeg10', 'eeg14']
negative_suffixes = ['eeg3', 'eeg4', 'eeg7', 'eeg12', 'eeg15']


for file_name in tqdm(files, desc="Processing .mat files"):
   file_path = os.path.join(mat_files_dir, file_name)


   data = loadmat(file_path)


   for var_name in data.keys():
       for suffix in positive_suffixes:
           if var_name.endswith(suffix):
               eeg_data = data[var_name]  
               all_data.append(eeg_data)  
               all_labels.append(1)  


       for suffix in negative_suffixes:
           if var_name.endswith(suffix):
               eeg_data = data[var_name]  
               all_data.append(eeg_data)  
               all_labels.append(-1)  


output_file = '/content/aggregated_data_variable_length.npy'
np.save(output_file, {'data': all_data, 'labels': all_labels})


print(f"Aggregated data and labels saved to {output_file}")


######


START_TIMESTAMP = 1001
END_TIMESTAMP = 27000
NUM_CHANNELS = 62
EXPECTED_WIDTH = END_TIMESTAMP - START_TIMESTAMP + 1 


input_file = '/content/aggregated_data_variable_length.npy'
data_dict = np.load(input_file, allow_pickle=True).item()


all_data = data_dict['data']
all_labels = data_dict['labels']


processed_data = []
processed_labels = []


print("Processing arrays to make widths uniform...")
for idx in tqdm(range(len(all_data)), desc="Processing EEG arrays"):
   eeg_array = all_data[idx]
   label = all_labels[idx]


   if eeg_array.shape[0] != NUM_CHANNELS:
       print(f"Skipping array at index {idx}: Expected {NUM_CHANNELS} channels, got {eeg_array.shape[0]}")
       continue


   total_columns = eeg_array.shape[1]
   if total_columns < END_TIMESTAMP:
       print(f"Skipping array at index {idx}: Not enough columns (has {total_columns}, needs {END_TIMESTAMP})")
       continue


   sliced_array = eeg_array[:, START_TIMESTAMP - 1 : END_TIMESTAMP]


   if sliced_array.shape[1] != EXPECTED_WIDTH:
       print(f"Warning: Sliced array at index {idx} has width {sliced_array.shape[1]}, expected {EXPECTED_WIDTH}")


   processed_data.append(sliced_array)
   processed_labels.append(label)


processed_dict = {
   'data': processed_data,
   'labels': processed_labels
}


output_file = '/content/aggregated_data_uniform_width.npy'


np.save(output_file, processed_dict)


print(f"\nProcessed data and labels saved to {output_file}")
print(f"Total processed samples: {len(processed_data)}")
print(f"Total labels: {len(processed_labels)}")


#####


file_path = '/content/aggregated_data_uniform_width.npy'
train_split = 0.9
channel_names = [
   "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ",
   "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2",
   "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4",
   "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6",
   "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7",
   "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "CB1", "O1", "OZ", "O2", "CB2"
]


# ============================
# Load data
# ============================
data_dict = np.load(file_path, allow_pickle=True).item()
all_data = data_dict['data']     
all_labels = data_dict['labels'] 


assert len(all_data) == len(all_labels), "Mismatch in number of samples and labels"


total_samples = len(all_data)
print(f"Total samples found: {total_samples}")


# ============================
# Train-Test Split
# ============================
num_train = int(total_samples * train_split)
num_test = total_samples - num_train


indices = np.arange(total_samples)
np.random.shuffle(indices)


train_indices = indices[:num_train]
test_indices = indices[num_train:]


X_train = [all_data[i] for i in train_indices]
y_train = [all_labels[i] for i in train_indices]
X_test = [all_data[i] for i in test_indices]
y_test = [all_labels[i] for i in test_indices]


np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)


print(f"Train samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")


# ============================
# Compute Average Correlation Matrix on TRAIN DATA
# ============================


num_channels = 62
corr_sum = np.zeros((num_channels, num_channels))


for i, sample in enumerate(X_train):
   corr_matrix = np.corrcoef(sample)
   corr_sum += corr_matrix


avg_corr_matrix = corr_sum / len(X_train)


print("Averaged 62x62 correlation matrix computed.")


# ============================
# Plot the Averaged Correlation Matrix
# ============================
plt.figure(figsize=(10, 8))
plt.imshow(avg_corr_matrix, cmap='jet', aspect='auto', vmin=-1, vmax=1)
plt.colorbar(label='Correlation Coefficient')
plt.title('Average Correlation Matrix (Across Training Samples)')
plt.xticks(range(num_channels), channel_names, rotation=90)
plt.yticks(range(num_channels), channel_names)
plt.tight_layout()
plt.show()


np.save('average_correlation_matrix.npy', avg_corr_matrix)
print("Average correlation matrix saved to 'average_correlation_matrix.npy'.")

