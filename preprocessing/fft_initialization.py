import matplotlib.pyplot as plt


train_file = '/content/numerical_train.csv'
test_file = '/content/numerical_test.csv'


train_data = np.loadtxt(train_file, delimiter=',', skiprows=1, encoding='utf-8-sig')
test_data = np.loadtxt(test_file, delimiter=',', skiprows=1, encoding='utf-8-sig')


train_labels = train_data[:, 0]
train_features = train_data[:, 1:]
test_labels = test_data[:, 0]
test_features = test_data[:, 1:]


train_mean = np.mean(train_features)
train_std = np.std(train_features)


test_mean = np.mean(test_features)
test_std = np.std(test_features)


if train_std == 0:
   train_features_norm = train_features
else:
   train_features_norm = (train_features - train_mean) / train_std


if test_std == 0:
   test_features_norm = test_features
else:
   test_features_norm = (test_features - test_mean) / test_std


train_matrices = train_features_norm.reshape(train_features_norm.shape[0], 14, 256)
test_matrices = test_features_norm.reshape(test_features_norm.shape[0], 14, 256)


def row_fft_power_spectrum(row_256):
   fft_vals = np.fft.rfft(row_256, n=256)
   power = np.abs(fft_vals)**2
   return power[:128]


train_transformed = np.zeros((train_matrices.shape[0], 14, 128))
for i in range(train_matrices.shape[0]):
   for j in range(14):
       train_transformed[i, j, :] = row_fft_power_spectrum(train_matrices[i, j, :])


test_transformed = np.zeros((test_matrices.shape[0], 14, 128))
for i in range(test_matrices.shape[0]):
   for j in range(14):
       test_transformed[i, j, :] = row_fft_power_spectrum(test_matrices[i, j, :])


np.save('train_labels.npy', train_labels)
np.save('train_transformed.npy', train_transformed)
np.save('test_labels.npy', test_labels)
np.save('test_transformed.npy', test_transformed)


##########################################
# Sanity checks and plotting
##########################################


first_row_original = train_features[0, :256] 
fft_original = np.fft.rfft(first_row_original, n=256)
power_original = np.abs(fft_original)**2


plt.figure(figsize=(10,4))
plt.title('FFT Power Spectrum of first 256 points (Original)')
plt.plot(power_original, label='Original first 256 segment')
plt.xlabel('Frequency bin')
plt.ylabel('Power')
plt.legend()
plt.grid(True)
plt.show()


first_example_transformed = train_transformed[0]
first_channel_transformed = first_example_transformed[0, :]


plt.figure(figsize=(10,4))
plt.title('Transformed FFT (14x128) first channel of first sample')
plt.plot(first_channel_transformed, label='First channel (normalized & processed)')
plt.xlabel('Frequency bin')
plt.ylabel('Power')
plt.legend()
plt.grid(True)
plt.show()
