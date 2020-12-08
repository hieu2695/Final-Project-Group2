import numpy as np
import matplotlib.pyplot as plt
from skmultilearn.model_selection import iterative_train_test_split

# %%------------ Load data --------------------------
x = np.array(np.load("../processed_data/inputs.npy", allow_pickle=True), dtype="float32")
y = np.array(np.load("../processed_data/targets.npy", allow_pickle=True), dtype="float32")

# %% ----------------
trans_y = y.T
n_classes = 28

counts_0 = []
counts_1 = []
for i in range(n_classes):
    value, count = np.unique(trans_y[i], return_counts=True)
    counts_0.append(count[0])
    if len(count) == 2:
        counts_1.append(count[1])
    else:
        counts_1.append(0)

inds = np.arange(n_classes)
labels = ["Nucleoplasm",
          "Nuclear membrane",
          "Nucleoli",
          "Nucleoli fibrillar center",
          "Nuclear speckles",
          "Nuclear bodies",
          "Endoplasmic reticulum",
          "Golgi apparatus",
          "Peroxisomes",
          "Endosomes",
          "Lysosomes",
          "Intermediate filaments",
          "Actin filaments",
          "Focal adhesion sites",
          "Microtubules",
          "Microtubule ends",
          "Cytokinetic bridge",
          "Mitotic spindle",
          "Microtubule organizing center",
          "Centrosome",
          "Lipid droplets",
          "Plasma membrane",
          "Cell junctions",
          "Mitochondria",
          "Aggresome",
          "Cytosol",
          "Cytoplasmic bodies",
          "Rods & rings"]

plt.figure(figsize=(10, 10))
plt.bar(inds, counts_0, width=0.4, label="Non Appearance")
plt.bar(inds, counts_1, bottom=counts_0, width=0.4, label="Appearance")
plt.xticks(inds, labels, rotation=90)
plt.legend(loc='best')
plt.title("Human Protein Atlas Distribution", fontsize=15)
plt.ylabel("Frequency", fontsize=13)
plt.xlabel("Types", fontsize=13)
plt.show()

# %% ------------------------ Spliting data into training and testing sets --------------
x_train, y_train, x_test, y_test = iterative_train_test_split(x, y, test_size=0.15)
x_train, y_train, x_val, y_val = iterative_train_test_split(x_train, y_train, test_size=0.15)

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
print(x_test.shape, y_test.shape)
# %% ------------------------- Saving data ---------------------------------------
np.save("../processed_data/splitted_data/x_train", x_train)
np.save("../processed_data/splitted_data/y_train.npy", y_train)

np.save("../processed_data/splitted_data/x_val", x_val)
np.save("../processed_data/splitted_data/y_val.npy", y_val)

np.save("../processed_data/splitted_data/x_test.npy", x_test)
np.save("../processed_data/splitted_data/y_test.npy", y_test)

#%% -------------------------- Finding rare classes -----------------------------
x_rare = []
y_rare = []

for i in range(len(y_train)):
    if (y[i,8] == 1) or (y[i,9] == 1) or (y[i,10] == 1) or (y[i,15] == 1) or (y[i,17] == 1) or (y[i,20] == 1) or  (y[i,27] == 1):
        y_rare.append(y[i])
        x_rare.append(x[i])
x_rare = np.array(x_rare)
y_rare = np.array(y_rare)

#%% -------------------------- Oversampling --------------------------------------
for k in range(4):
    x_train = np.concatenate((x_train, x_rare))
    y_train = np.concatenate((y_train, y_rare))

#%% --------------------- Save resampled data -------------
np.save("../processed_data/splitted_data/x_train_over.npy", x_train)
np.save("../processed_data/splitted_data/y_train_over.npy", y_train)
# %%--------------
print("Done!")
