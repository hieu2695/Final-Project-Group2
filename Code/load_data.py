import os
import numpy as np
import cv2
import pandas as pd

#%% ---------------- read train.csv -----------------------
df = pd.read_csv("../data/train.csv")


#%%------------- One-hot encoding labels --------------------
# Cell types with corresponding labels
label_dict = {
    0:  "Nucleoplasm",
    1:  "Nuclear membrane",
    2:  "Nucleoli",
    3:  "Nucleoli fibrillar center",
    4:  "Nuclear speckles",
    5:  "Nuclear bodies",
    6:  "Endoplasmic reticulum",
    7:  "Golgi apparatus",
    8:  "Peroxisomes",
    9:  "Endosomes",
    10:  "Lysosomes",
    11:  "Intermediate filaments",
    12:  "Actin filaments",
    13:  "Focal adhesion sites",
    14:  "Microtubules",
    15:  "Microtubule ends",
    16:  "Cytokinetic bridge",
    17:  "Mitotic spindle",
    18:  "Microtubule organizing center",
    19:  "Centrosome",
    20:  "Lipid droplets",
    21:  "Plasma membrane",
    22:  "Cell junctions",
    23:  "Mitochondria",
    24:  "Aggresome",
    25:  "Cytosol",
    26:  "Cytoplasmic bodies",
    27:  "Rods & rings"
}


# add names of labels as columns to df
for key in label_dict.keys():
    df[label_dict[key]] = 0

# one-hot encoding labels
targets = df["Target"].tolist()
for i in range(len(df)):
    targets = df.loc[i, "Target"]
    targets = targets.split(" ")
    for target in targets:
        col_name = label_dict[int(target)]
        df.loc[i,col_name] = 1

#%% ---------------- load images ------------------
# define a function to read RGBY images
def load_RBGY(path,id):
    RESIZE_TO = 128
    colors = ['red','green','blue','yellow']
    img = [cv2.resize(cv2.imread(os.path.join(path, id+'_'+color+'.png'), cv2.IMREAD_GRAYSCALE).astype(np.float32), (RESIZE_TO, RESIZE_TO))
           for color in colors]
    return np.stack(img, axis=-1)

id_list = df["Id"].tolist()

x, y = [], []

# load data

label_col = []
for key in label_dict.keys():
    label_col.append(label_dict[key])

for i in range(len(df)):
    id = df.loc[i, "Id"] # get id

    # load RBGY image
    input = load_RBGY("../data/train/", id)

    label = df.loc[i, label_col].to_numpy() # get labels

    x.append(input)
    y.append(label)

x = np.array(x)
y = np.array(y)

print(x.shape, y.shape)

#%%----------- Save data --------------------
np.save("../processed_data/Npy/inputs.npy", x)
np.save("../processed_data/Npy/targets.npy", y)