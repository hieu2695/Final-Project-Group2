## DATA SOURCE

https://www.kaggle.com/c/human-protein-atlas-image-classification/data

The data can be manually downloaded after accepting the competition rule.

Using Kaggle CLI command (requires Kaggle API token):

``
kaggle competitions download -c human-protein-atlas-image-classification
``

------------------

## Download from GCP

For convenience, I uploaded the data to my GCP so it can be downloaded via terminal using wget command.

Assuming that we are at the git repository Final-Project-Group2, else we need to move to the repo:

``
cd ~/Final-Project-Group2
`` 

At the git repo, download data via GCP using the command:

``
wget https://storage.googleapis.com/letranhieu-bucket-data/data.zip
``

Then, unzip the data.zip file:

``
unzip data.zip
``

The data will be unzipped to a folder named "data" containing "train" folder and "train.csv"



