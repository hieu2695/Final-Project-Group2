import os

# Make Model directory
directory = os.path.dirname('../Model/')
if not os.path.exists(directory):
    os.makedirs(directory)

# Make Figure directory
directory = os.path.dirname('../Figure/')
if not os.path.exists(directory):
    os.makedirs(directory)

# Make processed_data directory
directory = os.path.dirname('../processed_data/')
if not os.path.exists(directory):
    os.makedirs(directory)

# Make processed_data/Npy directory
directory = os.path.dirname('../processed_data/Npy/')
if not os.path.exists(directory):
    os.makedirs(directory)

# Make processed_data/splitted_data directory
directory = os.path.dirname('../processed_data/splitted_data/')
if not os.path.exists(directory):
    os.makedirs(directory)