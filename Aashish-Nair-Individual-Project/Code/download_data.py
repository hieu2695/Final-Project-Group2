import os
import pandas as pd
import cv2

if "train" not in os.listdir():
    os.system("wget https://storage.googleapis.com/letranhieu-bucket-data/data.zip")
    os.system("unzip data.zip")
