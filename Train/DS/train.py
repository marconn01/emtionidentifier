import cv2
import os
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
import numpy as np


def extract_hog_features(image_path):

    image = cv2.imread(image_path)
    

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    

    hog_features = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    
    return hog_features


main_folder = 'emtionidentifier\Train\DS'

features = []
labels = []

for folder_name in os.listdir(main_folder):
    folder_path = os.path.join(main_folder, folder_name)
    if not os.path.isdir(folder_path):
        continue
    
    print('Processing images in folder:', folder_name)

    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        

        hog_features = extract_hog_features(image_path)

        features.append(hog_features)
        labels.append(folder_name)
        
    print('Finished processing images in folder:', folder_name)

features = np.array(features)
labels = np.array(labels)

scaler = StandardScaler()
normalized_features = scaler.fit_transform(features)

print('Shape of feature matrix:', normalized_features.shape)
print('Shape of labels:', labels.shape)