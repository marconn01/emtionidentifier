import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import os

# Load the pre-trained VGG16 model
model = VGG16(weights='imagenet', include_top=False, pooling='avg')

# Function to extract features from an image using VGG16 model
def extract_features(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features.flatten()

# Main code
# Specify the path to the main folder containing the subfolders with images
main_folder = 'emtionidentifier\Train\DS'

# Lists to store the features and labels
features = []
labels = []

# Iterate over each folder in the main folder
for folder_name in os.listdir(main_folder):
    folder_path = os.path.join(main_folder, folder_name)
    if not os.path.isdir(folder_path):
        continue
    
    print('Processing images in folder:', folder_name)
    
    # Iterate over each image in the folder
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        
        # Extract features for the image
        img_features = extract_features(image_path)
        
        # Append the features and label to the lists
        features.append(img_features)
        labels.append(folder_name)
        
    print('Finished processing images in folder:', folder_name)

# Convert features and labels to NumPy arrays
features = np.array(features)
labels = np.array(labels)

# Print the shape of the feature matrix and labels
print('Shape of feature matrix:', features.shape)
print('Shape of labels:', labels.shape)