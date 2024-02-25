#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pickle


# In[3]:


from pathlib import Path


# In[4]:


import os


# In[5]:


folder_path = r"C:\Users\dell\Downloads\Dataset"
image_list = []


# In[6]:


import cv2


# In[7]:


root='C:\\Users\\dell\\Downloads\\Dataset\\Dataset'  
fnames=os.listdir(root)


# In[10]:


len(fnames)


# In[11]:


import cv2
import os

root = 'C:\\Users\\dell\\Downloads\\Dataset\\Dataset'
fnames = os.listdir(root)

# Display images from the folder
for filename in fnames:
    if filename.endswith(('.jpg', '.jpeg', '.png')):  # Add more image extensions if needed
        image_path = os.path.join(root, filename)
        image = cv2.imread(image_path)
        if image is not None:
            # Display image and its length
            cv2.imshow(f'Image: {filename} - Length: {len(image)}', image)
            cv2.waitKey(500)  # Add a delay of 500 milliseconds

# Wait for a key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[8]:


import zipfile as zf
files = zf.ZipFile(r"C:\Users\dell\Downloads\Dataset.zip", 'r')
files.extractall('Dataset')
files.close()


# In[17]:


import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Specify the correct folder path
folder_path = r"C:\Users\dell\Downloads\Dataset\Dataset"

# Get a list of all image files in the folder
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Process each image in the folder
for image_file in image_files:
    # Construct the full path to the image
    image_path = os.path.join(folder_path, image_file)

    # Load the image
    image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if image is not None:
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        # Perform edge detection using the Canny edge detector
        edges = cv2.Canny(blurred_image, 50, 150)

        # Display the original and processed images side by side
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(f'Original Image: {image_file}')

        plt.subplot(1, 3, 2)
        plt.imshow(gray_image, cmap='gray')
        plt.title('Grayscale Image')

        plt.subplot(1, 3, 3)
        plt.imshow(edges, cmap='gray')
        plt.title('Canny Edge Detection')

        plt.show()
    else:
        print(f"Failed to load the image: {image_file}")


# In[18]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# In[29]:


import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense


def extract_features(image_path):
    # Load the image in color
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the image to a fixed size (e.g., 100x100)
    resized_image = cv2.resize(gray_image, (100, 100))

    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(resized_image, (5, 5), 0)

    # Compute Histogram of Oriented Gradients (HOG) features
    nbins = 9
    win_size = (100, 100)

    # Adjust block size and cell size to ensure they divide evenly into the window size
    block_size = (win_size[0] // 2, win_size[1] // 2)
    cell_size = (block_size[0] // 2, block_size[1] // 2)

    hog = cv2.HOGDescriptor(win_size, block_size, cell_size, cell_size, nbins)
    hog_features = hog.compute(blurred_image).flatten()

    # Extract additional statistical features (e.g., mean, standard deviation)
    mean_intensity = np.mean(resized_image)
    std_intensity = np.std(resized_image)

    # Combine all features into a single array
    features = np.concatenate([hog_features, [mean_intensity, std_intensity]])

    return features
# Specify the folder path containing diabetes patient eye images
folder_path = r"C:\Users\dell\Downloads\Dataset\Dataset"

# Get a list of all image files in the folder
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Create lists to store features and corresponding glucose levels
features_list = []
glucose_levels = []

# Iterate through each image in the folder
for image_file in image_files:
    # Construct the full path to the image
    image_path = os.path.join(folder_path, image_file)

    # Extract features from the image
    features = extract_features(image_path)

    # Append features to the list
    features_list.append(features)

    # Replace this with the actual glucose level associated with the image
    # For demonstration purposes, using a random value between 80 and 120
    glucose_levels.append(np.random.uniform(80, 120))

# Convert lists to NumPy arrays
X = np.array(features_list)
y = np.array(glucose_levels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the number of samples in the training and testing sets
print(f'Number of samples in the training set: {len(X_train)}')
print(f'Number of samples in the testing set: {len(X_test)}')

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest Regressor
model = RandomForestRegressor()
model.fit(X_train_scaled, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test_scaled)

# Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Visualize the predicted vs. true glucose levels
plt.scatter(y_test, y_pred)
plt.xlabel('True Glucose Levels')
plt.ylabel('Predicted Glucose Levels')
plt.title('Glucose Level Prediction')
plt.show()


# In[31]:


import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

def extract_features(image_path):
    # Load the image in color
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the image to a fixed size (e.g., 100x100)
    resized_image = cv2.resize(gray_image, (100, 100))

    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(resized_image, (5, 5), 0)

    # Compute Histogram of Oriented Gradients (HOG) features
    nbins = 9
    win_size = (100, 100)

    # Adjust block size and cell size to ensure they divide evenly into the window size
    block_size = (win_size[0] // 2, win_size[1] // 2)
    cell_size = (block_size[0] // 2, block_size[1] // 2)

    hog = cv2.HOGDescriptor(win_size, block_size, cell_size, cell_size, nbins)
    hog_features = hog.compute(blurred_image).flatten()

    # Extract additional statistical features (e.g., mean, standard deviation)
    mean_intensity = np.mean(resized_image)
    std_intensity = np.std(resized_image)

    # Combine all features into a single array
    features = np.concatenate([hog_features, [mean_intensity, std_intensity]])

    return features

def build_neural_network(input_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))  # Output layer for regression

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Specify the folder path containing diabetes patient eye images
folder_path = r"C:\Users\dell\Downloads\Dataset\Dataset"

# Get a list of all image files in the folder
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Create lists to store features and corresponding glucose levels
features_list = []
glucose_levels = []

# Iterate through each image in the folder
for image_file in image_files:
    # Construct the full path to the image
    image_path = os.path.join(folder_path, image_file)

    # Extract features from the image
    features = extract_features(image_path)

    # Append features to the list
    features_list.append(features)

    # Replace this with the actual glucose level associated with the image
    # For demonstration purposes, using a random value between 80 and 120
    glucose_levels.append(np.random.uniform(80, 120))

# Convert lists to NumPy arrays
X = np.array(features_list)
y = np.array(glucose_levels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest Regressor
rf_model = RandomForestRegressor()
rf_model.fit(X_train_scaled, y_train)

# Make predictions using Random Forest
y_pred_rf = rf_model.predict(X_test_scaled)

# Train a Neural Network
nn_model = build_neural_network(X_train_scaled.shape[1])
nn_model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=1)

# Make predictions using Neural Network
y_pred_nn = nn_model.predict(X_test_scaled).flatten()

# Evaluate the models
mse_rf = mean_squared_error(y_test, y_pred_rf)
mse_nn = mean_squared_error(y_test, y_pred_nn)

print(f'Mean Squared Error (Random Forest): {mse_rf}')
print(f'Mean Squared Error (Neural Network): {mse_nn}')

# Visualize the predicted vs. true glucose levels for both models
plt.scatter(y_test, y_pred_rf, label='Random Forest')
plt.scatter(y_test, y_pred_nn, label='Neural Network')
plt.xlabel('True Glucose Levels')
plt.ylabel('Predicted Glucose Levels')
plt.legend()
plt.title('Glucose Level Prediction')
plt.show()


# In[32]:


import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

def extract_features(image_path):
    # Load the image in color
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the image to a fixed size (e.g., 100x100)
    resized_image = cv2.resize(gray_image, (100, 100))

    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(resized_image, (5, 5), 0)

    # Compute Histogram of Oriented Gradients (HOG) features
    nbins = 9
    win_size = (100, 100)

    # Adjust block size and cell size to ensure they divide evenly into the window size
    block_size = (win_size[0] // 2, win_size[1] // 2)
    cell_size = (block_size[0] // 2, block_size[1] // 2)

    hog = cv2.HOGDescriptor(win_size, block_size, cell_size, cell_size, nbins)
    hog_features = hog.compute(blurred_image).flatten()

    # Extract additional statistical features (e.g., mean, standard deviation)
    mean_intensity = np.mean(resized_image)
    std_intensity = np.std(resized_image)

    # Combine all features into a single array
    features = np.concatenate([hog_features, [mean_intensity, std_intensity]])

    return features

def build_neural_network(input_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))  # Output layer for regression

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Specify the folder path containing diabetes patient eye images
folder_path = r"C:\Users\dell\Downloads\Dataset\Dataset"

# Get a list of all image files in the folder
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Create lists to store features and corresponding glucose levels
features_list = []
glucose_levels = []

# Iterate through each image in the folder
for image_file in image_files:
    # Construct the full path to the image
    image_path = os.path.join(folder_path, image_file)

    # Extract features from the image
    features = extract_features(image_path)

    # Append features to the list
    features_list.append(features)

    # Replace this with the actual glucose level associated with the image
    # For demonstration purposes, using a random value between 80 and 120
    glucose_levels.append(np.random.uniform(80, 120))

# Convert lists to NumPy arrays
X = np.array(features_list)
y = np.array(glucose_levels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest Regressor
rf_model = RandomForestRegressor()
rf_model.fit(X_train_scaled, y_train)

# Make predictions using Random Forest
y_pred_rf = rf_model.predict(X_test_scaled)

# Train a Neural Network
nn_model = build_neural_network(X_train_scaled.shape[1])
nn_model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=1)

# Make predictions using Neural Network
y_pred_nn = nn_model.predict(X_test_scaled).flatten()

# Scale glucose levels to percentage (assuming the range [0, 100])
y_test_percent = (y_test - np.min(y_test)) / (np.max(y_test) - np.min(y_test)) * 100
y_pred_rf_percent = (y_pred_rf - np.min(y_test)) / (np.max(y_test) - np.min(y_test)) * 100
y_pred_nn_percent = (y_pred_nn - np.min(y_test)) / (np.max(y_test) - np.min(y_test)) * 100

# Evaluate the models
mse_rf = mean_squared_error(y_test_percent, y_pred_rf_percent)
mse_nn = mean_squared_error(y_test_percent, y_pred_nn_percent)

print(f'Mean Squared Error (Random Forest): {mse_rf}')
print(f'Mean Squared Error (Neural Network): {mse_nn}')

# Visualize the predicted vs. true glucose levels for both models
plt.scatter(y_test_percent, y_pred_rf_percent, label='Random Forest')
plt.scatter(y_test_percent, y_pred_nn_percent, label='Neural Network')
plt.xlabel('True Glucose Levels (%)')
plt.ylabel('Predicted Glucose Levels (%)')
plt.legend()
plt.title('Glucose Level Prediction')
plt.show()


# In[ ]:




