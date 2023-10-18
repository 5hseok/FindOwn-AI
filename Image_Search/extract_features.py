# extract_features.py

import models
import pickle

# Initialize the model.
similar_model = models.Image_Search_Model("C:\\Users\\DGU_ICE\\FindOwn\\ImageDB\\Logos")

# Extract features from all images.
similar_model.extract_features()

# Save the features to a file.
with open('features.pkl', 'wb') as f:
    pickle.dump(similar_model.features, f)
