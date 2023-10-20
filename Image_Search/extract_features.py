# extract_features.py

import models
import pickle

def main():
    # Initialize the model.
    similar_model = models.Image_Search_Model("C:\\Users\\DGU_ICE\\FindOwn\\ImageDB\\Logos")
    similar_model1 = models.Image_Search_Model("C:\\Users\\DGU_ICE\\AI_Trademark_IMG")
    # Extract features from all images.
    combined_features = similar_model.extract_features() + similar_model1.extract_features()

    # Save the features to a file.
    with open('features.pkl', 'wb') as f:
        pickle.dump(combined_features, f)

if __name__ == '__main__':
    main()
