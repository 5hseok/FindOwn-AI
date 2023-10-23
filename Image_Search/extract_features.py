# extract_features.py

import models
import pickle

def main():
    # Initialize the model.
    similar_model = models.Image_Search_Model("C:\\Users\\DGU_ICE\\FindOwn\\ImageDB\\Logos", "check1.pkl")
    similar_model1 = models.Image_Search_Model("C:\\Users\\DGU_ICE\\AI_Trademark_IMG","ckeck2.pkl")

    # Extract features from all images.
    combined_features = []
    
    for feature in similar_model.extract_features():
        combined_features.append(feature)

    for feature in similar_model1.extract_features():
        combined_features.append(feature)

    # Save the features to a file.
    with open('features.pkl', 'wb') as f:
        pickle.dump(combined_features, f)

if __name__ == '__main__':
    main()
