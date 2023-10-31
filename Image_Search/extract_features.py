import models
import pickle
import os

def main():
    root_dir = "C:\\Users\\DGU_ICE\\AI_Trademark_IMG"
    similar_model1 = models.Image_Search_Model(root_dir)
    Trademark_pkl = similar_model1.extract_features()
    with open('features_Trademark.pkl','wb') as f:
        pickle.dump(list(Trademark_pkl),f)      
    with open('features_Trademark.pkl','rb') as f:
        load = pickle.load(f)
    print(len(load))

if __name__ == '__main__':
    main()