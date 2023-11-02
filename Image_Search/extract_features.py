import models
import pickle
import os

def main():
    root_dir = "변환하고자 하는 이미지들이 있는 폴더로 지정 ex) C:\\user\\sam\\image "
    similar_model1 = models.Image_Search_Model(root_dir)
    Trademark_pkl = similar_model1.extract_features()
    with open('features_logo.pkl','wb') as f:
        pickle.dump(list(Trademark_pkl),f)      
    with open('features_logo.pkl','rb') as f:
        load = pickle.load(f)
    print(len(load))
    
    #color model
    color_model = models.ColorSimilarityModel()
    color_model.save_histograms(root_dir,'colorHistograms_logo.pkl')
if __name__ == '__main__':
    main()