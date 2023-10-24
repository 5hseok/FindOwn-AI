import models
import pickle
import os

def main():
    # Initialize the model.
    # similar_model = models.Image_Search_Model("C:\\Users\\DGU_ICE\\FindOwn\\ImageDB\\Logos", "check1.pkl")
    similar_model1 = models.Image_Search_Model("C:\\Users\\DGU_ICE\\AI_Trademark_IMG","check2.pkl")

    # # Extract features from all images.
    combined_features = []
    
    # with open('features_logo.pkl','wb') as f:
    #     pickle.dump(list(similar_model.extract_features()),f)
    # pickle_file = "C:\\Users\\DGU_ICE\\FindOwn\\check1.pkl"

    # # 파일이 존재하는지 확인 후 삭제
    # if os.path.isfile("C:\\Users\\DGU_ICE\\FindOwn\\Image_Search\\features_logo.pkl") and os.path.isfile(pickle_file):
    #     os.remove(pickle_file)
    # else:
    #     print(f"{pickle_file} does not exist.")
    Trademark_pkl = similar_model1.extract_features()
    with open('features_Trademark.pkl','wb') as f:
        pickle.dump(list(Trademark_pkl),f)      
    # 두 pkl 파일의 데이터를 불러와 combined에 저장하고 pkl로 만들기
    # Save the features to a file.
    # with open('features.pkl', 'wb') as f:
    #     pickle.dump(combined_features, f)

if __name__ == '__main__':
    main()
