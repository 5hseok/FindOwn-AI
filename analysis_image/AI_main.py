from . import AI_models
import os
import pickle
import os
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg
from django.conf import settings
import requests
import json
from tqdm import tqdm
from analysis_image.models import ImageFeature_CNN

class Image_Analysis:
    def min_max_normalize(self, scores):
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            return [1.0 for _ in scores]
        return [(score - min_score) / (max_score - min_score) for score in scores]

    def normalize_score(self, scores_list):
        max = 5.0
        scores_list = [(img_path,score / max) for (img_path,score) in scores_list]
        return scores_list

    def start_analysis(self, target_image_path, test_value):
        # Initialize the AI_models.
        # url을 받아오는 걸로 변경 요망
        ################################################################################################################
        #Test#
        # target_image_path = "C:\\Users\\DGU_ICE\\FindOwn\\ImageDB\\loading.png"
        # target_image_path= "C:\\Users\\DGU_ICE\\FindOwn\\ImageDB\\fakestar.png"
        # target_image_path = "C:\\Users\\DGU_ICE\\FindOwn\\ImageDB\\fakecapa.png"

        ################################################################################################################
        root_dir = "C:\\Users\\FindOwn\\AI_Trademark_IMG"
        similar_results_dict = {}
        #resnet_results
        cnn = AI_models.CNNModel()

        # 데이터베이스에서 모든 ImageFeature_CNN 객체를 가져옵니다.
        all_features = ImageFeature_CNN.objects.all()

        # 각 객체의 image_path와 feature를 similar_results_dict에 추가합니다.
        for feature in all_features:
            similar_results_dict.update({feature.image_path: 0.0})

        # pkl 파일 대신 데이터베이스의 내용을 사용하여 cnn_similarities를 계산합니다.
        # 이를 위해 compare_features 함수를 수정하여 데이터베이스에서 가져온 feature를 입력으로 받도록 해야 합니다.
        cnn_similarities = cnn.compare_features(target_image_path, all_features)

        cnn_scores = [accuracy for img_path, accuracy in cnn_similarities]
        cnn_scores = self.min_max_normalize(cnn_scores)

        
        for (img_path, _ ), score in zip(cnn_similarities,cnn_scores):
            img_path = img_path
            similar_results_dict[img_path] += 3.0 * score
            
        #         # color Histogram_result
        # color_model = AI_models.ColorSimilarityModel()
        # joblib_path =os.path.join(settings.BASE_DIR, 'analysis_image', 'colorHistograms_logo_Kipris.joblib')
        # if not os.path.exists(joblib_path):
        #     color_model.save_histograms(root_dir,joblib_path)

        # histograms = color_model.load_histograms(joblib_path)
        # similarities = color_model.predict(target_image_path, histograms)
        # color_scores = [accuracy for img_path, accuracy in similarities]
        # color_scores = self.min_max_normalize(color_scores)
        # for (image_path, _), score in zip(similarities,color_scores):
        #     try:
        #         similar_results_dict[image_path] -= 1.0 * score

        #     except KeyError:
        #         pass
            
        # pkl_path =os.path.join(settings.BASE_DIR, 'analysis_image', 'features_logo_Kiprix.pkl')
        # if not os.path.exists(pkl_path):
        #     similar_model = AI_models.Image_Search_Model()
        #     Trademark_pkl = similar_model.extract_features(root_dir)  
        # #EfficientNet_results
        # similar_model = AI_models.Image_Search_Model(pre_extracted_features=pkl_path)
        # efficientnet_image_list = similar_model.search_similar_images(target_image_path,len(similar_results_dict))
        # efficientnet_scores = [accuracy for img_path, accuracy in efficientnet_image_list]
        # efficientnet_scores = self.min_max_normalize(efficientnet_scores)
        # for (image_path, _), score in zip(efficientnet_image_list,efficientnet_scores):
        #     similar_results_dict[image_path] += 0.9 * score
        
        #     # object_detection_retinanet_result
        # Object_model  = AI_models.Image_Object_Detections(len(similar_results_dict))
        # if not os.path.exists('object_logo_Kipris.pkl'):
        #     Object_model.create_object_detection_pkl(root_dir,'object_logo_Kipris.pkl')
        # with open('object_logo_Kipris.pkl','rb') as f:
        #     detection_dict = pickle.load(f)
        # result = Object_model.search_similar_images(target_image_path,detection_dict)
        # if len(result) != 0:
        #     object_scores = [accuracy for img_path, _, accuracy in result]
        #     object_scores = self.min_max_normalize(object_scores)
        #     for (img_path, _, _),score in zip(result, object_scores):
        #         similar_results_dict[img_path] += 0.15 * score
            
        similar_results_dict = sorted(similar_results_dict.items(), key=lambda x: x[1], reverse=True)
        if test_value is True:
            #################################   Print Test Code  #########################################
            import matplotlib.image as mpimg
            import urllib.request
            import numpy as np
            from PIL import Image

            N = 10  # Display top N images
            fig, ax = plt.subplots(1, N+1, figsize=(20, 10))

            # Display target image
            if target_image_path.startswith('http://') or target_image_path.startswith('https://'):
                with urllib.request.urlopen(target_image_path) as url:
                    img = Image.open(url)
            else:
                img = mpimg.imread(target_image_path)
            ax[0].imshow(img)
            ax[0].set_title("Target Image (User's Image)")

            # Display top N similar images
            for i in range(1, N+1):
                img_path, accuracy = similar_results_dict[i-1]
                if img_path.startswith('http://') or img_path.startswith('https://'):
                    with urllib.request.urlopen(img_path) as url:
                        img = Image.open(url)
                        
                else:
                    img = mpimg.imread(img_path)
                ax[i].imshow(img)
                ax[i].set_title("Similarity : {:.8f}".format(accuracy))

            plt.tight_layout()
            plt.show()


        ####                                Sending json to server                                ####

        # Create a list of dictionaries, each containing the image path and accuracy
        top_results = []
        N=3

        # 이미지 침해도 설정
        specific_Logo = True
        for img_path, accuracy in self.normalize_score(similar_results_dict[:N]):
            if "disney" in os.path.basename(img_path) or "mickey" in os.path.basename(img_path) or "monster" in os.path.basename(img_path) or "minnie" in os.path.basename(img_path):
                specific_Logo = False
            if specific_Logo and accuracy > 0.82 :
                top_results.append((img_path, "danger", accuracy))
            elif accuracy > 0.74:
                top_results.append((img_path, "warning", accuracy))
            else:
                top_results.append((img_path, "safe", accuracy))

        results_list = [{"image_path": img_path, "result": result, "accuracy":accuracy} for img_path, result, accuracy in top_results]

        # Convert the list to JSON
        results_json = json.dumps(results_list)
        # Show json
        data = json.loads(results_json)
        # print data
        print(data)
        return results_json
        
if __name__ == "__main__":
    image_analysis = Image_Analysis()
    image_analysis.start_analysis("https://trademark.help-me.kr/images/blog/trademark-registration-all-inclusive/image-05.png", True)                      