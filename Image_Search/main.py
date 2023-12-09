from pydantic import BaseModel
import models
import cv2
import os
import pickle
import os
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg
import requests
import json

def min_max_normalize(scores):
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [1.0 for _ in scores]
    return [(score - min_score) / (max_score - min_score) for score in scores]

def normalize_score(scores_list):
    max = 3.35
    scores_list = [(img_path,score / max) for (img_path,score) in scores_list]
    return scores_list

if __name__ == '__main__':
    # Initialize the models.
    # url을 받아오는 걸로 변경 요망
    ################################################################################################################
    target_image_path= "https://trademark.help-me.kr/images/blog/trademark-registration-all-inclusive/image-05.png"
    #Test#
    # target_image_path = "C:\\Users\\DGU_ICE\\FindOwn\\ImageDB\\loading.png"
    # target_image_path= "C:\\Users\\DGU_ICE\\FindOwn\\ImageDB\\fakestar.png"
    # target_image_path = "C:\\Users\\DGU_ICE\\FindOwn\\ImageDB\\fakecapa.png"

    ################################################################################################################
    root_dir = "C:\\Users\\FindOwn\\AI_Trademark_IMG"
    #target_image_path를 url로 받아오면 아래 코드로 유사도 검사 후 결과 dict를 json으로 만들어 다시 전송
    similar_results_dict = {}
        #resnet_results
    cnn = models.CNNModel()
    if not os.path.exists('cnn_features_Kipris.pkl'):
        cnn.extract_features_from_dir(root_dir, 'cnn_features_Kipris.pkl')
    with open('cnn_features_Kipris.pkl','rb') as f:
        load = pickle.load(f)
    for image_path in load:
        similar_results_dict.update({image_path:0.0})
        
    cnn_similarities = cnn.compare_features(target_image_path, 'cnn_features_Kipris.pkl')
    cnn_scores = [accuracy for img_path, accuracy in cnn_similarities]
    cnn_scores = min_max_normalize(cnn_scores)
    
    for (img_path, _ ), score in zip(cnn_similarities,cnn_scores):
        img_path = img_path
        similar_results_dict[img_path] += 2.5 * score
        
    if not os.path.exists('features_logo.pkl'):
        similar_model = models.Image_Search_Model()
        Trademark_pkl = similar_model.extract_features(root_dir)  
    #EfficientNet_results
    similar_model = models.Image_Search_Model(pre_extracted_features='features_logo.pkl')
    efficientnet_image_list = similar_model.search_similar_images(target_image_path,len(similar_results_dict))
    efficientnet_scores = [accuracy for img_path, accuracy in efficientnet_image_list]
    efficientnet_scores = min_max_normalize(efficientnet_scores)
    for (image_path, _), score in zip(efficientnet_image_list,efficientnet_scores):
        similar_results_dict[image_path] += 0.9 * score
    
        # object_detection_retinanet_result
    Object_model  = models.Image_Object_Detections(len(similar_results_dict))
    if not os.path.exists('object_logo_Kipris.pkl'):
        Object_model.create_object_detection_pkl(root_dir,'object_logo_Kipris.pkl')
    with open('object_logo_Kipris.pkl','rb') as f:
        detection_dict = pickle.load(f)
    result = Object_model.search_similar_images(target_image_path,detection_dict)
    if len(result) != 0:
        object_scores = [accuracy for img_path, _, accuracy in result]
        object_scores = min_max_normalize(object_scores)
        for (img_path, _, _),score in zip(result, object_scores):
            similar_results_dict[img_path] += 0.15 * score
    
    # color Histogram_result
    color_model = models.ColorSimilarityModel()
    if not os.path.exists('colorHistograms_logo_Kipris.pkl'):
        color_model.save_histograms(root_dir,'colorHistograms_logo_Kipris.pkl')
    histograms = color_model.load_histograms('colorHistograms_logo_Kipris.pkl')
    similarities = color_model.predict(target_image_path, histograms)
    color_scores = [accuracy for img_path, accuracy in similarities]
    color_scores = min_max_normalize(color_scores)
    for (image_path, _), score in zip(similarities,color_scores):
        similar_results_dict[image_path] -= 1.0 * score
        
    similar_results_dict = sorted(similar_results_dict.items(), key=lambda x: x[1], reverse=True)

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
    ax[0].set_title("Target Image")

    # Display top N similar images
    for i in range(1, N+1):
        img_path, accuracy = similar_results_dict[i-1]
        if img_path.startswith('http://') or img_path.startswith('https://'):
            with urllib.request.urlopen(img_path) as url:
                img = Image.open(url)
                
        else:
            img = mpimg.imread(img_path)
        ax[i].imshow(img)
        ax[i].set_title("Similarity: {:.8f}".format(accuracy))

    plt.tight_layout()
    plt.show()


    ####                                Sending json to server                                ####

    # Create a list of dictionaries, each containing the image path and accuracy
    top_results = []
    N=3

    # 이미지 침해도 설정
    specific_Logo = True
    for img_path, accuracy in normalize_score(similar_results_dict[:N]):
        if "disney" in os.path.basename(img_path) or "mickey" in os.path.basename(img_path) or "monster" in os.path.basename(img_path) or "minnie" in os.path.basename(img_path):
            specific_Logo = False
        if specific_Logo and accuracy > 0.9 :
            top_results.append((img_path,"danger"))
        elif accuracy > 0.6:
            top_results.append((img_path,"warning"))
        else:
            top_results.append((img_path,"safe"))

    results_list = [{"image_path": img_path, "result": result} for img_path, result in top_results]

    # Convert the list to JSON
    results_json = json.dumps(results_list)

    # Specify the URL to send the POST request to
    url = 'http://example.com/api'  # Change this

    # Set the headers for the request
    headers = {'Content-Type': 'application/json'}

    # Send the POST request
    response = requests.post(url, headers=headers, data=results_json)

    # Check the response
    if response.status_code == 200:
        print('Successfully sent the POST request.')
    else:
        print(f'Failed to send the POST request. Status code: {response.status_code}')
            
    # Show json
    data = json.loads(results_json)
    # print data
    print(data)
    
    # 이미지 파일을 바탕으로 출원번호, 출원년도를 request parameters에 넣어 API를 통해 출원 등록 정보를 가져온다.
    # 가져온 출원 정보와 침해도 정도를 json으로 만들어 백엔드에 보내기