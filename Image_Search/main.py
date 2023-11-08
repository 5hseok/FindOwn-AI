from pydantic import BaseModel
import models
import cv2
import os
import pickle
import os
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg
def min_max_normalize(scores):
    min_score = min(scores)
    max_score = max(scores)
    return [(score - min_score) / (max_score - min_score) for score in scores]
# Initialize the models.
# url을 받아오는 걸로 변경 요망
################################################################################################################
target_image_path= "https://trademark.help-me.kr/images/blog/trademark-registration-all-inclusive/image-05.png"
#빽다방
# target_image_path = "https://upload.wikimedia.org/wikipedia/ko/thumb/3/33/%ED%86%A0%ED%8A%B8%EB%84%98_%ED%99%8B%EC%8A%A4%ED%8D%BC_FC_%EB%A1%9C%EA%B3%A0.svg/800px-%ED%86%A0%ED%8A%B8%EB%84%98_%ED%99%8B%EC%8A%A4%ED%8D%BC_FC_%EB%A1%9C%EA%B3%A0.svg.png"
# #토트넘
# target_image_path = "https://scontent-gmp1-1.xx.fbcdn.net/v/t39.30808-1/362637775_114168721749133_6181487638898434694_n.jpg?stp=cp0_dst-jpg_e15_p120x120_q65&_nc_cat=110&ccb=1-7&_nc_sid=5f2048&_nc_ohc=jmClhneOLzoAX8jTvls&_nc_ht=scontent-gmp1-1.xx&oh=00_AfCVJOHkeT9DFmpRTNrXp7lXqZTXkUy03MJ6U9YywHMcJg&oe=654A97FD"
# #백소정
################################################################################################################
root_dir = "C:\\Users\\DGU_ICE\\FindOwn\\ImageDB\\Logos"
#target_image_path를 url로 받아오면 아래 코드로 유사도 검사 후 결과 dict를 json으로 만들어 다시 전송
similar_results_dict = {}

if not os.path.exists('features_logo.pkl'):
    similar_model = models.Image_Search_Model()
    Trademark_pkl = similar_model.extract_features(root_dir)
    with open('features_logo.pkl','wb') as f:
        pickle.dump(list(Trademark_pkl),f)    
    
with open('features_logo.pkl','rb') as f:
    load = pickle.load(f)
for image_path, array in load:
    similar_results_dict.update({image_path:0.0})

similar_model = models.Image_Search_Model(pre_extracted_features='features_logo.pkl')
efficientnet_image_list = similar_model.search_similar_images(target_image_path,len(similar_results_dict))
efficientnet_scores = [accuracy for img_path, accuracy in efficientnet_image_list]
efficientnet_scores = min_max_normalize(efficientnet_scores)
for (image_path, _), score in zip(efficientnet_image_list,efficientnet_scores):
    similar_results_dict[image_path] += 0.6 * score
  
    
color_model = models.ColorSimilarityModel()
if not os.path.exists('colorHistograms_logo.pkl'):
    color_model.save_histograms(root_dir,'colorHistograms_logo.pkl')
histograms = color_model.load_histograms('colorHistograms_logo.pkl')
similarities = color_model.predict(target_image_path, histograms)
color_scores = [accuracy for img_path, accuracy in similarities]
color_scores = min_max_normalize(color_scores)
for (image_path, _), score in zip(similarities,color_scores):
    similar_results_dict[image_path] += 0.1 * score

    
    
Object_model  = models.Image_Object_Detections(len(similar_results_dict))
if not os.path.exists('object_logo.pkl'):
    Object_model.create_object_detection_pkl(root_dir,'object_logo.pkl')
with open('object_logo.pkl','rb') as f:
    detection_dict = pickle.load(f)
result = Object_model.search_similar_images(target_image_path,detection_dict)
object_scores = [accuracy for img_path, _, accuracy in result]
object_scores = min_max_normalize(object_scores)
for (img_path, _, _),score in zip(result, object_scores):
    similar_results_dict[img_path] += 0.1 * score


cnn = models.CNNModel()
if not os.path.exists('cnn_features.pkl'):
    cnn.extract_features_from_dir(root_dir, 'cnn_features.pkl')
cnn_similarities = cnn.compare_features(target_image_path, 'cnn_features.pkl')
cnn_scores = [accuracy for img_path, accuracy in cnn_similarities]
cnn_scores = min_max_normalize(cnn_scores)
for (img_path, _ ), score in zip(cnn_similarities,cnn_scores):
    img_path = root_dir+'\\'+img_path
    similar_results_dict[img_path] += 0.2 * score
    
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
        img = Image.open(url).convert('RGB')
        img = np.array(img)
else:
    img = mpimg.imread(target_image_path)
ax[0].imshow(img)
ax[0].set_title("Target Image")

# Display top N similar images
for i in range(1, N+1):
    img_path, accuracy = similar_results_dict[i-1]
    if img_path.startswith('http://') or img_path.startswith('https://'):
        with urllib.request.urlopen(img_path) as url:
            img = Image.open(url).convert('RGB')
            img = np.array(img)
    else:
        img = mpimg.imread(img_path)
    ax[i].imshow(img)
    ax[i].set_title("Similarity: {:.8f}".format(accuracy))

plt.tight_layout()
plt.show()
