from pydantic import BaseModel
import models
import cv2
import pickle
import os
import matplotlib.pyplot as plt
from PIL import Image

# Initialize the models.
# url을 받아오는 걸로 변경 요망
################################################################################################################
target_image_path = "https://trademark.help-me.kr/images/blog/trademark-registration-all-inclusive/image-05.png"
################################################################################################################
root_dir = "C:\\Users\\DGU_ICE\\FindOwn\\ImageDB\\Logos"
#target_image_path를 url로 받아오면 아래 코드로 유사도 검사 후 결과 dict를 json으로 만들어 다시 전송
similar_results_dict = {}
image_list = []
with open('features_logo.pkl','rb') as f:
    load = pickle.load(f)
for image_path, array in load:
    similar_results_dict.update({image_path:0.0})
    image_list.append(image_path)
similar_model = models.Image_Search_Model(pre_extracted_features='features_logo.pkl')
efficientnet_image_list = similar_model.search_similar_images(target_image_path)
for image_path, accuracy in efficientnet_image_list:
    similar_results_dict[image_path] = 0.6*accuracy
  
    
color_model = models.ColorSimilarityModel()
if not os.path.exists('colorHistograms_logo.pkl'):
    color_model.save_histograms(root_dir,'colorHistograms_logo.pkl')
histograms = color_model.load_histograms('colorHistograms_logo.pkl')
similarities = color_model.predict(target_image_path, histograms)
color_dicision_images = similarities

for img_path, color_accuracy in color_dicision_images:
    similar_results_dict[img_path] += 0.3 * color_accuracy


Object_model  = models.Image_Object_Detections(len(image_list))
result = Object_model.search_similar_images(target_image_path,image_list)
print(result)
for img_path, object_accuracy in result:
    similar_results_dict[img_path] += 0.1 * object_accuracy
print(similar_results_dict.sort())




# @app.post("/search_images")
# async def search_images(request: ImageSearchRequest):
#     try:
#         result = Object_model.search_similar_images(request.target_image_path, request.top10_image_list)
#         return {"result": result}
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))