from pydantic import BaseModel
import models
import cv2
import matplotlib.pyplot as plt
from PIL import Image
# class ImageSearchRequest(BaseModel):
#     target_image_path: str
#     top10_image_list: list[str]

# app = FastAPI()
# Initialize the models.
target_image_path = "C:\\Users\\DGU_ICE\\FindOwn\\ImageDB\\KakaoTalk_20230216_133749847.png"
target_image_path = "https://trademark.help-me.kr/images/blog/trademark-registration-all-inclusive/image-05.png"
similar_results = []

similar_model = models.Image_Search_Model(pre_extracted_features='features_logo.pkl')
top_10_image_list = similar_model.search_similar_images(target_image_path)
print(top_10_image_list)
# for similar_accuracy in top_10_image_list:
#     similar_results.append(similar_accuracy * 0.6)
    
root_dir = 'C:\\Users\\DGU_ICE\\FindOwn\\ImageDB\\Logos'
plt.figure(figsize=(20, 20))   # 플롯 크기 설정
for i, (img_path, sim) in enumerate(top_10_image_list):
    img = Image.open(img_path)    # 이미지 불러오기
    plt.subplot(2, 5, i+1)        # 2행 5열의 서브플롯 생성
    plt.imshow(img)               # 이미지 출력
    plt.title(f"Rank {i+1}: Similaritysimilar {sim}")    # 이미지 제목 설정
    plt.axis('off')               # 축 레이블 제거

plt.tight_layout()    # 플롯 간격 조절
plt.show()   
    
color_model = models.ColorSimilarityModel()
histograms = color_model.load_histograms('colorHistograms_logo.pkl')
similarities = color_model.predict(target_image_path, histograms)
top_10_images = similarities[:10]
print(top_10_images)
# for color_similarity in top_10_images:
    
plt.figure(figsize=(20, 20))   # 플롯 크기 설정
for i, (img_path, sim) in enumerate(top_10_images):
    img = Image.open(str(root_dir+'\\'+img_path))    # 이미지 불러오기
    plt.subplot(2, 5, i+1)        # 2행 5열의 서브플롯 생성
    plt.imshow(img)               # 이미지 출력
    plt.title(f"Rank {i+1}: SimilarityColor {sim}")    # 이미지 제목 설정
    plt.axis('off')               # 축 레이블 제거

plt.tight_layout()    # 플롯 간격 조절
plt.show()            # 플롯 보여주기 

Object_model  = models.Image_Object_Detections()
result = Object_model.search_similar_images(target_image_path,top_10_image_list)
print(result)
plt.figure(figsize=(20, 20))   # 플롯 크기 설정
for i, (img_path, _ ,sim) in enumerate(result):
    img = Image.open(str(img_path))    # 이미지 불러오기
    plt.subplot(2, 5, i+1)        # 2행 5열의 서브플롯 생성
    plt.imshow(img)               # 이미지 출력
    plt.title(f"Rank {i+1}: SimilarityObject {sim}")    # 이미지 제목 설정
    plt.axis('off')               # 축 레이블 제거

plt.tight_layout()    # 플롯 간격 조절
plt.show() 

# @app.post("/search_images")
# async def search_images(request: ImageSearchRequest):
#     try:
#         result = Object_model.search_similar_images(request.target_image_path, request.top10_image_list)
#         return {"result": result}
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))