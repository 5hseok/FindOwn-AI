# models.py 사용 얘시

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import models

# Pydantic 모델 정의
class ImageSearchRequest(BaseModel):
    target_image_path: str
    top10_image_list: list[str]

app = FastAPI()
    
# 피클 파일에서 모델 로드
# with open('image_search_model.pkl','rb') as f:
#     similar_model=pickle.load(f)
# with open('image_object_detections.pkl', 'rb') as f:
#     Object_model = pickle.load(f)

similar_model = models.Image_Search_Model("C:\\Users\\DGU_ICE\\FindOwn\\ImageDB\\Logos")
Object_model  = models.Image_Object_Detections()
#사용자가 등록한 이미지 경로
target_image_path = "C:\\Users\\DGU_ICE\\FindOwn\\ImageDB\\Logos\\000002_ef67f5045e3a44bdb9001d956746a391.jpg"  

#이미지 DB에서 target_image와 가장 비슷한 이미지 N개 생성 ( target_image_path, N = 10)
result = similar_model.search_similar_images(target_image_path)
# print(result) 
#result는 이미지의 경로와 유사도를 한 tuple로 묶은 10개의 이미지들의 리스트

#result 이미지들에서 가장 target_image와 object가 많이 겹치는 이미지 3개 찾기(target_image_path, result, object_detection_presicions = 0.05)
#여기서 presiciones는 객체가 그 이미지에 있을 확률. precision이 높을수록 이미지에서 객체를 더 엄격하게 검사한다.(object 수 감소)
final_result = Object_model.search_similar_images(target_image_path, result)
print(Object_model.search_similar_images_test(target_image_path, result))
# print(final_result)
#final_result는 top 3 이미지들의 경로와 겹치는 object의 개수를 각각 튜플로 묶은 총 3개의 원소를 지닌 리스트

@app.post("/search_images")
async def search_images(request: ImageSearchRequest):
    try:
        result = Object_model.search_similar_images(request.target_image_path, request.top10_image_list)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))