from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import models
import pickle

class ImageSearchRequest(BaseModel):
    target_image_path: str
    top10_image_list: list[str]

app = FastAPI()
# Initialize the models.
target_image_path = "C:\\Users\\DGU_ICE\\FindOwn\\ImageDB\\Logos\\000001_07cbc019bcf34352bf73e821ae50340a.jpg"
similar_model = models.Image_Search_Model(pre_extracted_features='features_logo.pkl')
top_10_image_list = similar_model.search_similar_images(target_image_path)
print(top_10_image_list)
Object_model  = models.Image_Object_Detections()
result = Object_model.search_similar_images(target_image_path,top_10_image_list)
print(result)

@app.post("/search_images")
async def search_images(request: ImageSearchRequest):
    try:
        result = Object_model.search_similar_images(request.target_image_path, request.top10_image_list)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))