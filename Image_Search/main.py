from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import models
import pickle

class ImageSearchRequest(BaseModel):
    target_image_path: str
    top10_image_list: list[str]

app = FastAPI()

# Load the features from a file.
with open('features.pkl', 'rb') as f:
    features = pickle.load(f)

# Initialize the models.
similar_model = models.Image_Search_Model("C:\\Users\\DGU_ICE\\FindOwn\\ImageDB\\Logos", pre_extracted_features='features.pkl')
Object_model  = models.Image_Object_Detections()

@app.post("/search_images")
async def search_images(request: ImageSearchRequest):
    try:
        result = Object_model.search_similar_images(request.target_image_path, request.top10_image_list)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
