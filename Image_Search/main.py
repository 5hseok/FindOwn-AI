from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import models

class ImageSearchRequest(BaseModel):
    target_image_path: str
    top10_image_list: list[str]

app = FastAPI()

# Initialize the models.
similar_model = models.Image_Search_Model("C:\\Users\\DGU_ICE\\FindOwn\\ImageDB\\Logos")
Object_model  = models.Image_Object_Detections()

similar_model.extract_features()
# Test the models.
target_image_path = "C:\\Users\\DGU_ICE\\FindOwn\\ImageDB\\Logos\\ace-cafe-london-logo-vector-download.jpg"  
top10_images = similar_model.search_similar_images(target_image_path)
final_result = Object_model.search_similar_images(target_image_path, top10_images)

print(final_result)  # Print the final result for testing.

@app.post("/search_images")
async def search_images(request: ImageSearchRequest):
    try:
        result = Object_model.search_similar_images(request.target_image_path, request.top10_image_list)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
