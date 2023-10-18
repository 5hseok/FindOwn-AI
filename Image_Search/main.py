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
target_image_path = "C:\\Users\\DGU_ICE\\FindOwn\\ImageDB\\Logos"
similar_model = models.Image_Search_Model(target_image_path, pre_extracted_features='features.pkl')
top_10_image_list = similar_model.search_similar_images(target_image_path)
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


# # Testclint 사용한 테스트 코드
# from fastapi.testclient import TestClient

# # Initialize the test client
# client = TestClient(app)

# # Define the test data
# test_data = {
#     "target_image_path": "path/to/your/target/image.jpg",
#     "top10_image_list": ["path/to/image1.jpg", "path/to/image2.jpg", ...]  # replace with actual image paths
# }

# # Send a POST request to the /search_images endpoint with the test data
# response = client.post("/search_images", json=test_data)

# # Print the response status code and body
# print(response.status_code)
# print(response.json())
