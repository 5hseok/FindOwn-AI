from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models.feature_extraction import create_feature_extractor

# weights = EfficientNet_B0_Weights.DEFAULT
model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
model = create_feature_extractor(model, return_nodes={'features': 'features'})
# model = create_feature_extractor(model,return_nodes={'avgpool' : 'avgpool'})
model.eval()

import requests
import torchvision.transforms as T
from PIL import Image

def image_resize(image_url):
    image = Image.open(requests.get(image_url, stream=True).raw)
    rgb_image = image.convert('RGB')
    preprocess = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor()]
    )
    return preprocess(rgb_image).unsqueeze(0)


from numpy import dot
from numpy.linalg import norm
import torch


def cos_sim(A, B):
    return dot(A, B) / (norm(A) * norm(B))


def predict(image_url):
    resized_image = image_resize(image_url)
    predicted_result = model(resized_image)
    image_feature = torch.flatten(predicted_result['features'])
    # image_feature = torch.flatten(predicted_result['avgpool'])
    return image_feature.detach().numpy()

# 이미지 URL 쌍 리스트를 정의합니다
image_pairs = [
    ("https://previews.123rf.com/images/rclassenlayouts/rclassenlayouts1209/rclassenlayouts120900196/15362188-3d-%EC%BB%A4%ED%94%BC-%EC%B9%B4%ED%8E%98-%EC%BD%A9-%EA%B8%B0%EC%97%85%EC%9D%98-%EB%94%94%EC%9E%90%EC%9D%B8-%EC%95%84%EC%9D%B4%EC%BD%98-%EB%A1%9C%EA%B3%A0-%EC%83%81%ED%91%9C.jpg","https://www.shutterstock.com/image-vector/coffee-cup-icon-600w-223212751.jpg"),
    ("https://drive.google.com/uc?export=download&id=15SFZlK07iWIDUn1NRU7njl_7pU-AlLY1","https://drive.google.com/uc?export=download&id=119biFlAhymgClAAxGTzEQyuhgVtWiXNn"),
    ("https://previews.123rf.com/images/rclassenlayouts/rclassenlayouts1209/rclassenlayouts120900196/15362188-3d-%EC%BB%A4%ED%94%BC-%EC%B9%B4%ED%8E%98-%EC%BD%A9-%EA%B8%B0%EC%97%85%EC%9D%98-%EB%94%94%EC%9E%90%EC%9D%B8-%EC%95%84%EC%9D%B4%EC%BD%98-%EB%A1%9C%EA%B3%A0-%EC%83%81%ED%91%9C.jpg","https://previews.123rf.com/images/rclassenlayouts/rclassenlayouts1209/rclassenlayouts120900196/15362188-3d-%EC%BB%A4%ED%94%BC-%EC%B9%B4%ED%8E%98-%EC%BD%A9-%EA%B8%B0%EC%97%85%EC%9D%98-%EB%94%94%EC%9E%90%EC%9D%B8-%EC%95%84%EC%9D%B4%EC%BD%98-%EB%A1%9C%EA%B3%A0-%EC%83%81%ED%91%9C.jpg"),

    #(sourece_url, target_url)
    # 다른 이미지 URL 쌍을 여기에 추가하세요
]

# 각 이미지 쌍의 코사인 유사도를 저장할 리스트를 초기화합니다
similarities = []
import cv2
import urllib.request
from matplotlib import pyplot as plt
import numpy as np

for pair in image_pairs:
    source_embedding = predict(pair[0])
    target_embedding = predict(pair[1])
    similarity = cos_sim(source_embedding, target_embedding)
    similarities.append(similarity)

    plt.figure(figsize=(10, 5)) # 적절한 크기를 지정할 수 있습니다.

    # 첫 번째 이미지와 유사도 값을 출력합니다
    with urllib.request.urlopen(pair[0]) as url:
        image_data = url.read()
    image_array = np.asarray(bytearray(image_data), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.subplot(1, 2, 1) # 1행 2열의 subplot에서 첫 번째 위치에 이미지를 넣습니다
    plt.title("Image 1")
    plt.imshow(image)
    plt.axis('off')

    # 두 번째 이미지와 유사도 값을 출력합니다
    with urllib.request.urlopen(pair[1]) as url:
        image_data = url.read()
    image_array = np.asarray(bytearray(image_data), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.subplot(1, 2, 2) # 1행 2열의 subplot에서 두 번째 위치에 이미지를 넣습니다
    plt.title(f"Image 2 (Similarity: {similarity * 100:.2f}%)")
    plt.imshow(image)
    plt.axis('off')

    # 이미지와 함께 유사도 값을 출력합니다
    plt.show()


# 유사도의 평균을 계산합니다
average_similarity = sum(similarities) / len(similarities)
print("Average similarity: {:.4f}%".format(round(average_similarity * 100,4)))

import csv
# CSV 파일을 쓰기 모드로 연다.
with open('image_pairs_similarity.csv', mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Source Image', 'Target Image', 'Similarity'])

    for pair in image_pairs:
        source_embedding = predict(pair[0])
        target_embedding = predict(pair[1])
        similarity = cos_sim(source_embedding, target_embedding)
        similarities.append(similarity)

        # 유사도 값을 CSV 파일에 기록한다.
        csv_writer.writerow([pair[0], pair[1], similarity])


