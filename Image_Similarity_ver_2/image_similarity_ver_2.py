# from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
# from torchvision.models.feature_extraction import create_feature_extractor
from efficientnet_pytorch import EfficientNet

    ####################################################
    #   이 py 파일은 침해도 기준을 정하기 위한 모델로    #
    #   유사도 검색을 진행하지만, 한 쌍의 이미지들에     #
    #   대해서만 유사도 검색을 진행하여 평균적인 유사도  #
    #   결과를 도출한다.                               #
    ###################################################

def create_feature_extractor(model, return_nodes=None):
    if return_nodes is None:
        return_nodes = {'avgpool': 'avgpool'}

    return_nodes_output = {}
    for name, module in model.named_modules():
        if name in return_nodes:
            return_nodes_output[name] = module

    return return_nodes_output

model = EfficientNet.from_pretrained('efficientnet-b0')
model_features = create_feature_extractor(model)
model.eval()    

import requests
import torchvision.transforms as T
from PIL import Image

def image_resize(image_url):                #이미지 url로 받아올 때 사용
    image = Image.open(requests.get(image_url, stream=True).raw)
    rgb_image = image.convert('RGB')
    preprocess = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor()]
    )
    return preprocess(rgb_image).unsqueeze(0)


from torchvision.transforms import Compose, Resize, ToTensor, Normalize

def image_resize_local(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error: Unable to open image. {e}")
        return None

    # 이미지 전처리: 크기 조정, 텐서 변환, 정규화
    preprocess = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    return preprocess(image)

from numpy import dot
from numpy.linalg import norm
import torch


def cos_sim(A, B):
    return dot(A, B) / (norm(A) * norm(B))


def predict(image_path):
    resized_image = image_resize_local(image_path)
    if resized_image is None:
        return None
    model.eval()
    with torch.no_grad():
        image_transformed=resized_image.unsqueeze(0)
        predicted_result=model(image_transformed)
        image_feature=torch.flatten(predicted_result)
    return image_feature.detach().numpy()
    

# 이미지 URL 쌍 리스트를 정의합니다 - 기존 버전입니다.
image_pairs = [
    ("https://previews.123rf.com/images/rclassenlayouts/rclassenlayouts1209/rclassenlayouts120900196/15362188-3d-%EC%BB%A4%ED%94%BC-%EC%B9%B4%ED%8E%98-%EC%BD%A9-%EA%B8%B0%EC%97%85%EC%9D%98-%EB%94%94%EC%9E%90%EC%9D%B8-%EC%95%84%EC%9D%B4%EC%BD%98-%EB%A1%9C%EA%B3%A0-%EC%83%81%ED%91%9C.jpg","https://www.shutterstock.com/image-vector/coffee-cup-icon-600w-223212751.jpg"),
    ("https://drive.google.com/uc?export=download&id=15SFZlK07iWIDUn1NRU7njl_7pU-AlLY1","https://drive.google.com/uc?export=download&id=119biFlAhymgClAAxGTzEQyuhgVtWiXNn"),
    ("https://previews.123rf.com/images/rclassenlayouts/rclassenlayouts1209/rclassenlayouts120900196/15362188-3d-%EC%BB%A4%ED%94%BC-%EC%B9%B4%ED%8E%98-%EC%BD%A9-%EA%B8%B0%EC%97%85%EC%9D%98-%EB%94%94%EC%9E%90%EC%9D%B8-%EC%95%84%EC%9D%B4%EC%BD%98-%EB%A1%9C%EA%B3%A0-%EC%83%81%ED%91%9C.jpg","https://previews.123rf.com/images/rclassenlayouts/rclassenlayouts1209/rclassenlayouts120900196/15362188-3d-%EC%BB%A4%ED%94%BC-%EC%B9%B4%ED%8E%98-%EC%BD%A9-%EA%B8%B0%EC%97%85%EC%9D%98-%EB%94%94%EC%9E%90%EC%9D%B8-%EC%95%84%EC%9D%B4%EC%BD%98-%EB%A1%9C%EA%B3%A0-%EC%83%81%ED%91%9C.jpg"),
    #(sourece_url, target_url)
    # 다른 이미지 URL 쌍을 여기에 추가하세요
]

root_dir = "C:\\Users\\DGU_ICE\\FindOwn\\Image"
#한 쌍의 이미지 파일을 넣는 이미지 파일 dir 주소. 여기서 소송 및 분쟁에 관련된 이미지 쌍을 가져온다.
#자신의 로컬에 있는 Image 파일 주소로 설정할 것.

import os
import re
from collections import defaultdict

image_pairs_local = []
image_file_pattern = re.compile(r'^page(\d+)_image(\d+)\.(?:png|jpg)$')
image_dict = defaultdict(lambda: {})

for (dirpath, dirnames, filenames) in os.walk(root_dir):
    for filename in filenames:
        match = image_file_pattern.match(filename)
        if match:
            page_num = match.group(1)
            image_num = int(match.group(2))
            if image_num in [1, 2]:
                image_dict[page_num][image_num] = os.path.join(dirpath, filename)
                
for page_num, images in image_dict.items():
    if 1 in images and 2 in images:
        pair = (images[1], images[2])
        image_pairs_local.append(pair)




# 각 이미지 쌍의 코사인 유사도를 저장할 리스트를 초기화합니다
similarities = []
import cv2
import urllib.request
from matplotlib import pyplot as plt
import numpy as np

# for pair in image_pairs:      #url 버전 pair
for pair in image_pairs_local:  #local image 버전 pair
    source_embedding = predict(pair[0])
    target_embedding = predict(pair[1])
    similarity = cos_sim(source_embedding, target_embedding)
    similarities.append(similarity)
    print(similarity)

    if similarity > 0.9 :
        plt.figure(figsize=(10, 5)) # 적절한 크기를 지정할 수 있습니다.

        # 첫 번째 이미지와 유사도 값을 출력합니다
        image_path = pair[0]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.subplot(1, 2, 1) # 1행 2열의 subplot에서 첫 번째 위치에 이미지를 넣습니다
        plt.title("Image 1")
        plt.imshow(image)
        plt.axis('off')

        # 두 번째 이미지와 유사도 값을 출력합니다
        image_path = pair[1]
        image = cv2.imread(image_path)
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

    for pair in image_pairs_local:
        source_embedding = predict(pair[0])
        target_embedding = predict(pair[1])
        similarity = cos_sim(source_embedding, target_embedding)
        similarities.append(similarity)

        # 유사도 값을 CSV 파일에 기록한다.
        csv_writer.writerow([pair[0], pair[1], similarity])