from efficientnet_pytorch import EfficientNet

def create_feature_extractor(model, return_nodes=None):
    if return_nodes is None:
        return_nodes = {'avgpool': 'avgpool'}

    return_nodes_output = {}
    for name, module in model.named_modules():
        if name in return_nodes:
            return_nodes_output[name] = module

    return return_nodes_output

model = EfficientNet.from_pretrained('efficientnet-b0')
model_features = create_feature_extractor(model,return_nodes={'avgpool':'avgpool'})
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
    

root_dir = "C:\\Users\\sam\\Desktop\\user\\Pictures\\Pattern_Image"
#자신의 로컬에 있는 Image 파일 주소로 설정할 것.

import os
import re
from collections import defaultdict

# 타겟 이미지 경로
target_image_path = "C:\\Users\\sam\\Desktop\\user\\Pictures\\Pattern_Image\\1.jpg"

# 디렉토리에서 이미지 파일들 찾기
image_files = []
for (dirpath, dirnames, filenames) in os.walk(root_dir):
    for filename in filenames:
        image_files.append(os.path.join(dirpath, filename))

# 각 이미지와 타겟 이미지 간에 코사인 유사도 저장할 리스트 초기화
similarities = []

import cv2
import urllib.request
from matplotlib import pyplot as plt
import numpy as np

# 타겟 이미지 특징 추출
target_embedding = predict(target_image_path)

# 각 이미지와 타겟 이미지의 유사도 계산
for image_path in image_files:
    source_embedding = predict(image_path)
    if source_embedding is None:
        print(f"Skipping {image_path} due to prediction error.")
        continue
    if target_embedding is None:
        print(f"Skipping {image_path} due to prediction error.")
        continue
    similarity = cos_sim(source_embedding, target_embedding)
    similarities.append(similarity)

    print("Similarity between", image_path, "and target image:", similarity)
    
    # if similarity > 0.8:
    #     # 이미지와 유사도 출력
    #     plt.figure(figsize=(10, 5)) # 적절한 크기를 지정

    #     # 첫 번째 이미지와 유사도 값 출력
    #     image = cv2.imread(image_path)
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #     plt.subplot(1, 2, 1) # 1행 2열의 subplot에서 첫 번째 위치에 이미지를 넣습니다
    #     plt.title("Image 1")
    #     plt.imshow(image)
    #     plt.axis('off')

    #     # 두 번째 이미지와 유사도 값을 출력합니다
    #     image = cv2.imread(target_image_path)
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
    #     plt.subplot(1, 2, 2) # 1행 2열의 subplot에서 두 번째 위치에 이미지를 넣습니다
    #     plt.title(f"Target Image (Similarity: {similarity * 100:.2f}%)")
    #     plt.imshow(image)
    #     plt.axis('off')

    #     # 이미지와 함께 유사도 값을 출력합니다
    #     plt.show()

# 유사도의 평균을 계산합니다
average_similarity = sum(similarities) / len(similarities)
print("Average similarity: {:.4f}%".format(round(average_similarity * 100,4)))

import csv
# CSV 파일을 쓰기 모드로 연다.
with open('image_similarity_NN.csv', mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Image', 'Target Image', 'Similarity'])

    for image_path in image_files:
        source_embedding = predict(image_path)
        if source_embedding is None:
            print(f"Skipping {image_path} due to prediction error.")
            continue
        similarity = cos_sim(source_embedding, target_embedding)
        similarities.append(similarity)

        # 유사도 값을 CSV 파일에 기록한다.
        csv_writer.writerow([image_path, target_image_path, similarity*100])