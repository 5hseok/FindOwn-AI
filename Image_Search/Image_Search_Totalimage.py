from efficientnet_pytorch import EfficientNet

#전체 이미지 DB에 있는 이미지들에서 유사도를 전부 측정한다.
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

def is_image_file(filename):
    # 파일 확장자 검사
    VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    _, ext = os.path.splitext(filename)
    return ext.lower() in VALID_EXTENSIONS

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
    

root_dir = "C:\\Users\\DGU_ICE\\FindOwn\\ImageDB\\Logos"
#자신의 로컬에 있는 Image 파일 주소로 설정할 것.

import os
import re
from collections import defaultdict

# 타겟 이미지 경로
target_image_path = "C:\\Users\\DGU_ICE\\FindOwn\\ImageDB\\Logos\\uefa-champions-league-eps-vector-logo-400x400.png"

# 디렉토리에서 이미지 파일들 찾기
image_files = []
for (dirpath, dirnames, filenames) in os.walk(root_dir):
    for filename in filenames:
        if is_image_file(filename):
            image_files.append(os.path.join(dirpath, filename))

# 각 이미지와 타겟 이미지 간에 코사인 유사도 저장할 리스트 초기화
similarities = []

import cv2
import urllib.request
from matplotlib import pyplot as plt
import numpy as np
from concurrent.futures import ThreadPoolExecutor


# 타겟 이미지 특징 추출
def process_image(image_path):
    source_embedding = predict(image_path)
    if source_embedding is None or target_embedding is None:
        return image_path, None
    similarity = cos_sim(source_embedding,target_embedding)
    return image_path, similarity

import time
start_time=time.time()
target_embedding = predict(target_image_path)

# 각 이미지와 타겟 이미지의 유사도 계산
with ThreadPoolExecutor() as executor:
    results = list(executor.map(process_image, image_files))

# 유효한 결과만 저장하고 출력합니다.
top_results = []
for image_path, similarity in results:
    if similarity is not None:
        top_results.append((image_path, similarity))
    
top_results=sorted(top_results, key=lambda x: x[1],reverse=True)[:10]
elapsed_time = time.time() - start_time
print(f"병렬 처리 시간: {elapsed_time}초")
for image_path, similarity in top_results:
    print("Similarity between", image_path, "and target image:",similarity)


# # 유사도의 평균을 계산합니다
# average_similarity = sum(similarities) / len(similarities)
# print("Average similarity: {:.4f}%".format(round(average_similarity * 100,4)))

# import csv
# # CSV 파일을 쓰기 모드로 연다.
# with open('image_similarity_Top10image.csv', mode='w', newline='') as csv_file:
#     csv_writer = csv.writer(csv_file)
#     csv_writer.writerow(['Image', 'Target Image', 'Similarity'])

#     for image_path, similarity in top_results:
#         # 유사도 값을 CSV 파일에 기록한다.
#         csv_writer.writerow([image_path, target_image_path, similarity * 100])

# 생성할 subplot의 행과 열 계산
n_rows = 3
n_cols = 4

# 하나의 figure에서 타겟 이미지와 top-10 이미지 출력
plt.figure(figsize=(15, 12))

# 타겟 이미지 출력
image = cv2.imread(target_image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.subplot(n_rows, n_cols, 1)
plt.title("Target Image")
plt.imshow(image)
plt.axis('off')

# 상위 10개 이미지 출력
for i, (image_path, similarity) in enumerate(top_results):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.subplot(n_rows, n_cols, i + 2)
    plt.title(f"Image {i + 1} (similarity: {similarity * 100:.2f}%)")
    plt.imshow(image)
    plt.axis('off')

# plt.tight_layout()
plt.show()

################## 타겟 이미지와 가장 유사한 10개의 이미지들로 객체 인식 돌리기 ####3###################
# import tensorflow as tf
# import tensorflow_hub as hub
# from PIL import ImageDraw
# import json
# import requests

# def get_class_names():
#     """Returns the class names dictionary for OpenImages V4 dataset."""
#     class_names_url = "https://storage.googleapis.com/openimages/v6/oidv6-class-descriptions.csv"
#     class_id_to_name = {}
#     response = requests.get(class_names_url)
#     response_text = response.text.split("\r\n")[:-1]

#     for row in response_text:
#         id, name = row.split(',', 1)
#         class_id_to_name[int(id)] = name

#     return class_id_to_name

# class_id_to_name = get_class_names()


# model=tf.saved_model.load("saved_resnetmodel_dir\\")
# print(model.signatures)
# def detect_objects(image_path, objects_to_detect=None):

#     # Load the image
#     original_image = Image.open(image_path)
#     byte_image = tf.keras.preprocessing.image.img_to_array(original_image)
#     byte_image = tf.expand_dims(byte_image, 0)

#     # Object detection
#     detection_output = model(input_tensor = tf.constant(byte_image))
#     num_detections = int(detection_output["num_detections"])
#     object_class_ids = detection_output["detection_classes"][0].numpy().astype(int)
#     object_boxes = detection_output["detection_boxes"][0].numpy()
#     object_scores = detection_output["detection_scores"][0].numpy()

#     # Draw the detected objects
#     draw = ImageDraw.Draw(original_image)
#     detected_objects = []

#     # Draw the detected objects
#     for i in range(num_detections):
#         # Check if the detection is in the provided list of objects_to_detect
#         if objects_to_detect is None or object_class_ids[i] in objects_to_detect:
#             object_name = class_id_to_name[object_class_ids[i]]
#             object_score = object_scores[i]
#             detected_objects.append((object_name, object_score))

#             y1, x1, y2, x2 = object_boxes[i].tolist()
#             width, height = original_image.size
#             draw.rectangle(((x1 * width, y1 * height), (x2 * width, y2 * height)), outline="red", width=3)

#     return detected_objects

# # 타겟 이미지의 객체 감지 및 출력
# target_image_detected_objects = detect_objects(target_image_path)
# print("타겟 이미지의 객체 class:")
# for obj, score in target_image_detected_objects:
#     print(f"- {obj}")

# print("\n")

# # 상위 10개 이미지의 객체 감지 및 출력
# print("Top-10 유사한 이미지의 객체 class:")
# for index, (image_path, similarity) in enumerate(top_results):
#     print(f"{index + 1}. 유사도: {similarity:.4f}")

#     detected_objects = detect_objects(image_path)
#     for obj, score in detected_objects:
#         print(f"- {obj}")

#     print("\n")
