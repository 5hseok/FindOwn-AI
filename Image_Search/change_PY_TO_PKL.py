# py 파일과 데이터셋을 PKL 파일로 바꾸기 위한 코드입니다. 
# code = """  ~~~ """ 부분에 py파일의 코드를 넣으시면 됩니다.
# dataset_url = """~~~ """ 부분에 데이터셋을 넣으시면 됩니다. 이미지들의 url을 모은 리스트로 변경 예정. 현재는 로컬 이미지 파일 주소

import pickle

code = """
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

model = EfficientNet.from_pretrained('efficientnet-b7')
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
target_image_path = "C:\\Users\\DGU_ICE\\FindOwn\\ImageDB\\Logos\\manchester-city-logo-vector-download-400x400.jpg"

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
top10_image_list=[]
# 상위 10개 이미지 출력
for i, (image_path, similarity) in enumerate(top_results):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    top10_image_list.append(image_path)
    plt.subplot(n_rows, n_cols, i + 2)
    plt.title(f"Image {i + 1} (similarity: {similarity * 100:.2f}%)")
    plt.imshow(image)
    plt.axis('off')

# plt.tight_layout()
plt.show()
import tensorflow as tf
import tensorflow_hub as hub
import json
import matplotlib.pyplot as plt
import cv2

# Load the model
model = hub.load("https://tfhub.dev/tensorflow/efficientdet/d7/1")

# Load image and preprocess it
image = tf.image.decode_jpeg(tf.io.read_file(target_image_path))
if image.shape[-1] != 3:
    if image.shape[-1] == 1:
        # Convert grayscale to RGB
        image = tf.image.grayscale_to_rgb(image)
    elif image.shape[-1] == 4:
        # Convert RGBA to RGB by discarding the alpha channel
        image = image[..., :3]
        
image = tf.image.resize(image, [224, 224])
image = tf.cast(image, dtype=tf.uint8)

image = image[tf.newaxis, ...] # Add batch dimension and normalize

# Run detection
detections = model(image)

with open('coco-labels-2014_2017.txt','r') as f:
    mscoco_labels = [line.rstrip() for line in f]

# Print detected classes and bounding boxes
check = False
target_image_label = []
for i in range(int(detections['num_detections'])):
    score = detections['detection_scores'][0][i]
    
    # Only consider detections with a confidence score of at least 0.5
    if score >= 0.4:
        class_id = int(detections['detection_classes'][0][i])
        box = detections['detection_boxes'][0][i]

        label = mscoco_labels[class_id]
        check = True
        print(f"Detected class: {label}")
        target_image_label.append(label)
        print("Detection score :", score.numpy())
if check == False:
    print("No object detected")

##############top 10 images object detected##############
final_labels=[]

for i in range(len(top10_image_list)):
    image = tf.image.decode_jpeg(tf.io.read_file(top10_image_list[i]))
    if image.shape[-1] != 3:
        if image.shape[-1] == 1:
            # Convert grayscale to RGB
            image = tf.image.grayscale_to_rgb(image)
        elif image.shape[-1] == 4:
            # Convert RGBA to RGB by discarding the alpha channel
            image = image[..., :3]
            
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, dtype=tf.uint8)

    image = image[tf.newaxis, ...] # Add batch dimension and normalize

    # Run detection
    detections = model(image)
    check = False
    for j in range(int(detections['num_detections'])):
        score = detections['detection_scores'][0][j]
        
        # Only consider detections with a confidence score of at least 0.5
        if score >= 0.4:
            class_id = int(detections['detection_classes'][0][j])
            box = detections['detection_boxes'][0][j]
            label = mscoco_labels[class_id]
            check = True
            print(f"Detected class",top10_image_list[i],":" ,{label})
            final_labels.append(label)
            print("Detection score :", score.numpy())
    
endpoint = 0
final_result_images_index = []
print(final_labels)
for i in range(len(final_labels)):
    if final_labels[i] in target_image_label:
        print(final_labels[i],i)
        final_result_images_index.append(i)

if endpoint == 0:
    print("no images")
    
plt.subplot(1, len(final_result_images_index) + 1, 1)
plt.imshow(cv2.imread(target_image_path))
plt.axis('off')

for i in range(len(final_result_images_index)):
    plt.subplot(1, len(final_result_images_index) + 1, i+2)
    plt.imshow(cv2.imread(top10_image_list[final_result_images_index[i]]))
    plt.axis('off')
plt.show()

"""

import os
import shutil

# 이 예제에서는 dataset을 리스트 형태로 가정합니다.
dataset_url = []
dataset_local = "C:\\Users\\DGU_ICE\\FindOwn\\ImageDB\\Logos"


data_to_pickle = {
    'python_code' : code,
    'dataset' : dataset_url
}

#######################dataset을 구성하는 이미지가 url이 아닌 로컬 주소에서 가져올 경우########################

dataset_in = []
# 이미지 파일의 확장자 목록
image_extensions = ['.jpg', '.jpeg', '.png']

# dataset_local 경로에서 이미지 파일을 찾아서 dataset에 추가
for root, dirs, files in os.walk(dataset_local):
    for file in files:
        _, ext = os.path.splitext(file)
        if ext.lower() in image_extensions:
            image_path = os.path.join(root, file)
            dataset_in.append(image_path)

# 데이터셋에 이미지가 없는 경우 예외 처리
if len(dataset_in) == 0:
    print("No images found in the dataset!")
    
#######################dataset을 구성하는 이미지가 url이 아닌 로컬 주소에서 가져올 경우########################

data_to_pickle = {
    'python_code' : code,
    'dataset' : dataset_in
}

with open('FindOwn_AI.pkl','wb') as file:
    pickle.dump(data_to_pickle,file)

# # 필요할 때 피클 파일에서 코드와 dataset을 불러옵니다.
# with open('your_pickle_file.pkl', 'rb') as file:
#     loaded_data = pickle.load(file)

# loaded_code = loaded_data['python_code']
# loaded_dataset = loaded_data['dataset']