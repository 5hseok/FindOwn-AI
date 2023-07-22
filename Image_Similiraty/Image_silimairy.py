import os
import time
import math
import random
import numpy as np
import json
import struct
import glob
import csv
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow as tf
from sklearn.preprocessing import normalize
import faiss
import cv2
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

#이미지 파일 전처리
def preprocess(img_path, input_shape):
    if not os.path.exists(img_path) or not os.path.isfile(img_path):
        raise ValueError(f"이미지 파일이 없거나 경로가 잘못되었습니다: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = tf.convert_to_tensor(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, input_shape[:2])
    img = preprocess_input(img)
    return img

#파일 이름 찾기
def find_file_name(idx):
    if idx == -1:
        return 'None'
    with open('fnames.txt', 'r') as f:
        names = f.readlines()
    return names[idx].strip('\n').strip('\t')

#이미지 출력
def show_images(filenames,similiraty):
    if not filenames:
        print("Error: No images to display.")
        return
    fig, axs = plt.subplots(1, len(filenames), figsize=(3 * len(filenames), 3),num="Image Similarity Retrieval")
    i=0
    if len(filenames)==1:
        axs=[axs]
    for ax, filename in zip(axs, filenames):
        print(filename,"{:.4f}".format(similiraty[i]*100)+"%")
        img = Image.open(filename)
        ax.imshow(img)
        ax.axis('off')
        if i==0:
            ax.set_title(f"Upload_image: {similiraty[i]*100:.4f}%")
        else:
            ax.set_title(f"{similiraty[i]*100:.4f}%")
        i+=1
    plt.tight_layout()
    plt.show()
    
model=None      #전역변수로 모델 정의

#csv 파일에 결과 작성
def save_results_to_csv(user_file, image_filenames, similarities):
    with open('result.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['File_name', 'Similarity', 'Prejudice']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        user_file_letter = os.path.basename(user_file).split('.')[0][0].lower()

        # 사용자 이미지를 2행에 추가합니다. Prejudice 값을 설정합니다.
        basename = os.path.basename(user_file)
        user_similarity_percentage = format(max(similarities) * 100, '.9f')

        user_file_abs = os.path.abspath(user_file)
        user_image_index = next(i for i, filename in enumerate(image_filenames) if os.path.abspath(filename) == user_file_abs)

        #이미지 소송 및 분쟁과 관련한 이미지가 관련도가 가장 높다면 prejudice로 판단
        if user_image_index < len(image_filenames) - 1:
            next_file_letter = os.path.basename(image_filenames[user_image_index + 1]).split('.')[0][0].lower()
            prejudice_value = 'prejudice' if user_file_letter == next_file_letter else 'None'
        else:
            prejudice_value = 'None'

        writer.writerow({'File_name': basename + " (User img)", 'Similarity': user_similarity_percentage, 'Prejudice': prejudice_value})
        
        for i, (filename, similarity) in enumerate(zip(image_filenames, similarities)):
            if os.path.abspath(filename) != os.path.abspath(user_file):

                # 이미지의 기본 이름만 추출합니다.
                basename = os.path.basename(filename)

                # 유사도에 100을 곱하고 소수점 아래 값의 길이를 9자리로 제한합니다.
                similarity_percentage = "{:.9f}".format(similarity * 100)+'%'

                writer.writerow({'File_name': basename, 'Similarity': similarity_percentage, 'Prejudice': ''})


def main():
    global model
    
    root_dir = "C:\\Users\\DGU_ICE\\ImageCopy\\ImageSeparation\\Image"
    #이미지 더미들 넣는 파일, User가 입력한 이미지와 여기 dir에 있는 이미지들을 비교해서 유사도를 검색한다.
    
    #모델 구현
    base = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    model = tf.keras.Model(inputs=base.input, outputs=x)
    
    base_dim = 1280
    
    #이미지 파일 불러오기
    images = []
    for(dirpath, dirnames, filenames) in os.walk(root_dir):
        for filename in filenames:
            if ".png" in filename or ".jpg" in filename:
                images.append(dirpath + "/" + filename)
    
    #벡터화 시키기
    d = 0
    qb = np.zeros((len(images), base_dim), dtype='float32')
    with open('fvecs.bin', 'wb') as fvecs_f:
        for img_path in images:
            processed_img = preprocess(img_path, (224, 224, 3))
            feature = model.predict(processed_img.numpy().reshape([1, 224, 224, 3])).reshape(base_dim)
            qb[d] = feature
            fvecs_f.write(struct.pack('f' * base_dim, *feature))
            d += 1

    #fnames.txt에 이미지 경로 파일 작성
    with open('fnames.txt', 'w') as f:
        f.write('\n'.join(images))
        
    #faiss 사용
    index = faiss.IndexHNSWFlat(1280, 16, faiss.METRIC_INNER_PRODUCT)
    index.verbose = True
    index.add(normalize(qb))
    faiss.write_index(index, "fvecs.bin.hnsw.index")


if __name__ == '__main__':
    main()
    #이미지 경로 입력 받기
    user_file = input("원하시는 이미지의 경로를 입력하세요: ")
    
    #이미지 전처리
    user_vector = preprocess(user_file, (224, 224, 3))
    user_vector = np.expand_dims(user_vector, axis=0)

    dim = 1280
    fvec_file = 'fvecs.bin'
    index_type = 'hnsw'
    index_file = f'{fvec_file}.{index_type}.index'
    if os.path.getsize(fvec_file) > 0:
        fvecs = np.memmap(fvec_file, dtype='float32', mode='r').view('float32').reshape(-1, dim)
    else:
        print("Error: fvecs.bin 파일이 비어 있습니다. 이미지가 있는지 확인하세요.")
        exit()
    base_dim=1280
    user_vector = preprocess(user_file, (224, 224, 3))
    user_vector = np.expand_dims(user_vector, axis=0)
    index = faiss.read_index(index_file)
    user_feature = model.predict(user_vector.reshape([1, 224, 224, 3])) 

    normalized_user_feature = normalize(user_feature.reshape(1,-1))
    
    k = fvecs.shape[0]  #검사할 이미지 개수, 여기서는 총 이미지 파일의 개수와 동일

    dists, idxs = index.search(normalized_user_feature, k)
    
    sim_files = []      #유사도 조건 만족하는 파일
    sim_data=[]         #유사도 조건 만족하는 파일의 유사도 값
    all_img=[]          #모든 파일
    all_data=[]         #모든 파일의 유사도 값
    for i, idx in enumerate(idxs[0]):
        all_img.append(find_file_name(idx))
        all_data.append(dists[0][i])
        if dists[0][i] >= 0.996 and idx != -1:   
            sim_files.append(find_file_name(idx))
            sim_data.append(dists[0][i])
    show_images(sim_files,sim_data)
    save_results_to_csv(user_file,all_img,all_data)