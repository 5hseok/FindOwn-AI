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
import matplotlib.pyplot as plt


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


def find_file_name(idx):
    if idx == -1:
        return 'None'
    with open('fnames.txt', 'r') as f:
        names = f.readlines()
    return names[idx].strip('\n').strip('\t')


def show_images(filenames,similiraty):
    if not filenames:
        print("Error: No images to display.")
        return
    fig, axs = plt.subplots(1, len(filenames), figsize=(3 * len(filenames), 3))
    i=0
    for ax, filename in zip(axs, filenames):
        print(filename,str(similiraty[i]*100)+"%")
        img = Image.open(filename)
        ax.imshow(img)
        ax.axis('off')
        i+=1
    plt.show()
model=None
def main():
    global model
    root_dir = "C:\\Users\\DGU_ICE\\ImageCopy\\ImageSeparation\\Image"
    base = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    model = tf.keras.Model(inputs=base.input, outputs=x)
    
    base_dim = 1280
    
    images = []
    for(dirpath, dirnames, filenames) in os.walk(root_dir):
        for filename in filenames:
            if ".png" in filename or ".jpg" in filename:
                images.append(dirpath + "/" + filename)
    
    d = 0
    qb = np.zeros((len(images), base_dim), dtype='float32')
    with open('fvecs.bin', 'wb') as fvecs_f:
        for img_path in images:
            processed_img = preprocess(img_path, (224, 224, 3))
            feature = model.predict(processed_img.numpy().reshape([1, 224, 224, 3])).reshape(base_dim)
            qb[d] = feature
            fvecs_f.write(struct.pack('f' * base_dim, *feature))
            d += 1


    with open('fnames.txt', 'w') as f:
        f.write('\n'.join(images))
            
    index = faiss.IndexHNSWFlat(1280, 16, faiss.METRIC_INNER_PRODUCT)
    index.verbose = True
    index.add(normalize(qb))
    faiss.write_index(index, "fvecs.bin.hnsw.index")


if __name__ == '__main__':
    main()
    user_file = input("원하시는 이미지의 경로를 입력하세요: ")
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
    user_feature = model.predict(user_vector.reshape([1, 224, 224, 3])) #.reshape(-1, base_dim)

    normalized_user_feature = normalize(user_feature.reshape(1,-1))
    k = 10
    dists, idxs = index.search(normalized_user_feature, k)
    sim_files = []
    sim_data=[]
    for i, idx in enumerate(idxs[0]):
        print(dists[0][i],idx)
        if dists[0][i] >= 0.99968 and idx != -1:        #출력 기준
            sim_files.append(find_file_name(idx))
            sim_data.append(dists[0][i])
    show_images(sim_files,sim_data)
