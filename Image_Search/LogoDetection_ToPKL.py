import os
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from PIL import Image
from fastai.vision.all import *
from ipywidgets import FileUpload, widget
# Load the EfficientNet model
model = tf.saved_model.load('C:\\Users\\DGU_ICE\\FindOwn\\Image_Search\\EfficientNet')
classes = ["Fake", "Genuine"]

path = Path('C:/Users/DGU_ICE/FindOwn/ImageDB/Logos')   # 데이터셋 경로

# 상표 침해에 해당하는 폴더를 지정하는 함수
def is_infringement(x):
    x = str(x)
    return '/infringing/' in x

# EfficientNet 모델을 사용하여 이미지를 분류
def classify_image(path): 
    img = Image.open(path).convert('RGB')
    img = img.resize((300, 300 * img.size[1] // img.size[0]), Image.ANTIALIAS)
    inp_numpy = np.array(img)[None]
    inp = tf.constant(inp_numpy, dtype='float32')
    class_scores = model(inp)[0].numpy()
    return classes[class_scores.argmax()]

file_images = [f for f in os.listdir(path) if f.endswith('.jpg')]
def main():
    dls = ImageDataLoaders.from_path_func(
        path,
        get_image_files(path),
        valid_pct=0.2,
        seed=42,
        label_func=is_infringement,
        item_tfms=Resize(224)
    )
    dls.show_batch(max_n=10)
    
    from IPython.display import display

    test_img_path = uploader.data[0]
    result = classify_image(test_img_path)
    print(f"Is this a trademark infringement?: {result}.")

    # 파일명과 결과를 피클 파일로 저장
    filename = uploader.metadata[0]['name']
    result_dict = {
        'filename': filename,
        'infringement': bool(result == "Fake"),
    }

    with open('results.pkl', 'wb') as f:
        pickle.dump(result_dict, f)

if __name__ == '__main__':
    main()
from IPython.display import display, clear_output
import io
from PIL.Image import open as open_image

uploader = file_images
n_uploaded_imgs = len(uploader)
if n_uploaded_imgs > 0:
    for img_data in uploader.data:
        clear_output()
        display(open_image(io.BytesIO(img_data)))
