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
    
    import requests
    from PIL import Image
    import io
    
    test_img_path = ["https://w7.pngwing.com/pngs/869/485/png-transparent-google-logo-computer-icons-google-text-logo-google-logo-thumbnail.png"]
    if len(test_img_path) > 0:
        for image_url in test_img_path:
            response = requests.get(image_url)

            # Check if the request was successful
            if response.status_code == 200:
                image_data = response.content

                try:
                    # Try to open the image from the data
                    img = Image.open(io.BytesIO(image_data)).convert('RGB')

                    filename = image_url.split("/")[-1]

                    result = classify_image(img)
                    
                    print(f"Is this a trademark infringement?: {result}.")

                    result_dict = {
                        'filename': filename,
                        'infringement': bool(result == "Fake"),
                    }

                    with open('results.pkl', 'wb') as f:
                        pickle.dump(result_dict, f)
                    print(1)
                except IOError:
                    print(f"Cannot identify image file from URL: {image_url}")

if __name__ == '__main__':
    main()
    