import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torch import nn
import cv2
import os
import pickle
from PIL import Image
import numpy as np

class CNNModel:
    def __init__(self):
        self.model = resnet50(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def extract_feature(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        image = image.unsqueeze(0)

        if torch.cuda.is_available():
            image = image.cuda()

        feature = self.model(image)
        return feature.cpu().data.numpy().flatten()  # 1차원 배열로 변환

    def extract_features_from_dir(self, root_dir, save_path):
        features = {}
        for filename in os.listdir(root_dir):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(root_dir, filename)
                feature = self.extract_feature(image_path)
                features[filename] = feature

        with open(save_path, 'wb') as f:
            pickle.dump(features, f)

    def cosine_similarity(self, a, b):  # self 매개변수 추가
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def compare_features(self, target_image_path, features_path):
        target_feature = self.extract_feature(target_image_path)

        with open(features_path, 'rb') as f:
            features = pickle.load(f)

        similarities = {}
        for key, value in features.items():
            if value.ndim > 1:  # value가 1차원이 아니라면
                value = value.flatten()  # 1차원으로 변환
            similarities[key] = self.cosine_similarity(target_feature, value)
        similarities = sorted(similarities.items(),key=lambda x: x[1], reverse=True)
        return similarities

import cv2
import numpy as np
from sklearn.cluster import KMeans

class LogoColorSimilarityModel:
    def __init__(self, num_bins=30, resize_shape=(256, 256), num_colors=3):
        self.num_bins = num_bins
        self.resize_shape = resize_shape
        self.num_colors = num_colors

    def apply_grabcut(self, img):
        mask = np.zeros(img.shape[:2], np.uint8)
        bgdModel = np.zeros((1,65), np.float64)
        fgdModel = np.zeros((1,65), np.float64)
        rect = (50, 50, 450, 290)
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        img = img*mask2[:,:,np.newaxis]
        return img

    def calculate_histogram(self, img_path):
        try:
            img = Image.open(img_path)  # PIL 라이브러리를 이용하여 이미지 불러오기
            img = np.array(img)  # 이미지를 numpy 배열로 변환
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # RGB에서 BGR로 변환

        except IOError:
            print(f"Cannot read image at {img_path}")
            return None        
        img = self.apply_grabcut(img)
        img_cv = cv2.resize(img, self.resize_shape)
        hsv_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv_img], [0, 1, 2], None, [self.num_bins]*3, [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        hist = hist.flatten().astype('float32')
        if len(hist) < self.num_bins**3:
            hist = np.concatenate([hist, np.zeros(self.num_bins**3 - len(hist))])
        return hist

    # 나머지 코드는 이전과 동일합니다...
    @staticmethod
    def calculate_histogram_cross_entropy(hist1, hist2):
        hist1 = hist1 / np.sum(hist1)
        hist2 = hist2 / np.sum(hist2)
        hist1 = np.clip(hist1, a_min=1e-10, a_max=1.0)
        hist2 = np.clip(hist2, a_min=1e-10, a_max=1.0)
        cross_entropy = -np.sum(hist1 * np.log(hist2))
        return cross_entropy

    def save_histograms(self, root_dir, save_path):
        histograms = {}
        for filename in os.listdir(root_dir):
            if filename.endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(root_dir, filename)
                hist = self.calculate_histogram(img_path)
                histograms[img_path] = hist
        with open(save_path, 'wb') as f:
            pickle.dump(histograms, f)

    def load_histograms(self, load_path):
        with open(load_path, 'rb') as f:
            histograms = pickle.load(f)
        return histograms

    def extract_dominant_colors(self, img_path):
        try:
            img = Image.open(img_path)  # PIL 라이브러리를 이용하여 이미지 불러오기
            img = np.array(img)  # 이미지를 numpy 배열로 변환
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # RGB에서 BGR로 변환
        except IOError:
            print(f"Cannot read image at {img_path}")
            return None 

        img_cv = cv2.resize(img, self.resize_shape)
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_rgb = img_rgb.reshape(-1, 3)

        # 이미지의 고유 색상 수를 확인하고, n_clusters 값 설정
        num_unique_colors = len(np.unique(img_rgb, axis=0))
        n_clusters = min(num_unique_colors, self.num_colors)

        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        kmeans.fit(img_rgb)
        dominant_colors = kmeans.cluster_centers_
        return dominant_colors


    def compare_histograms(self, hist1, hist2):
        return self.calculate_histogram_cross_entropy(hist1, hist2)

    def compare_colors(self, colors1, colors2):
        color_distance = np.linalg.norm(colors1 - colors2, axis=1)
        return np.mean(color_distance)

    def predict(self, target_image_path, histograms):
        target_hist = self.calculate_histogram(target_image_path)
        target_colors = self.extract_dominant_colors(target_image_path)
        similarities = []
        for filename, hist in histograms.items():
            similarity_hist = self.compare_histograms(target_hist, hist)
            similarity_colors = self.compare_colors(target_colors, self.extract_dominant_colors(filename))
            similarity = similarity_hist + similarity_colors
            similarities.append((filename, similarity))
        sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=False)
        return sorted_similarities
