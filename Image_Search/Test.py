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