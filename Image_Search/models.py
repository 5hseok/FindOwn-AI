import os
from PIL import Image, ImageFile
import torch
import requests
from io import BytesIO
import urllib
from efficientnet_pytorch import EfficientNet
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from numpy import dot
from numpy.linalg import norm
import torch
from torchvision.models.detection.retinanet import RetinaNet_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
import numpy as np
from multiprocessing import Pool 
import pickle 
import torchvision.models as models
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2

ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImageDataset(Dataset):
    def __init__(self, image_files, transform=None):
        self.image_files = image_files
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]

        try:
            # Try to open the image file and convert to RGB
            img = Image.open(img_path).convert('RGB')

            if self.transform:
                img = self.transform(img)
        except Exception as e:
            print(f"Error occurred when loading image file {img_path}: {e}")
            return None

        return img_path, img
    
class Image_Search_Model:
    def __init__(self, model_name='efficientnet-b7', pre_extracted_features=None):
        self.model = EfficientNet.from_pretrained(model_name)

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        # 이미지 전처리: 크기 조정, 텐서 변환, 정규화 
        self.preprocess = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        # Load pre-extracted features if provided.
        if pre_extracted_features is not None:
            print(f"Loading features from {pre_extracted_features}")
            with open(pre_extracted_features,'rb') as f:
                self.features=pickle.load(f)
            print(f"Loaded {len(self.features)} features")
            
    def predict(self, img):
        # Load the image
        # img = Image.open(image_path).convert('RGB')

        # Preprocess the image
        img_tensor = self.preprocess(img)
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()

        # Extract features using the model
        with torch.no_grad():
            self.model.eval()
            features = self.model.extract_features(img_tensor)

            # Average pooling and flatten for simplicity.
            out_features = F.adaptive_avg_pool2d(features, 1).reshape(features.shape[0], -1).cpu().numpy()

        return out_features[0]

    def extract_features(self,root_dir):
        features = []
        # Load image files from the root directory 
        self.image_files = [os.path.join(dirpath, f)
                            for dirpath, dirnames, files in os.walk(root_dir)
                            for f in files if f.endswith('.jpg') or f.endswith('.png')]
        print(len(self.image_files))
        dataset = ImageDataset(self.image_files, transform=self.preprocess)

        dataloader = DataLoader(dataset,
                                batch_size=32,
                                num_workers=4,
                                pin_memory=True if torch.cuda.is_available() else False)

        pbar = tqdm(total=len(self.image_files), desc="Extracting Features")
        for paths, images in dataloader:
            if torch.cuda.is_available():
                images = images.cuda()
                self.model.eval()
                features_batch = self.model.extract_features(images)
                out_features_batch = F.adaptive_avg_pool2d(features_batch , 1).reshape(features_batch .shape[0], -1).cpu().numpy()

            for path,out_feature in zip(paths,out_features_batch ):
                new_feature_pair=(path,out_feature)
                features.append(new_feature_pair)

                yield new_feature_pair

            pbar.update(images.shape[0])
        
        pbar.close()
        
    def remove_duplicated_images(self, image_list, topN, error_rate=0.05):
        # Sort image list by similarity score
        sorted_list = sorted(image_list, key=lambda x: x[1], reverse=True)
        
        # Initialize the result list with the first image
        result_list = [sorted_list[0]]
        
        for i in range(1, len(sorted_list)):
            # If the similarity score difference is within the error rate, skip this image
            if abs(result_list[-1][1] - sorted_list[i][1]) / result_list[-1][1] < error_rate:
                continue
            # Otherwise, add this image to the result list
            result_list.append(sorted_list[i])
            # If the length of the result list is equal to topN, break the loop
            if len(result_list) == topN:
                break

        return result_list

    def search_similar_images(self, target_image_path, topN=10):
            # Check if target_image_path is a URL
        if target_image_path.startswith('http://') or target_image_path.startswith('https://'):
            # If target_image_path is a URL, download the image and convert it to a PIL image
            response = requests.get(target_image_path)
            target_image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            # If target_image_path is not a URL, assume it is a file path
            target_image = Image.open(target_image_path).convert('RGB')
        
        # Extract feature from target image
        target_embedding = self.predict(target_image)

        if target_embedding is not None and hasattr(self, "features"):
            distances = []

            for feature in self.features:
                feature_path, feature_vector = feature
                distance = torch.nn.functional.cosine_similarity(torch.tensor(target_embedding), torch.tensor(feature_vector), dim=0)
                distances.append((feature_path, distance.item()))

            topN_image_list = self.remove_duplicated_images(distances, topN)

            return topN_image_list


class Image_Object_Detections:
    def __init__(self,topN=10):
        self.target_object = set()
        self.topN_object = []
        for i in range(topN):
            self.topN_object.append(set())
        self.model = models.detection.retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.COCO_V1)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            
    @staticmethod
    def get_display_name_from_id(target_id):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, 'mscoco_label_map.pbtxt')

        with open(file_path, 'r') as f:
            lines = f.readlines()

        items = []
        item = {}
        for line in lines:
            if line.strip() == "}":
                items.append(item)
                item = {}
                continue

            if ":" in line:
                key, value = [x.strip() for x in line.split(":")]
                if key == "id":
                    item[key] = int(value)
                else:
                    # Remove quotes around the string
                    item[key] = value.strip('"')

        # Now we have a list of dictionaries where each dictionary represents an item.
        # We can search this list to find the display_name corresponding to the target_id.
        
        for item in items:
            if 'id' in item and 'display_name' in item and item['id'] == target_id:
                return item['display_name']
            
    def detect_objects(self, image_path, search_score):
        image = Image.open(image_path).convert("RGB")
        image_tensor = to_tensor(image).unsqueeze(0)

        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()

        outputs = self.model(image_tensor)
        
        detected_objects = set()
        
        for i, output in enumerate(outputs):
            for box, label, score in zip(output['boxes'], output['labels'], output['scores']):
                if label != 15 and label != 28 and score >= search_score:
                    detected_objects.add(self.get_display_name_from_id(label))
                    
        return detected_objects

    def visualize_image(self, image_path):
        """Just display the original image without any annotations."""
        image = Image.open(image_path)
        plt.imshow(image)
        plt.axis('off')  # Don't show axis

    def search_similar_images(self,target_image_path,topN_image_list, search_score = 0.10):
        
        target_object=self.detect_objects(target_image_path,search_score)
        
        image_object_counts = []

        for (image_path_in_list, precision) in topN_image_list:
            topN_object = self.detect_objects(image_path_in_list, search_score)
            common_objects = list(target_object & topN_object)
            image_object_counts.append((image_path_in_list, common_objects, len(common_objects)))

        # Sort images by the number of common objects in descending order and select top 3
        top3_images = sorted(image_object_counts, key=lambda x: x[2], reverse=True)[:3]

        titles = ["Top 1", "Top 2", "Top 3", "Target Image"]

        fig=plt.figure(figsize=(20,20))
        
        for i in range(len(top3_images)):
            plt.subplot(2 ,2 , i+1) 
            image_path,_ ,_ = top3_images[i]
            
            self.visualize_image(image_path) 
            
            plt.title(titles[i])
        
        # Show target image at last position
        plt.subplot(2 ,2 ,4) 
        
        self.visualize_image(target_image_path) 

        plt.title(titles[-1])
        
        plt.tight_layout() 
        plt.show()
        return top3_images
                    
class ColorSimilarityModel:
    def __init__(self, num_bins=30):
        self.num_bins = num_bins

    def calculate_histogram(self, img_path):
        img = cv2.imread(img_path)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv_img], [0], None, [self.num_bins], [0, 180])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist

    def save_histograms(self, root_dir, save_path):
        histograms = {}

        for filename in os.listdir(root_dir):
            if filename.endswith(('.jpg', '.png', '.jpeg')):  # 이미지 파일만 처리
                img_path = os.path.join(root_dir, filename)
                hist = self.calculate_histogram(img_path)
                histograms[filename] = hist

        with open(save_path, 'wb') as f:
            pickle.dump(histograms, f)

    def load_histograms(self, load_path):
        with open(load_path, 'rb') as f:
            histograms = pickle.load(f)
        return histograms

    def compare_histograms(self, hist1, hist2):
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    def predict(self, target_image_path, histograms):
        target_hist = self.calculate_histogram(target_image_path)
        similarities = []

        for filename, hist in histograms.items():
            similarity = self.compare_histograms(target_hist, hist)
            similarities.append((filename, similarity))

        return similarities