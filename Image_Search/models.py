import os
from PIL import Image
import torch
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


class Image_Search_Model:
    def __init__(self, root_dir, model_name='efficientnet-b7', return_nodes={'avgpool':'avgpool'}, pre_extracted_features=None):
        self.model = EfficientNet.from_pretrained(model_name)
        self.return_nodes = return_nodes
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        # 이미지 전처리: 크기 조정, 텐서 변환, 정규화 
        self.preprocess = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        # Load image files from the root directory 
        self.image_files = [os.path.join(root_dir,f) for f in os.listdir(root_dir) if f.endswith('.jpg') or f.endswith('.png')]
        
        # Load pre-extracted features if provided.
        if pre_extracted_features is not None:
            with open(pre_extracted_features,'rb') as f:
                self.features=pickle.load(f)
            
    def predict(self,image_path):
        
        feature_file=image_path+'.feature.pkl'
       
        if os.path.isfile(feature_file):
            with open(feature_file,'rb') as f:
                return pickle.load(f)

    def extract_features(self):
         # If features are already loaded no need to do it again
         if hasattr(self,"features"):
             return
         
         with Pool() as p:
             self.features=p.map(self.predict,self.image_files)

    def search_similar_images(self,target_image_path,topN=10):
        
          # Extract feature from target image
          target_embedding=self.predict(target_image_path)
         
          if target_embedding is not None and hasattr(self,"features"):
              distances=[]
             
              for feature in self.features:
                  distance=torch.nn.functional.cosine_similarity(torch.tensor(target_embedding),torch.tensor(feature),dim=0)
                  distances.append(distance.item())
             
              indices=np.argsort(distances)[::-1][:topN]
             
              return [(self.image_files[i],distances[i]) for i in indices]


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
        with open('mscoco_label_map.pbtxt', 'r') as f:
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
        image_tensor = F.to_tensor(image).unsqueeze(0)

        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()

        outputs = self.model(image_tensor)
        
        detected_objects = set()
        
        for i, output in enumerate(outputs):
            for box, label, score in zip(output['boxes'], output['labels'], output['scores']):
                if label != 15 and label != 28 and score >= search_score:
                    detected_objects.add(self.get_display_name_from_id(label))
                    
        return detected_objects

    def visualize_detections(self, image_path, detected_objects,search_score):
        image = Image.open(image_path).convert("RGB")
        image_tensor = F.to_tensor(image).unsqueeze(0)

        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()

        outputs = self.model(image_tensor)

        fig, ax = plt.subplots(1)
        ax.imshow(image)

        for i, output in enumerate(outputs):
            for box, label, score in zip(output['boxes'], output['labels'], output['scores']):
                if label != 15 and label != 28 and score >= search_score and self.get_display_name_from_id(label) in detected_objects:
                    xmin, ymin, xmax, ymax = box.detach().cpu().numpy()
                    rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,
                                        edgecolor='r',facecolor='none')
                    ax.add_patch(rect)
                    
    def search_similar_images_test(self,target_image_path,topN_image_list, search_score = 0.05):
        # 이미지에 객체를 잘 탐지했나 이미지로 출력해줌.
        # 각 이미지의 공통 객체를 출력하고, 가장 많이 발생하는 3개의 객체를 출력함.
        # object_detection의 정확성을 알아보기 위한 코드.
        
        target_object=self.detect_objects(target_image_path, search_score)
        
        topN_object=[]
        
        for image_path, precision in topN_image_list:
            topN_object.append(self.detect_objects(image_path, search_score))
            
        count=0

        counter=Counter()
        
        # Create a figure with multiple subplots
        fig=plt.figure(figsize=(20,20))
        
        for i in range(len(topN_object)):
            object_list=list(target_object & topN_object[i])
            if len(object_list)>0:
                print(topN_image_list[i])
                print("And object is",object_list)
                counter.update(object_list)
                                
                # Visualize detections on the current image
                self.visualize_detections(topN_image_list[i], object_list, search_score)
                
            else:
                count+=1
    
        most_common_elements=counter.most_common(3)

        print(most_common_elements)
        
        if len(target_object)==count:
            print("No")
        
        plt.show()
    
    def search_similar_images(self,target_image_path,topN_image_list, search_score = 0.05):
        
        target_object=self.detect_objects(target_image_path,search_score)
        
        image_object_counts = []

        for image_path_in_list, precision in topN_image_list:
            topN_object = self.detect_objects(image_path_in_list, search_score)
            common_objects = list(target_object & topN_object)
            image_object_counts.append((image_path_in_list, common_objects, len(common_objects)))

        # Sort images by the number of common objects in descending order and select top 3
        top3_images = sorted(image_object_counts, key=lambda x: x[1], reverse=True)[:3]

        # Create a figure with multiple subplots
        fig=plt.figure(figsize=(20,20))
        
        for i in range(len(top3_images)):
            image_path, object_contents, object_count = top3_images[i]

            # Visualize detections on the current image
            detected_objects = self.detect_objects(image_path, search_score)
            self.visualize_detections(image_path, detected_objects, search_score)

        plt.show()
        return top3_images
