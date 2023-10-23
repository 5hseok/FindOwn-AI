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
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class ImageDataset(Dataset):
    def __init__(self, image_files, transform=None):
        self.image_files = image_files
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img_path, img
    
class Image_Search_Model:
    def __init__(self, root_dir, checkpoint_file,model_name='efficientnet-b7', return_nodes={'avgpool':'avgpool'}, pre_extracted_features=None):
        self.model = EfficientNet.from_pretrained(model_name)
        self.return_nodes = return_nodes
        self.checkpoint_file = checkpoint_file
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        # 이미지 전처리: 크기 조정, 텐서 변환, 정규화 
        self.preprocess = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        # Load image files from the root directory 
        self.image_files = [os.path.join(dirpath, f)
                            for dirpath, dirnames, files in os.walk(root_dir)
                            for f in files if f.endswith('.jpg') or f.endswith('.png')]
        
        # Load pre-extracted features if provided.
        if pre_extracted_features is not None:
            print(f"Loading features from {pre_extracted_features}")
            with open(pre_extracted_features,'rb') as f:
                self.features=pickle.load(f)
            print(f"Loaded {len(self.features)} features")
            
    def predict(self, image_path):
        # Load the image
        img = Image.open(image_path).convert('RGB')

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


    def extract_features(self):
        checkpoint_interval = 640  # Save every 1,000 images.
        batch_num = 0

        # Load from checkpoint if it exists
        if os.path.isfile(self.checkpoint_file):
            with open(self.checkpoint_file, 'rb') as f:
                features = pickle.load(f)
                print(f"Loaded {len(features)} features from checkpoint")

        processed_files = {path for path, _ in features}

        dataset = ImageDataset(self.image_files, transform=self.preprocess)

        dataloader = DataLoader(dataset,
                                batch_size=32,
                                num_workers=4,
                                pin_memory=True if torch.cuda.is_available() else False)

        pbar = tqdm(total=len(self.image_files), initial=len(processed_files), desc="Extracting Features")

        for paths, images in dataloader:
            if torch.cuda.is_available():
                images = images.cuda()

            with torch.no_grad():
                self.model.eval()
                features_batch = self.model.extract_features(images)
                out_features_batch = F.adaptive_avg_pool2d(features_batch , 1).reshape(features_batch .shape[0], -1).cpu().numpy()

            for path,out_feature in zip(paths,out_features_batch ):
                if path not in processed_files: 
                    new_feature_pair=(path,out_feature)
                    features.append(new_feature_pair)

                    yield new_feature_pair

            pbar.update(images.shape[0])

            # Save intermediate results to separate files at intervals
            if batch_num % checkpoint_interval == 0:
                with open(f'features_checkpoint_{batch_num // checkpoint_interval}.pkl', 'wb') as f:
                    pickle.dump(features, f)
                features.clear()   # Clear the list to save memory
            
            batch_num += 1

        pbar.close()
       
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

    def search_similar_images(self,target_image_path,topN_image_list, search_score = 0.05):
        
        target_object=self.detect_objects(target_image_path,search_score)
        
        image_object_counts = []

        for image_path_in_list, precision in topN_image_list:
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
        
        return top3_images
                    
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
