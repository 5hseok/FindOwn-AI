import pickle
# from analysis_image.models import ImageFeature_CNN  # DRF 모델
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'analysis_image.settings')

# pkl 파일 로드
with open('C:\\Users\\DGU_ICE\\FindOwn\\analysis_image\\features_logo_Kipris.pkl', 'rb') as f:
    data = pickle.load(f)

# 모델 인스턴스 생성 및 저장
for image_path, feature in data.items():
    instance = ImageFeature_CNN(image_path=image_path, feature=feature.tolist())  # numpy 배열을 리스트로 변환
    instance.save()