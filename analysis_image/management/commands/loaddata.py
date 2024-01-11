from django.core.management.base import BaseCommand
from analysis_image.models import ImageFeature_Eff  # 이미지 특징을 저장할 모델
import pickle
class Command(BaseCommand):
    help = 'Load image features from pickle file'

    def handle(self, *args, **options):
        # pickle 파일에서 데이터를 불러옵니다.
        with open('C:\\Users\\DGU_ICE\\FindOwn\\analysis_image\\features_logo_Kipris.pkl', 'rb') as f:
            data = pickle.load(f)

        # 불러온 데이터를 처리합니다.
        for item in data:
            image_path = item[0]
            feature = item[1]

            # ImageFeature 모델에 데이터를 저장합니다.
            # 이 부분은 실제 사용하려는 모델과 필드에 따라 수정해야 합니다.
            ImageFeature_Eff.objects.create(image_path=image_path, feature=feature)
