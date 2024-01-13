from django.db import models

class ImageFeature_CNN(models.Model):
    image_path = models.TextField()  # 이미지 경로. 중복을 피하기 위해 unique=True 설정.
    feature = models.JSONField()  # 이미지 특징. JSON 형태로 저장합니다.

    def __str__(self):
        return self.image_path

class ImageFeature_Eff(models.Model):
    image_path = models.TextField(unique=True)  # 이미지 경로. 중복을 피하기 위해 unique=True 설정.
    feature = models.BinaryField()  # 기본값을 빈 바이너리 문자열로 설정


