# 앱 레벨의 urls.py
from django.urls import path
from .views import ImageProcessView

urlpatterns = [
    path('image_process/', ImageProcessView.as_view(), name='image_process'),
]
