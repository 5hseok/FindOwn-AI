# 앱 레벨의 urls.py
from django.urls import path
from .views import ImageProcessView

urlpatterns = [
    path('imageprocess/', ImageProcessView.as_view(), name='ImageProcessView'),
]
