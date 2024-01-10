import requests
from django.core.files.base import ContentFile
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse
from django.test import TestCase, Client
from rest_framework import status

class ImageProcessViewTest(TestCase):
    def setUp(self):
        self.client = Client()

    def test_image_process(self):
        data = {
            'image': "https://trademark.help-me.kr/images/blog/trademark-registration-all-inclusive/image-05.png"
        }
        response = self.client.post(reverse('ImageProcessView'), data, format='application/json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)