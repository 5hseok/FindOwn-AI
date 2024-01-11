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
        image_url = "https://trademark.help-me.kr/images/blog/trademark-registration-all-inclusive/image-05.png"
        response = self.client.get(reverse('ImageProcessView'), {'image': image_url})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
