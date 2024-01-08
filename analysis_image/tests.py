from django.test import TestCase, Client
from django.core.files.uploadedfile import SimpleUploadedFile

class ImageProcessViewTest(TestCase):
    def setUp(self):
        self.client = Client()

    def test_image_process(self):
        # 이미지 파일 읽기
        with open('path_to_your_image.jpg', 'rb') as image_file:
            image_content = image_file.read()

        # SimpleUploadedFile 객체 생성
        image = SimpleUploadedFile('test.jpg', image_content)

        # POST 요청 보내기
        response = self.client.post('/api/image_process/', {'image': image})

        # 응답 확인
        self.assertEqual(response.status_code, 200)
        print(response.json())
