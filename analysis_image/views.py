from rest_framework import viewsets, permissions, generics, status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from django.http import JsonResponse, HttpResponseBadRequest
from django.core.files.base import ContentFile
from django.core.files.uploadedfile import SimpleUploadedFile
from django.views import View
import requests
import json
import xml.etree.ElementTree as ET
from xml.etree import ElementTree
from .AI_main import Image_Analysis
from .search_Trademark_API import get_info_from_api
import xmltodict

class ImageProcessView(View):
    def get(self, request):
        # 이미지 URL을 받아 분석 및 Open API 요청 처리
        image_url = request.GET.get('image')
        if image_url is None:
            return HttpResponseBadRequest("Image URL is required.")
        
        analyzed_info = self.analyze_image(image_url)
        api_info = self.get_info_from_Kipris(analyzed_info)
        result = self.combine_info(analyzed_info, api_info)
        return JsonResponse({'result': result}, safe=False)

    def analyze_image(self, image_path):
        # 이미지 분석 코드 작성
        image_analysis = Image_Analysis()
        return image_analysis.start_analysis(image_path, False)
    
    def get_info_from_Kipris(self, analyzed_info):
        analyzed_info = json.loads(analyzed_info)
        image_urls = [result['image_path'] for result in analyzed_info]
        result = get_info_from_api(image_urls)
        return result
    
    def combine_info(self, analyzed_info, api_info):
        analyzed_info = json.loads(analyzed_info)
        combined_info = []
        for i in range(len(analyzed_info)):
            combined_dict = analyzed_info[i]
            del(combined_dict['image_path'])
            xml_string = ET.tostring(api_info[i], encoding="utf-8").decode("utf-8")
            dict_ = xmltodict.parse(xml_string)
            combined_dict.update(dict_['response']['body']['items']['TradeMarkInfo'])
            combined_info.append(combined_dict)
        return combined_info