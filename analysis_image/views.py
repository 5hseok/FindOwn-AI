from rest_framework import viewsets, permissions, generics, status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from django.http import JsonResponse
from django.views import View
import requests
import json
from xml.etree import ElementTree
from .AI_main import Image_Analysis
from .search_Trademark_API import get_info_from_api

class ImageProcessView(View):
    def post(self, request):
        # 이미지 분석 및 Open API 요청 처리
        image = request.FILES['image']
        analyzed_info = self.analyze_image(image)
        api_info = self.get_info_from_Kipris(analyzed_info)
        result = self.combine_info(analyzed_info, api_info)
        
        return JsonResponse(result)
    
    def analyze_image(self, image_path):
        # 이미지 분석 코드 작성
        image_analysis = Image_Analysis()
        return image_analysis.start_analysis(image_path)
    
    def get_info_from_Kipris(self, analyzed_info):
        image_urls = [result['image_url'] for result in analyzed_info['image_path']]
        print(image_urls)
        result = get_info_from_api(image_urls)
        return result
    
    def combine_info(self, analyzed_info, api_info):
        # 두 결과를 합친 dictionary 생성
        combined_results = {
            "patent_info": [ElementTree.tostring(root, encoding='unicode') for root in api_info],
            "results_list": analyzed_info,
        }
        
        return combined_results
