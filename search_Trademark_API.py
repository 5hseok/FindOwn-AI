import json
import xml

import requests
import json

# API URL
url = "http://api.example.com/endpoint"

# API 키
api_key = "api_key"

# 파라미터
params = {
    "param1": "value1",
    "param2": "value2",
    # 필요한 만큼 추가
}

# 헤더
headers = {
    "Authorization": "Bearer " + api_key,  # Bearer 다음에 공백이 필요합니다.
    "Content-Type": "application/json",
    # 필요한 만큼 추가
}

# API 요청 보내기
response = requests.get(url, params=params, headers=headers)

# 응답 확인
if response.status_code == 200:
    # 응답을 JSON 형식으로 파싱
    data = response.json()
    print(json.dumps(data, indent=4))  # JSON 데이터를 보기 좋게 출력
else:
    print("Error:", response.status_code)
