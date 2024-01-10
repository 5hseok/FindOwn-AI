import requests
from xml.etree import ElementTree

def get_info_from_api(image_urls):
    
    # API URL
    base_url = "http://plus.kipris.or.kr/openapi/rest/trademarkInfoSearchService/applicationNumberSearchInfo"

    # API 키
    access_key = "DhCS9cfe7FbMasOCcTQlQSzIL9lDRNwf70eiMNQlM3M="

    # 결과를 저장할 리스트
    results = []

    # 이미지 URL 리스트를 순회
    for image_url in image_urls:
        # 출원번호 추출 (이미지 URL의 마지막 부분)
        application_number = image_url.split('/')[-1]
        application_number = application_number.split('_')[0]
        # 요청 URL
        request_url = f"{base_url}?applicationNumber={application_number}&accessKey={access_key}"

        # API 요청 보내기
        response = requests.get(request_url)

        # 응답 확인
        if response.status_code == 200:
            # 응답을 XML 형식으로 파싱
            root = ElementTree.fromstring(response.content)
            
            # 결과에 추가
            results.append(root)
        else:
            print(f"Error for {image_url}: {response.status_code}")
    # 결과 반환
    return results
