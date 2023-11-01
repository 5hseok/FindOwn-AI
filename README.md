# FindOwn-AI
### extract_features.py
ImageDB에서 이미지들을 미리 특징 추출하여 pkl로 바꾸는 작업을 위한 코드
실제 사용에서는 필요 x
### models.py
실질적인 AI 모델을 class 2~3개로 정의. pretrained 모델을 사용한터라 import만 잘 시켜주면 init에서 모델 설정 끝
메소드를 통해 유사도 판단 결과 반환
Image_Search_Model은 타겟 이미지와 유사한 이미지 10개를 반환하고, Object_Model은 Image_Search_Model에서 반환한 리스트를 입력받아 그 이미지 리스트에서 가장 유사한 객체를 사용한 top 3이미지를 반환
ColorModel은 색상 분포가 가장 유사한 순서대로 리스트를 반환. 리스트의 길이는 이미지 DB 길이와 일치. 여기서는 logo 데이터만 pkl로 만들어서 1710개
SSIM 방식은 폐지 예정
이 3가지 모델의 유사도를 가중치를 두어(예시 : 이미지 유사도 0.6, 객체 탐지 0.1, 색상 분포 0.3) 점수를 계산할 예정
안전, 주의, 위험의 기준은 정해봐야 할 듯
### main.py
models.py를 사용하는 사용 예시
현재는 깔끔하게 이미지만 url로 넘기는 방식이 아니라 구현을 위해 print문과 plt.show()로 이미지를 직접 확인하게 했다.(수정 필요)
### features_logo.pkl & colorHistograms_logo.pkl
이미지 로고 파일의 특징 추출 파일과 색상 분포 추출 파일. 총 파일 개수 각각 1710개
키프리스 플러스에서 받은 Trademark.pkl도 존재하나, 용량이 너무 커서(3GB) 깃허브에 올리진 못하고 직접 전달해야할 듯. + 도형상표만 다운받은게 아니라 전체 상표이기 때문에 문자 상표 다수 포함. 총 파일 개수 36만8천개
### mscoco_lable_map.pbtxt
object_model에서 필요한 pbtxt파일. 어떤 객체를 탐지했는지를 글자로 보여주기 위한 파일인데, 실제 구현에서는 무슨 객체를 탐지했는지를 전달하지 않으므로 사용하진 않지만, 파일을 전달 안했을 시 오류가 발생할 수 있어서 코드 수정 or 그냥 파일 전달 필요
### ImageDB
로고 이미지 데이터들이 존재하는 폴더. 여기서 이미지를 출력해오고 전송한다. 이미지를 url로 전송할 수 있다면, 역시 필요 x. 그러나 root_Dir에 들어가는 부분이므로 코드 수정 필요
### EfficientNet
이미지 유사도 모델에서 불러와야 하는 모델인데 없어도 되나? 싶다. 확인 필요
### Image_Search_TotalImage.ipynb
사용 x 