# FindOwn-AI

## 레포 사용 순서
### 키프리스 이미지 사용 x - 유사한 로고 이미지 찾기
1. ImageDB\\Logos에 있는 이미지 파일들을 내려받는다.
2. requirement.txt에 있는 lib들을 설치한다. (pip install -r requirements.txt)
3. main.py의 35행 root_dir에 본인이 이미지 파일들을 저장한 디렉토리의 주소를 입력한다.
4. main.py의 target_image_path에는 사용자가 업로드한 이미지의 url을 넣는다.
5. main.py의 155행 부분의 url을 josn 데이터를 보낼 url로 설정한다.

### 키프리스 이미지 사용 o - API를 통해 상표 정보 제공
1. 따로 받은 이미지 파일들 C:\\users\\FindOwn에 내려받기
2. 위와 동일하게 실행하여 pkl 파일을 직접 저장하거나(첫 소요시간 매우 큼), 따로 전달받은 pkl 파일을 같은 위치에 넣어둔다.
3. search_Trademark_API.py 파일 개발 중

# 현재는 키프리스 이미지 데이터 36만개를 전처리하여 pkl로 저장하는 과정 중에 있습니다. 그러나, 너무 많은 파일 크기로 인해 상당한 시간이 요구되고 있습니다. 

### 비고
- json의 post url과 사용자의 이미지 url을 받아오는 target_image_path는 우선 임의의 url을 넣어뒀습니다. 현재 target_image_path는 빽다방 로고이고, post url은 example로 해놨습니다. json의 header 역시 application/json으로 임의로 넣었습니다.
- main.py의 167행부터 170행까지는 json이 잘 생성되었는지 확인하는 용도이므로 삭제하셔도 무방합니다.
- features_logo.pkl과 object_logo.pkl은 생성에 시간이 조금 걸립니다. 혹시 pkl 파일 만들어지는 과정에서 오류가 발생하여 pkl 파일이 만들어지지 않는다면, root_dir과 함께 오류 메시지 알려주시면 제쪽에서 따로 처리 후 보내드리겠습니다.
- pkl 파일은 한번 만들어놓으면 그 다음 시행부터는 다시 생성하지 않습니다.


### models.py
실질적인 AI 모델을 class 3~4개로 정의. pretrained 모델을 사용한터라 import만 잘 시켜주면 init에서 모델 설정 끝
메소드를 통해 유사도 판단 결과 반환
사용자의 타겟 이미지만 따로 특징 추출하여 미리 다른 이미지들의 특징을 추출해 저장해둔 pkl 파일에서 값을 불러와 비교 후 
점수를 매김.

### main.py
models.py를 사용하는 코드
root_dir에 있는 이미지들을 pkl화 시키고, 사용자의 타겟 이미지와 비교하여 점수를 매긴다. 가장 높은 점수를 기록한 이미지 N개(현재는 3개)를 json 데이터로 만듬. 
json 데이터는 한 이미지 당 image_path, result로 묶었으며, image_path에서는 로컬 이미지 주소, result는 침해도 정도이다. 
침해도 정도는 danger, warning, safe로 구분하였고, 디즈니, 몬스터와 같은 로고에 민감한 브랜드들이 top N 이미지에 포함된다면 danger로 구분하였다.

### ~~~.pkl
이미지 로고 파일의 모델별 특징 추출 파일. Logos에 있는 이미지 1710개를 전처리 시킨 데이터들이다.
키프리스 플러스 혹은 AI 허브에 존재하는 상표들 내지는 이미지 데이터를 모아놓은 파일을 root_dir로 지정해서 pkl로 생성이 가능하다.
하지만, 도형 상표만 다루기 때문에 따로 문자 상표를 지워야 해서 일단은 로고 이미지만 사용. 

### mscoco_lable_map.pbtxt
object_model에서 필요한 pbtxt파일. 어떤 객체를 탐지했는지를 글자로 보여주기 위한 파일인데, 실제 구현에서는 무슨 객체를 탐지했는지를 전달하지 않으므로 사용하진 않지만, 파일을 전달 안했을 시 오류가 발생할 수 있어서 코드 수정 or 그냥 파일 전달 필요

### ImageDB\\Logos
로고 이미지 데이터들이 존재하는 폴더. 이 디렉토리의 위치를 root_dir에 입력해야 한다.
### EfficientNet
이미지 유사도 모델에서 불러와야 하는 모델인데 없어도 되나? 싶다. 확인 필요

### 점수 판별 방식
각 모델에서 얻어진 유사도 점수를 normalization시켜 일정하게 만든 후, 가중치를 곱해 점수를 내었다.
최종 점수가 0.9점 이상이라면 danger, 0.6점 이상이라면 warning, 0.6미만의 점수를 가진다면 safe로 판단하였다. 
safe가 잘 안 나오는 기준 점수라서 warning 점수를 높이는 등의 조치 필요 