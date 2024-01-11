# FindOwn-AI

## 레포 사용 순서

### 키프리스 이미지 사용 o - API를 통해 상표 정보 제공
1. requirements.txt에서 필요한 라이브러리 및 패키지 install
2. zip 파일들에서 pkl 파일들을 analysis_image 폴더 내에 압축 해제 
3. python manage.py runserver 실행
4. api 사용

### ~~~.pkl
이미지 로고 파일의 모델별 특징 추출 파일. 키프리스에 존재하는 상표들 이미지 데이터를 모아놓은 파일을 root_dir로 지정해서 pkl로 생성이 가능하다. 현재는 구동을 위해 cnn_features_Kipris.pkl 파일만 사용한다.

### 점수 판별 방식
각 모델에서 얻어진 유사도 점수를 normalization시켜 일정하게 만든 후, 가중치를 곱해 점수를 내었다.
최종 점수가 0.72점 이상이라면 danger, 0.59점 이상이라면 warning, 0.59미만의 점수를 가진다면 safe로 판단하였다. 

### 비고
색상 유사도에 영향을 많이 줄 수 있는 데이터가 colorHistograms_logo_Kipris 파일입니다. AI_main 52행부터 68행의 주석을 풀고 사용이 가능한데, 너무 요청이 돌아오는 시간이 오래 걸려서 이 모델은 넣지 않았습니다. 추후에 가능할 것 같으면 주석 풀어서 추가하는 것도 고려하면 좋을 것 같습니다.