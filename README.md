# 🐾 FARM 동아리 4팀
FindOwn 프로젝트팀입니다.

## ✍️ 프로젝트 개요
 ‘Find OWN - 나만의 상표권을 찾아서’ <br/>
 사용자들이 상표권을 등록하는 과정에서 겪는 어려움이나 불편함을 해결할 수 있도록 다양한 서비스 및 정보를 제공하는 웹 서비스입니다. <br/> 
<br/>

## 🧑‍💻 프로젝트 소개
> 프로젝트명

FindOwn, 상표권 침해 판단 웹플랫폼 <p>
[FindOwn 사이트에 접속해보세요!](https://www.find-own.site/)
<br/> <br/>

> 프로젝트 내용

상표권 등록을 위한 정보를 찾아보면 자세한 정보는 변리사에게 상담받아야 얻을 수 있다는 답변을 많이 찾아볼 수 있습니다. 상표권 등록을 위해 인터넷에서 정보를 찾아보는 경우는 1인 창업 혹은 스타트업을 목적으로 하는 경우가 많은데 상표 등록을 위한 정보를 찾는 것부터 비용을 지불하는 것은 많은
부담이 됩니다. 또한 상표 등록을 위한 상담이 아닌 상표 등록 자체도 변리사에게 맡기는 경우가 많은데 이런 부담을 줄이기 위해 고안한 것이 상표권의 침해도를 인공지능으로 판단하고, 상표권 등록에 비슷한 어려움과 고민을 갖고 있는 사람들에게 자문을 구하거나 경험을 공유하여 상표권 등록 절차를 돕는 것이 <b>FIND OWN</b> 서비스입니다.
<br/><br/>

> 프로젝트 목표

- 상표권 등록의 어려움 해소

   - FIND OWN은 사용자들이 상표권 등록을 위해 필요한 정보를 더욱 손쉽게 얻을 수 있도록 하고, 변리사 상담에 의존하지 않고도 필수적인 단계들을 이해하고 진행할 수 있도록 돕습니다
- 비용 부담 감소

   - 무료로 이용 가능한 상표권 침해 판단 도구 및 상표권 커뮤니티를 제공하여 사용자들이 비용을 절감하면서도 효과적인 침해 여부를 확인할 수 있도록 지원합니다. 
- 상표 침해 판단 및 대응 강화

    - AI 기술을 활용하여 사용자가 등록하려는 상표가 다른 등록된 상표들과의 침해 여부를 판단하여 위험도를 제시합니다.
   - 이에 대한 적절한 대응 방안과 조언을 제공하여 사용자가 더욱 신속하게 대응할 수 있도록 지원합니다.
- 정보 제공과 커뮤니티 구축

   - 서비스는 상표 등록과 관련된 다양한 정보를 트렌드, 침해, 등록, 판례 등의 카테고리로 제공하여 사용자들이 필요한 정보를 쉽게 찾을 수 있도록 합니다.
   - 사용자 커뮤니티를 통해 상표 등록과 관련된 경험을 나누고 서로에게 조언을 구할 수 있는 플랫폼을 제공하여 사용자 간의 지식 공유를 촉진합니다.
- 상표침해도 판단

   -  사용자가 업로드한 상표 이미지를 분석하여 침해 여부를 정확하게 판단하는 능력을 지속적으로 향상시키며, 사용자가 안전하게 타인의 상표 또는 자신의 상표를 보호할 수 있도록 지속적인 발전을 목표로 합니다.

<br/>

> 접근 방법

사용자가 직접만든 상표를 사이트에 업로드하면 유사한 아이템들과 그 침해도를 알려줍니다. 기호에 따라 공개적으로 저장할 수 있고 비공개저장을 할 수 있는데 공개적으로 저장한 경우 판단결과에 대해 '침해사례' 목록으로 다른 사람들과 공유할 수 있습니다. 
<br/><br/>

## 🎞️ 시연영상

[시연영상보러가기](https://youtu.be/aiCQ1GBuDR4?si=es3hVhS8A-FpTbfD)
<br/><br/>

## ⚙️ 시스템 아키텍처

<p align="center">
<img src="https://github.com/FarmSystem/FindOwn-Frontend/assets/96400257/b4d52c4c-7341-4988-85b6-ab509b4201ab.png" width="400" height="200" />
</p>

<br/>

## 🧨 패키지설치
- backend
```
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install openjdk-17.jdk
cd FindOwn-Backend/FindOwn-v2
./gradlew build
cd build/libs
java -jar FindOwn-v2-0.0.1-SNAPSHOT.jar
```
- frontend
```
yarn install
yarn start
```
_노드버전은 최소 20이상으로 해주셔야 합니다._
<br/><br/>

## 📚 기술 스택
<b>Common</b>

<img src="https://img.shields.io/badge/visualstudiocode-007ACC?style=flat-square&logo=visualstudiocode&logoColor=white"> <img src="https://img.shields.io/badge/ESLint-4B3263?style=flat-square&logo=ESLint&logoColor=white"> <img src="https://img.shields.io/badge/prettier-F7B93E?style=flat-square&logo=prettier&logoColor=white"> <img src="https://img.shields.io/badge/notion-000000?style=flat-square&logo=notion&logoColor=white"> <img src="https://img.shields.io/badge/slack-4A154B?style=flat-square&logo=slack&logoColor=white">

<b>Frontend</b>

<img src="https://img.shields.io/badge/react-444444?style=flat-square&logo=react&logoColor=white"> <img src="https://img.shields.io/badge/TypeScript-3178C6?style=flat-square&logo=TypeScript&logoColor=white"> <img src="https://img.shields.io/badge/recoil-f26b00?style=flat-square&logo=recoil&logoColor=white"> <img src="https://img.shields.io/badge/styledcomponents-DB7093?style=flat-square&logo=styledcomponents&logoColor=white"> <img src="https://img.shields.io/badge/mui-007FFF?style=flat-square&logo=mui&logoColor=white"> 
 

<b>Backend</b>

<img src="https://img.shields.io/badge/Spring-6DB33F?style=flat-square&logo=Spring&logoColor=white"> <img src="https://img.shields.io/badge/springboot-6DB33F?style=flat-square&logo=springboot&logoColor=white"> <img src="https://img.shields.io/badge/Amazon AWS-232F3E?style=flat-square&logo=Amazon AWS&logoColor=white"> <img src="https://img.shields.io/badge/redis-DC382D?style=flat-square&logo=redis&logoColor=white"> <img src="https://img.shields.io/badge/mysql-4479A1?style=flat-square&logo=mysql&logoColor=white">

<b>AI & Backend</b>

<img src="https://img.shields.io/badge/django-092E20?style=flat-square&logo=django&logoColor=white"> <img src="https://img.shields.io/badge/pytorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white">

<br/>

## 😈 팀원 소개

|       팀장       | 팀원 |       팀원       | 팀원 |       팀원       |       팀원       
|:--------------:|:--:|:--------------:|:--:|:--------------:|:--:|
|    **박서영**     |**임정우**|    **최재원**     |**이지민**|    **오현석**    |**정영준**| 
|     **BE**     |**BE**|   **FE**    |**FE**|   **AI**    |**Security**|
|   **경영정보학과**   |**정보통신공학과**|   **정보통신공학과**   |**수학과**|   **정보통신공학**   |**컴퓨터공학과**|
|   **2021111509**   |**2019112088**|   **2019112086**   |**2020110408**|   **2022112053**   |**2023112435**|g


# FindOwn-AI

## 레포 사용 순서

### 키프리스 API를 통해 상표 정보 제공

1. requirements.txt에서 필요한 라이브러리 및 패키지 install
2. git pull 후 dvc pull -> 데이터가 불러와짐.
2-1. 불러와진 tar.gz 파일을 압축해제하여 analysis_image dir 안에 넣기
3. python manage.py runserver 실행
4. api 사용

### Dir

analysis_image : app dir

FindOwn : 가상환경 dir

findown_AI : project dir

### ~~~.pkl

이미지 로고 파일의 모델별 특징 추출 파일. 키프리스에 존재하는 상표들 이미지 데이터를 모아놓은 파일을 root_dir로 지정해서 pkl로 생성이 가능하다. 현재는 구동을 위해 cnn_features_Kipris.pkl 파일만 사용한다.

### 점수 판별 방식

각 모델에서 얻어진 유사도 점수를 normalization시켜 일정하게 만든 후, 가중치를 곱해 점수를 내었다.
최종 점수가 0.72점 이상이라면 danger, 0.59점 이상이라면 warning, 0.59미만의 점수를 가진다면 safe로 판단하였다.

### 비고

1. 모든 년도의 도형 상표를 다룬다면 그 개수가 너무 많아 처리할 수 없습니다. 그래서 2018, 2019, 2020, 2021년도의 상표 데이터만을 사용하였습니다. 침해도 안전, 주의, 위험으로 출력되는 모든 이미지들은 이 4개 년도 내에 출원된 이미지들입니다.

2. 색상 유사도에 영향을 많이 줄 수 있는 데이터가 colorHistograms_logo_Kipris 파일입니다. AI_main 52행부터 68행의 주석을 풀고 사용이 가능한데, 너무 요청이 돌아오는 시간이 오래 걸려서 이 모델은 넣지 않았습니다. 추후에 가능할 것 같으면 주석 풀어서 추가하는 것도 고려하면 좋을 것 같습니다.
