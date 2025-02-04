# SKN07-3rd-1Team

# 3차 프로젝트

## 1. 팀 소개
- 팀명
  <table>
  <tr>
    <th>김나예</th>
    <th>김서진</th>
    <th>유수현</th>
    <th>정승연</th>
   
  </tr>
  <tr>
    <td><img src="" width="175" height="175"></td>
    <td><img src= "" width="175" height="175"></td>
    <td><img src="" width="175" height="175"></td>
    <td><img src="" width="175" height="175"></td>
  </tr>
  <tr>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
  </tr>
  </table>

- 멤버 개인 깃허브 계정과 연동

---
 
## 2. 프로젝트 개요

### 2.1 프로젝트 명

지역별 숙박 정보 제공 및 맞춤 추천

### 2.2 프로젝트 소개

> 작다는 것 그 이상

적은 데이터지만, LLM과 함께 보다 상세한 정보를 제공합니다.<br>
GPT도 정확히 알 수 없는 부대시설, 객실 어메니티 정보를 알려주고,<br>
세세한 요구사항에도 맞춤형 숙박업소를 추천을 해주는 질의응답 시스템입니다.


### 2.3 프로젝트 필요성 및 배경
  
관광 산업은 전 세계적으로 중요한 경제 활동 중 하나로 그 중에서도 숙박은 필수적인 요소로 자리 잡고 있습니다. 여행 계획을 세울 때 가장 많은 시간을 소비하는 부분 중 하나가 숙박시설을 찾는 것이기에 사용자들에게 정확하고 빠른 정보를 제공함으로 시간과 노력을 절약할 수 있다는 점에서 사용자들에게 편리함을 줄 수 있습니다. 또한 인공지능과 자연어 처리 기술의 발전을 이러한 정보 검색 과정에 도입한다면 더 편리하게 사용자가 원하는 맞춤형 숙박 추천을 해줄 수 있어 여행에 대한 만족도가 더 높아질 수 있습니다.

  
### 2.4 프로젝트 목표

- 한국관광공사 데이터와 **LLM을 연동한 질의 응답 시스템** 구축
- 편리하고 직관적인 **UI 인터페이스**
- **RAG 방식**을 접목해 보다 구체적이고 정확한 답변 제공

---
## 3. 수행 과정

### 3.1 데이터 수집
- Open API 활용 : TourAPI 4.0 

- 전체 숙소 리스트를 가져오는 api로 숙소의 명칭과 contentid, contenttypeid 리스트화 (infoDF)
``` python
import requests
from bs4 import BeautifulSoup
import pandas as pd
mykey = '비밀'
url = f'http://apis.data.go.kr/B551011/KorService1/searchStay1?areaCode=&sigunguCode=&ServiceKey={mykey}&listYN=Y&MobileOS=ETC&MobileApp=AppTest&arrange=A&numOfRows=3900&pageNo=1'
rt = requests.get(url)
items = BeautifulSoup(rt.text).select('item')

data = []
for x in items:
    title = x.find('title').text
    contentid = x.find('contentid').text
    contenttypeid = x.find('contenttypeid').text
    
    data.append({'title': title, 'contentid': contentid, 'contenttypeid' : contenttypeid})

infoDF = pd.DataFrame(data)
``` 
- infoDF 를 사용해 개별 숙소의 기본정보, 숙소정보, 객실정보 조회 api를 사용해 데이터 취합 및 DF 작성
``` python
# 데이터 취합
infoData = []
from tqdm import tqdm
for x in tqdm(range(3887)):
    # infoDF 데이터 사용
    title = infoDF.loc[x]['title']
    contentid = infoDF.loc[x]['contentid']
    contenttypeid = infoDF.loc[x]['contenttypeid']

    # 기본정보 조회함수 : 주소, 설명, 이미지링크
    addr1, overview, imglilk = selectInfo2(contentid, contenttypeid)
    # 숙소정보 조회함수 : 숙소의 전제적인 정보
    txtinfo = sukso_info(contentid, contenttypeid)
    # 객실정보 조회함수 : 숙소에 속하는 객실별 정보 리스트로 get
    all_room_info = room_info(contentid, contenttypeid)

    infoData.append({'title': title, 'addr1': addr1, 'overview' : overview, 'imglilk' : imglilk, 'txtinfo' : txtinfo, 'all_room_info' : all_room_info})

# DF 
suk_data = []
for x in infoData:
    name = x['title']
    address = x['addr1']
    overview = x['overview']
    imglink = x['imglilk']
    generalInfo = x['txtinfo']
    roomInfo = x['all_room_info']
    
    
    suk_data.append({'name': name, 'address': address, 'overview' : overview, 'imglink' : imglink, 'generalInfo' : generalInfo, 'roomInfo' : roomInfo})

suksoDF = pd.DataFrame(suk_data)

``` 

- 취합 데이터의 오류 정정 및 가중치 부여를 위한 tag 작성 
  - 지역을 tag에 포함
  ``` python
  # 태그 추출 함수 (최적화)
  def extract_tag(row):
      # 주소 추출
      road_address = row['address']
      
      # 주소가 유효하고 2개 이상의 단어로 구성된 경우 사용
      if pd.notna(road_address) and len(road_address.split()) >= 2:
          return ' '.join(road_address.split()[:2])

      
      # 둘 다 유효하지 않을 경우
      return '없음'

  suksoDF['tag'] = suksoDF.apply(lambda row: extract_tag(row), axis=1)

  ```

  - 영어주소 -> 한글 변경 
  ``` python
  suksoDF['tag'].unique() # 잘못 입력된 내용 확인 후 
  # 주소 오류 수기 수정
  suksoDF.loc[suksoDF['address'].str.contains('18, Hoegi-ro', na=False), 'address'] = '서울특별시 동대문구 회기로 29길 18'
  suksoDF.loc[suksoDF['tag'].str.contains('18, Hoegi-ro', na=False), 'tag'] = '서울특별시 동대문구'

  ```

  - 태그 강화를 위해 지역명 tag 추가 
  ``` python
  suksoDF.loc[suksoDF['tag'].str.contains('강원도', na=False), 'tag'] = suksoDF['tag'] + ' 강원특별자치도'
  suksoDF.loc[suksoDF['tag'].str.contains('강원특별자치도', na=False), 'tag'] = suksoDF['tag'] + ' 강원도'
  suksoDF.loc[suksoDF['tag'].str.contains('경기도', na=False), 'tag'] = suksoDF['tag'] + ' 경기'
  suksoDF.loc[suksoDF['tag'].str.contains('경상남도', na=False), 'tag'] = suksoDF['tag'] + ' 경상도'
  suksoDF.loc[suksoDF['tag'].str.contains('경상북도', na=False), 'tag'] = suksoDF['tag'] + ' 경상도'
  suksoDF.loc[suksoDF['tag'].str.contains('광주광역시', na=False), 'tag'] = suksoDF['tag'] + ' 광주'
  suksoDF.loc[suksoDF['tag'].str.contains('대구광역시', na=False), 'tag'] = suksoDF['tag'] + ' 대구'
  suksoDF.loc[suksoDF['tag'].str.contains('대전광역시', na=False), 'tag'] = suksoDF['tag'] + ' 대전'
  suksoDF.loc[suksoDF['tag'].str.contains('부산광역시', na=False), 'tag'] = suksoDF['tag'] + ' 부산'
  suksoDF.loc[suksoDF['tag'].str.contains('서울특별시', na=False), 'tag'] = suksoDF['tag'] + ' 서울'
  suksoDF.loc[suksoDF['tag'].str.contains('세종특별자치시', na=False), 'tag'] = suksoDF['tag'] + ' 세종'
  suksoDF.loc[suksoDF['tag'].str.contains('울산광역시', na=False), 'tag'] = suksoDF['tag'] + ' 울산'
  suksoDF.loc[suksoDF['tag'].str.contains('인천광역시', na=False), 'tag'] = suksoDF['tag'] + ' 인천'
  suksoDF.loc[suksoDF['tag'].str.contains('전라남도', na=False), 'tag'] = suksoDF['tag'] + ' 전라도'
  suksoDF.loc[suksoDF['tag'].str.contains('전북특별자치도', na=False), 'tag'] = suksoDF['tag'] + ' 전라북도 전라도'
  suksoDF.loc[suksoDF['tag'].str.contains('제주특별자치도', na=False), 'tag'] = suksoDF['tag'] + ' 제주도 제주'
  suksoDF.loc[suksoDF['tag'].str.contains('충청남도', na=False), 'tag'] = suksoDF['tag'] + ' 충청도'
  suksoDF.loc[suksoDF['tag'].str.contains('충청북도', na=False), 'tag'] = suksoDF['tag'] + ' 충청도'

  ```

  - 소개문구내 특정 문구 추출해 tag입력 (시간 부족으로 인해 추가 정리 필요: '오션뷰', '바다', '도심지' 등의 키워드)
  ``` python
  suksoDF.loc[suksoDF['overview'].str.contains('한옥', na=False), 'tag'] = suksoDF['tag'] + ' 한옥'

  ```
- 위 작성 데이터 저장 
``` python
suksoDF.to_csv('./data/suksoDF.csv', index=False, encoding='utf-8')
```

### 3.2. 데이터 로드 및 텍스트 분할

### 3.3. 데이터 벡터화 및 저장

### 3.4 프롬프트 작성

### 3.5 Streamlit을 이용한 웹 구현 
streamlit을 사용하여 웹을 만들기 

Langchain을 이용하여 질문을 입력하면 답변이 나오는 챗봇

Markdown을 이용하여 챗봇의 대화 기록을 깔끔하게 표시하고 대화창을 디자인

---
## 4. 기술 Stack
 - ![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
 - ![LangChain](https://img.shields.io/badge/LangChain-0.3.7-orange)
 - ![Chroma](https://img.shields.io/badge/Chroma-Vector%20DB-0091FF?style=flat&logo=pinecone&logoColor=white)
 - ![OpenAI GPT-3.5 turbo](https://img.shields.io/badge/OpenAI-GPT--3.5--turbo-blueviolet?logo=openai&logoColor=white)
 - ![OpenAI GPT-4](https://img.shields.io/badge/OpenAI-GPT--4-blueviolet?logo=openai&logoColor=white)
 - ![Streamlit](https://img.shields.io/badge/Streamlit-1.39.0-red?logo=streamlit&logoColor=white)

---
## 5. 수행 결과(테스트/시연 페이지)

--- 
## 한 줄 회고
- 김나예:
- 김서진:
- 유수현:
- 정승연:
