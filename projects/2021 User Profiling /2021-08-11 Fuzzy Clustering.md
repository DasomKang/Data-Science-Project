## Fuzzy Clustering
https://rfriend.tistory.com/230
- Fuzzy Clustering은 Partitional Clustering 중에서 Prototype-based기법
- 각 관측치가 여러 군집에 속할 수 있으며, 각 군집에 속할 가능성, 확률을 제시
- Soft Clustering이라고도 함.

### 1. fuzzy algorithm
1) Fuzzy C-means(FCM) Clustering Algorithm
- k-means 와 유사, 군집 내 관측치들간의 유사성을 최대화하고, 군집 간 비유사성을 최대화하는 optimal solution 찾아가는 방식
- k개 군집을 사전적으로 정해줘야함.
- 유사성의 측도로 distance 사용

### 2. research
[Fuzzy Clustering 기반의 화재 상황 인식 모델](https://scienceon.kisti.re.kr/commons/util/originalView.do?cn=JAKO201125736639827&oCn=JAKO201125736639827&dbt=JAKO&journal=NJOU00423616)
- summary
	-  여러 인자들 간의 조합에 의한 규칙을 생성, 불명확한 데이터 처리가 가능한 퍼지추론을 사용하여 화재상황을 인식하는 방식을 제안
	-  퍼지추론 방식에서 지식의 일반화, 형식화의 문제점을 해결하기 위해, 화재의 특정 패턴들의 특징을 찾아서 분석하고 규칙베이스를 구축함으로써 시스템의 성능을 더욱 향상
	-  화재의 레벨을 3단계(정상, 주의, 위험) 로 나누고, 각 단계별로 훈련데이터를 FCM(fuzzy C-means clustering)에 의해 규칙화 하여 추론하는 시스템을 제안
- fuzzy inference system
	- 기존의 수학적인 시스템 모델링 에서는 잘 나타낼 수 없는 불확실한 정보와 복잡하고 잘 정의되지 않는 시스템을 if-then 형태의 규칙을 이용하는 형태를 보이는 추론 시스템
	-  입력변수의 소속 함수, 퍼지규칙, 출력변수의 소속 함수로 구성

[Application of the C-Means Fuzzy Clustering Method for the Patient’s State Recognition Problems in the Medical Monitoring Systems](http://ceur-ws.org/Vol-2362/paper20.pdf)
refer

- https://www.researchgate.net/publication/4309998_Similarity-Based_Fuzzy_Clustering_for_User_Profiling
