---
title: Outlier Detection using Rule Based and Correlation Analysis Method
date: 2021-08-10
tags: [outlier detection]
---


- Outlier Detection in Time Series Monitoring Datasets using Rule Based and Correlation Analysis Method
- https://www.koreascience.or.kr/article/JAKO201513251334883.pdf

- 건설분야 계측 데이터
- Data fault(오류 데이터)
	- Short fault, Constant fault, Noise fault
- 이상치 탐지 기법
	- rule-based method
		- short fault 
				- 각 개별 데이터의 변화량 분석, 그 값이 일정 한계값을 벗어날 때 감지
				- 각 데이터간 1차 차분 값에 대한 히스토그램 작성
				- 일정 오차율에 의한 신뢰구간 설정, 이를 벗어날 경우 fault로 판정
		- noise fault
			- 일련의 시계열 데이터를 N개씩 일정 간격의 군(Window)으로 분류
			- 윈도우 내 값의 표준편차 또는 차분 값에 대한 히스토그램을 그림
			- 특정 단절구간을 찾아내고 한계값으로 선정, fault 판정
	- Correlation Analysis Method
		- 상호 연관성을 나타낼 수 있는 타 계측 데이터를 동시에 분석
		- 선형 최소 자승법 회귀분석 모델 이용
		- 계측데이터로 타 계측기의 결과값 예측, 실제 계측결과와 비교하면서 이상치 판정
		- 이상치로 판단하는 한계값 설정 방법
			- 최대 오차값, 신뢰구간 적용(Williams et al., 2002; Ramanathan et al., 2006; Sharma et al., 2010)
				- 최대 오차값은 오류데이터를 전혀 포함하지 않는 학습데이터 내 값을 이용하여 산정해야하므로 현실적  으로는 불가능
			
		  	
	  
- 정상적으로 증가 추이를 보이는 경우에도 rule-based method의 경우에는 이상치라고 탐지할 가능성이 높음
- 상관분석 방법을 이용할 시에도, 1차 차분 후 정상시계열의 형태로 데이터를 이용함
	-  outlier 제거하지 않고 과거데이터 모두 학습에 이용

- 후속 논문 
- https://www.koreascience.or.kr/article/JAKO202013965594859.pdf

