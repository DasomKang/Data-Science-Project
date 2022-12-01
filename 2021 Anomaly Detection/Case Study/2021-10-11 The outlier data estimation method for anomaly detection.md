---
title: reading abstracts 3 | the outlier data estimation method for anomaly detection 
date: 2021-10-12
tags: [anomaly detection, DTW]
---

- Paper : [A study on the outlier data estimation method for anomaly detection of photovoltaic system](https://www.koreascience.or.kr/article/JAKO202019962560548.pdf)
    
**abstracts**
- 파라미터의 상관관계 도출 후 k-nearest neighbor로  정상/비정상 데이터 분류
- 발전 시스템 결함에 의한 outlier, 구름 등에 의한 단기간 발생하는 부분 음영, 전체음영의 일시적인 전력 손실

  ---
  
  
**데이터**
- 변수
	- 태양광 발전 데이터 : 현재발전(인버터)전력, 누적발전량, 패널전력, 패널전류
	- 기상청 데이터 : 기온, 강수량, 풍속, 풍향
	- 환경센서 데이터 : 경사면일사량, 수평면일사량, 패널온도
- 8시 ~ 21시 데이터 이용 (발전시간)
- 연속 10회 결측 시, 해당 시각 전/후 값 활용해 선형보간법의 방법으로 대체
	
	
	
**real data analysis**
1. 태양광 발전량과 다른 변수들 간 상관관계 분석 
	1.  일사량, 패널 온도 변수와 상관관계 높음
2. 태양광 발전량 알고리즘 산정 및 이상치 후보군 분류
		1. 실제 발전량과 위 알고리즘으로 계산한 발전량 비교
			1. 발전 오차값 10kW 이하 구간 : 태양광 발전의 설비효율 손실
			2. 발전 오차값 20kW~30kW 구간 : 주요 이상감지 구간
			3. 발전 오차값 40kW 초과 구간 : 구름 등에 의한 일시적 발전 저하
3. K-Nearest Neighbor로 정상/이상 데이터 분류
4. 이상데이터 대상 경사면일사량과 패널전력의 DTW 거리 알고리즘 적용, 최종 outlier 판별
	1. DTW가 클수록 데이터 유사성이 작다는 것을 의미, 서로 다른 패턴의 형태를 보여줌


**memo**
- 태양광 발전량 알고리즘을 상관관계가 높은 변수(0.9 이상)로 만들어서 예측치를 얻을 수 있는 데이터 셋이었음.