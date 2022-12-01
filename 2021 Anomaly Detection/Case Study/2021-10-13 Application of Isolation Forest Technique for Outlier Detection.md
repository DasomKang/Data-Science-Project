---
title: reading abstracts 5 | Application of Isolation Forest Technique for Outlier Detection
date: 2021-10-13
tags: [Isolation Forest, Anomaly Detection]
---


- Paper : [수질자료의 이상치 탐색을 위한 Isolation Forest기법의 적용](https://www.jksee.or.kr/upload/pdf/KSEE-2018-40-12-473.pdf)
  
  
**abstracts**
- 통계적 상관관계 유무에 따라 집단을 분류
- 상관관계가 유의미한 집단에 대해서는 거리기반 및 Isolation Forest 기법 적용 후 성능 비교
	- 결과 : Isolation Forest 기법이 거리기반 기법에 비해 보다 넓은 범위에서 이상치들을 탐색
- 상관관계가 유의미하지 않은 집단에서는 Isolation Forest 기법 적용 후 기계학습의 변화에 따른 이상치 탐색 성능을 분석(?) 
	- 결과 : 기계학습량의 변화에 따른 이상치 탐색 성능의 변화는 미미



**이상치 탐색 기법**
- 단변량
	- Discordancy Testing : Dixon's test, Discordance test, Rosner's test. Walsh's test 
- 다변량
	- 분산기반 접근 : 자료들의 집합에서 어떤 개체를 제거할 때 분산이 최소화되는 경우 해당 개체를  이상치로 정의,  smoothing factor계산하여 이상치 탐색
	- 거리기반 접근 : 자료 내 개체들 사이의 유클리드 거리 또는 마할라노비스 거리(mahalanobis distance) 를 바탕으로 정상 자료의 군집에서 멀리 떨어져있는 경우 이상치로 판단
	- 밀도기반 접근  : 자료들의 밀도를 나타내는 Local Outlier Factor(LOF)를 이용해 한 데이터 개체를 대상으로 그 점과 이웃한 개체들의 LOF를 비교하여 이상치 판단
	- 분류모델기반 접근 : 자기연상 학습 신경망(Auto-associative neural network)와 같은 지도학습을 통해 학습데이터로 분류모델 생성, 이상치 탐색



**연구**
- 모수기반의 다변량 이상치 탐지기법 의 단점을 보완하고자 분류모델 기반의 다변량 이상치 탐 색기법인 Isolation Forest (iForest) 기법을 수질자료에 적용
