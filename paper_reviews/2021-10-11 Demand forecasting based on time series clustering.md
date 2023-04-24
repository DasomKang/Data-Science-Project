
## About

- Paper: [A study on electricity demand forecasting based on time series clustering in smart grid](https://scienceon.kisti.re.kr/srch/selectPORSrchArticle.do?cn=JAKO201611059007425&dbt=NART)
- keywords: time series clustering, demand forecasting
- date: 2021-10-11


---
## Abstract
- 시계열 기반 군집분석을 통한 군집별 예측량의 결합 
- Time series clustering methods
	- Periodogram-based normalized clustering, predictive analysis clustering, DTW(Dynamic TIme Warping)
- Time series prediction
	- DSHW(Double Seasonal Holt-Winters), TBATS(Trigonometric, Box-Cox transform, ARMA errors, Trend and Seasonal components)모형, FARIMA(Fractional ARIMA)
- 결론 : 전체 사용량을 기반으로 예측하는 방식이 아닌 군집분석을 통한 군집별 예측량의 결합이 더 낮은 MAPE로 나타남에 따라 우수한 예측 방법으로 판단
  


## Summary

### 수요 예측을 위한 다양한 접근법 
- SOM, Kmeans로 요일별, 시간별 자료 군집화 후 exponential smoothing, arima 
- exponential smoothing을 포함한 regression과 신경망
- SVR(support vector regression)에 기반한 전력 수요 예측
- 계측적 군집분석 이용
- 오차분해 후 
- 실질국민소득과 에너지수요에 대하여 VEC모형 추정 및 Granger causality model 실시
- ARIMA 를 이용, 천연가스 수요에서 단기, 장기 요금 및 수입탄력성 예측 
- ARMAX, GARCH이용,  일일전력수요와 관광효과에 대한 연구 실시 
- hybrid non linear model인 SEGARCH-ANN, WARCH-ANN 
- 이중 계절성, 삼중 계절성
- BATS
	- 복잡한 주기성 고려한 일반화 모형 
	- TBATS, 주기성을 삼각함수의 합으로 나타냄
	
	
### real data analysis
- 데이터
	- 86개 기업의 전력 사용량 (결측값 없음)
	- 시간별 데이터
- 전력 수요 예측 방법
	- 요약 : 86개 기업의 사용량 합계에 대한 예측값과 군집별 예측의 합계와 비교
	- 내용
		- Periodogram-based normalized clustering 으로 10개 cluster 생성
		- cluster별 DSHW, TBATS, FARIMA 모형 적용
			- 이중 계절형 : 24(일별), 168(주)
		- 평가 방법 : MAPE
		- 전체 사용량의 예측(DSHW)에 대한 MAPE와 군집별 예측합 비교

