# 📋 House Prices: Advanced Regression Techniques

Kaggle "House Prices: Advanced Regression Techniques" 대회 데이터를 기반으로  
Ames 도시의 주택 특성을 분석하고, 주택 가격을 예측하는 회귀 모델을 구현한 프로젝트입니다.

회귀 알고리즘을 비교하고 성능 향상을 위해 하이퍼파라미터 튜닝과 스태킹을 적용하였으며  
최종 예측 결과는 Kaggle 제출 형식에 맞추어 평가하였습니다.

> 프로젝트 구조화: &nbsp;[velog.io/@jul-ee](https://velog.io/@jul-ee/DS-ML-Regression-%ED%9A%8C%EA%B7%80-%EB%AC%B8%EC%A0%9C-%EA%B5%AC%EC%A1%B0%ED%99%94)

> 🛠️ **Tech Stack**
> 
>Language: &nbsp;Python  
Data Analysis & EDA: &nbsp;pandas, numpy, Jupyter Notebook  
Visualization: &nbsp;matplotlib, seaborn  
Machine Learning:<br>- Modeling: &nbsp;`scikit-learn` (LinearRegression, Ridge, Lasso, RandomForest), `LightGBM`, `XGBoost`<br>- Model Evaluation: &nbsp;`scikit-learn` (cross_val_score, GridSearchCV, mean_squared_error)

<br>
<br>

## 프로젝트 개요

- 데이터: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Ames Housing Dataset (Train: 1460건, Test: 1459건)
- 타겟 변수: &nbsp;`SalePrice` (주택 판매가)
- 예측 목표: &nbsp;RMSE (Root Mean Squared Error) 최소화
- 주요 기법: &nbsp;로그 변환, One-Hot Encoding, 모델 튜닝, 스태킹

<br>
<br>

## 목차

1. [데이터 로드 및 확인](#1-데이터-로드-및-확인)  
2. [데이터 전처리](#2-데이터-전처리)  
3. [모델 학습 및 평가 함수 정의](#3-모델-학습-및-평가-함수-정의)  
4. [모델 비교 및 성능 평가](#4-모델-비교-및-성능-평가)  
5. [하이퍼파라미터 튜닝](#5-하이퍼파라미터-튜닝)  
6. [Fine-tuning](#6-Fine-tuning)  
7. [Ensemble Stacking](#7-Stacking)  
8. [Kaggle 제출](#8-Submission)

<br>
<br>

## 모델링 과정

#### 1. 데이터 로드 및 확인

- 라이브러리 불러오기 및 데이터 구조 확인
- 결측치 및 이상치 확인
- 범주형/수치형 변수 분류


#### 2. 데이터 전처리

- 타겟 변수(`SalePrice`)에 로그 변환 적용  
- 수치형 결측치 평균값 대체
- One-Hot Encoding 적용  
- 이상치 제거 및 Train/Test 분리

#### 3. 모델 학습 및 평가 함수 정의

- RMSE, RMSLE 등 회귀 성능 지표 계산 함수 정의  
- 학습 및 예측 결과 비교 함수 구현  
- 회귀 계수 시각화 및 중요 피처 비교

<br>

#### 4. 모델 비교 및 성능 평가

| Model            | RMSLE | RMSE     | MSE      |
| ---------------- | ----- | -------- | -------- |
| **Lasso**        | 0.104 | 22024.99 | 4.85e+08 |
| **Ridge**        | 0.121 | 20997.94 | 4.41e+08 |
| LinearRegression | 0.128 | 21618.35 | 4.67e+08 |
| LightGBM         | 0.114 | 25604.36 | 6.56e+08 |
| RandomForest     | 0.120 | 27645.77 | 7.64e+08 |
| XGBoost          | 0.129 | 27766.77 | 7.71e+08 |

- RMSLE 기준 최고 성능 모델은 Lasso(alpha=0.01) &nbsp;→  &nbsp;RMSLE: 0.104
- RMSE 기준 최고 성능 모델은 Ridge(alpha=50)  &nbsp;→ &nbsp;RMSE: 20,997.939
- 트리 기반 모델(LightGBM, RandomForest, XGBoost)은 튜닝 전에는 전반적으로 선형 모델보다 낮은 성능을 보임

> **Insight**  
> 선형 회귀 계열(Lasso, Ridge)이 기본 성능에서는 우위  
> 트리 계열은 하이퍼파라미터 튜닝 또는 앙상블 적용 시 기대 효과 있음

<br>

#### 5. 하이퍼파라미터 튜닝

- 목표: 기본 성능이 우수한 선형 모델보다 더 낮은 RMSE를 달성하기 위해, 트리 기반 모델의 성능을 향상시키는 것
- GridSearchCV, RandomizedSearchCV 를 활용하여 LightGBM 및 XGBoost 모델의 주요 하이퍼파라미터(n_estimators, learning_rate, max_depth 등) 최적화
- 앙상블 기반 모델(Stacking)의 성능 향상에 기여하기 위한 트리 모델의 최적 파라미터를 찾음

#### 6. Fine-tuning

- 튜닝된 LightGBM, XGBoost 모델로 성능 개선  
- 하이퍼파라미터 조합 수동 조정 실험 과정은 따로 기록

#### 7. Stacking

- 목표: 개별 모델보다 향상된 예측 성능 확보
- Base Learner: &nbsp;Ridge, Lasso, LinearRegression, LightGBM, XGBoost, RandomForest  
- Meta Model: &nbsp;Lasso (XGBoost에서는 성능 하락 관찰)

#### 8. Submission

- 최종 예측 결과를 로그 역변환하여 `submission.csv` 파일 저장  
- Kaggle 기준에 맞춘 평가로 제출 완료

<br>
<br>

## 최종 성능 및 선택 모델

| Model        | 방식             | 주요 내용                        | 결과 (RMSLE / RMSE) |
|------------------|------------------|----------------------------------|-------------|
| Lasso            | 단일 모델        | 로그 변환 + 기본 하이퍼파라미터 | 0.104 &nbsp;/ &nbsp;22024.99    |
| XGBoost         | 튜닝 모델        | GridSearchCV 적용               |  0.195 &nbsp;/ &nbsp;23534.75     |
| Stacking         | 앙상블           | 메타모델: Lasso                 | 0.099 &nbsp;/ &nbsp;18542.13 |

- 최종 선택 모델: &nbsp;스태킹 기반 앙상블 (Lasso 메타모델)  
- 성능 지표
  - RMSLE: *0.0990*
  - RMSE: *18542.13*
  - MAE: *12170.05*

튜닝 및 스태킹을 통해 선형 회귀보다 낮은 RMSE를 달성하여 최종 제출 모델로 선정하였다.  

Kaggle에 submission한 최초 score는 0.12365를 기록하였다.

<br>
<br>

## 인사이트 및 회고

결측치 처리, 로그 변환, 이상치 제거, 파생 피처 생성 등 전처리 방향을 어떻게 잡느냐에 따라 모델 성능이 눈에 띄게 향상하거나 하락하는 것을 보았다. 모델의 성능이 이 초기 판단에서 갈린다는 것을 실험을 통해 확인하였고, 문제의 목적에 맞게 데이터를 충분히 이해하고 상황에 맞는 전처리 전략을 세우는 것의 중요성을 다시 느꼈다. 

해당 문제에서는 선형 회귀 계열 모델이 baseline에서 가장 안정적인 성능을 보여주었다.
트리 기반 모델은 초기 성능이 다소 낮았으나 튜닝을 통해 충분히 개선 가능함을 확인하였다.
Stacking을 적용했을 때 개별 모델보다 RMSE가 더 낮아졌으며 메타 모델로는 Lasso가 적합하다고 판단하였다.

모델 성능은 높였지만 feature selection이나 Bagging, Boosting 등 다른 앙상블 방식도도 추가적으로 실험해 볼 여지가 있다.

회귀 문제를 해결하기 위한 하나의 방향성을 정리할 수 있었다. 해당 프로젝트에서 정의한 함수들과 실험을 구조화해 봄으로써 추가적인 실험을 설계에 인사이트가 될 것 같다. 이후 새로운 데이터셋에서 다른 목적을 가지고 회귀 모델을 구현할 때에도 이번에 구조화한 내용을 바탕으로 고도화해 나간다면 다양한 상황에서 문제를 해결할 수 있는 능력을 기를 수 있을 것이다.

<br>
<br>

> 🔗 [Kaggle "House Prices: Advanced Regression Techniques" 대회](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)


