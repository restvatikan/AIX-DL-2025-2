# AI+X: Deep Learning 2025-2
AI+X: 딥러닝 2025-2 기말 프로젝트

-----

정보시스템학과 2024062042 김규민 oplgk0576@hanyang.ac.kr    
정보시스템학과 2023092606 송정빈 happysongjb@hanyang.ac.kr

-----
# 한국어 영화 리뷰 감성 분석 : 머신러닝과 딥러닝 모델 성능 비교 및 개선

본 프로젝트는 대표적인 한국어 감성 분석 데이터셋인 **NSMC(Naver Sentiment Movie Corpus)**를 활용하여, 다양한 머신러닝(ML) 및 딥러닝(DL) 방법론을 적용하고 그 성능을 비교 분석합니다.

궁극적인 목표는 각 모델의 특성을 이해하고, 성능 개선 시도를 통해 한국어 텍스트 처리에 대한 실질적인 인사이트를 도출하는 것입니다. 모든 코드는 **Jupyter Notebook(.ipynb)**으로 작성되어 손쉬운 재현(Reproducibility)을 목표로 하고 있습니다.

-----

## 목차

1.  [프로젝트 개요](https://www.google.com/search?q=%231-%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-%EA%B0%9C%EC%9A%94)
2.  [사용 데이터셋](https://www.google.com/search?q=%232-%EC%82%AC%EC%9A%A9-%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%85%8B-nsmc)
3.  [실험 환경 및 제약사항](https://www.google.com/search?q=%233-%EC%8B%A4%ED%97%98-%ED%99%98%EA%B2%BD-%EB%B0%8F-%EC%A0%9C%EC%95%BD%EC%82%AC%ED%95%AD)
4.  [프로젝트 구조](https://www.google.com/search?q=%234-%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-%EA%B5%AC%EC%A1%B0)
5.  [적용 방법론](https://www.google.com/search?q=%235-%EC%A0%81%EC%9A%A9-%EB%B0%A9%EB%B2%95%EB%A1%A0)
      * [5.1. 텍스트 전처리 및 벡터화](https://www.google.com/search?q=%2351-%ED%85%8D%EC%8A%A4%ED%8A%B8-%EC%A0%84%EC%B2%98%EB%A6%AC-%EB%B0%8F-%EB%B2%A1%ED%84%B0%ED%99%94)
      * [5.2. 머신러닝 (Shallow Learning) 모델](https://www.google.com/search?q=%2352-%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-shallow-learning-%EB%AA%A8%EB%8D%B8)
      * [5.3. 딥러닝 (Deep Learning) 모델](https://www.google.com/search?q=%2353-%EB%94%A5%EB%9F%AC%EB%8B%9D-deep-learning-%EB%AA%A8%EB%8D%B8)
      * [5.4. 앙상블 (Ensemble) 기법](https://www.google.com/search?q=%2354-%EC%95%99%EC%83%81%EB%B8%94-ensemble-%EA%B8%B0%EB%B2%95)
6.  [실험 과정 및 결과](https://www.google.com/search?q=%236-%EC%8B%A4%ED%97%98-%EA%B3%BC%EC%A0%95-%EB%B0%8F-%EA%B2%B0%EA%B3%BC)
      * [6.1. 평가 지표](https://www.google.com/search?q=%2361-%ED%8F%89%EA%B0%80-%EC%A7%80%ED%91%9C)
      * [6.2. 모델별 성능 비교](https://www.google.com/search?q=%2362-%EB%AA%A8%EB%8D%B8%EB%B3%84-%EC%84%B1%EB%8A%A5-%EB%B9%84%EA%B5%90)
7.  [성능 개선 시도](https://www.google.com/search?q=%237-%EC%84%B1%EB%8A%A5-%EA%B0%9C%EC%84%A0-%EC%8B%9C%EB%8F%84)
8.  [결론 및 고찰](https://www.google.com/search?q=%238-%EA%B2%B0%EB%A1%A0-%EB%B0%8F-%EA%B3%A0%EC%B0%B0)
9.  [프로젝트 실행 방법](https://www.google.com/search?q=%239-%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-%EC%8B%A4%ED%96%89-%EB%B0%A9%EB%B2%95)

-----

## 1\. 프로젝트 개요

자연어 처리(NLP) 분야에서 감성 분석(Sentiment Analysis)은 텍스트에 담긴 감정(긍정/부정)을 판별하는 핵심 기술입니다. 특히 E-commerce, 마케팅, 여론 분석 등 다양한 산업에서 그 중요성이 대두되고 있습니다.

본 프로젝트는 한국어 감성 분석의 표준 벤치마크 중 하나인 **NSMC** 데이터를 사용하여, 전통적인 머신러닝 기법부터 딥러닝 모델에 이르기까지 **"얕고 넓은(Shallow and Wide)"** 범위의 방법론을 적용합니다.

각 모델의 성능을 **정량적**으로 비교하고, 하이퍼파라미터 튜닝 및 앙상블 기법을 통해 **성능 향상**을 시도한 과정을 상세히 기록합니다.

## 2\. 사용 데이터셋 (NSMC)

  * **데이터셋:** [NSMC (Naver Sentiment Movie Corpus)](https://github.com/e9t/nsmc)
  * **구성:**
      * `ratings_train.txt`: 150,000개 (학습 데이터)
      * `ratings_test.txt`: 50,000개 (테스트 데이터)
  * **특징:**
      * 각 데이터는 **id, document(리뷰 텍스트), label(0: 부정, 1: 긍정)** 3개의 필드로 구성됩니다.
      * 리뷰의 길이가 매우 짧거나(단답형), 중복 데이터 및 결측치가 존재하여 **전처리(Preprocessing)** 과정이 필수적입니다.

> **[데이터 전처리 예시]**
>
>   * `document` 필드의 결측치(NaN) 제거
>   * 중복된 리뷰 텍스트 제거
>   * 한국어 외 문자 (특수문자, 영어, 한자) 제거 (선택적)
>   * 형태소 분석기(Tokenizer)를 사용한 토큰화

## 3\. 실험 환경 및 제약사항

  * **실행 환경:** `Google Colab` 또는 `Jupyter Notebook`
  * **주요 라이브러리:**
      * `Pandas`, `Numpy`
      * `Scikit-learn` (ML 모델 및 전처리)
      * `TensorFlow` / `Keras` (DL 모델)
      * `Matplotlib`, `Seaborn` (시각화)
  * **제약사항 (Constraint):**
      * **Java 비의존성:** Google Colab 런타임에서 발생할 수 있는 **Java 호환성 문제**를 원천적으로 차단하기 위해, `Konlpy`의 `Okt`, `Kkma` 등 Java 기반 형태소 분석기 사용을 **지양**합니다.
      * **대안:** C++ 기반의 `Mecab` (Colab 설치 필요) 또는 순수 Python으로 구현된 토크나이저, 혹은 `SentencePiece`와 같은 서브워드 토크나이저 사용을 권장합니다. 본 프로젝트에서는 `Mecab-ko-mecab` (pip 설치 가능) 또는 간단한 공백 기반 분리를 사용합니다.

## 4\. 프로젝트 구조

```bash
.
├── README.md                 # 프로젝트 개요 (본 파일)
├── data/
│   ├── ratings_train.txt     # 학습 데이터
│   └── ratings_test.txt      # 테스트 데이터
├── notebooks/
│   ├── 1_Data_Preprocessing.ipynb  # 데이터 로드 및 전처리
│   ├── 2_ML_Models.ipynb         # TF-IDF 및 ML 모델 (로지스틱, 결정트리)
│   ├── 3_DL_Model_CNN.ipynb    # Keras Embedding 및 CNN 모델
│   └── 4_Ensemble_Analysis.ipynb # 앙상블 및 최종 성능 비교
└── requirements.txt            # 필요 라이브러리 목록
```

## 5\. 적용 방법론

### 5.1. 텍스트 전처리 및 벡터화

텍스트 데이터를 모델이 이해할 수 있는 숫자형 벡터로 변환합니다.

  * **형태소 분석 (Tokenization):** 위에서 언급한 Java 비의존적 토크나이저 (예: `Mecab`)를 사용합니다.
  * **불용어(Stopwords) 제거:** 의미 없는 조사, 어미, 부사 등을 제거합니다.
  * **벡터화 (Vectorization):**
      * **TF-IDF (Term Frequency-Inverse Document Frequency):** ML 모델을 위한 특성 추출. 단어의 빈도와 문서 빈도의 역수를 활용하여 단어의 중요도를 계산합니다.
      * **Word Embedding:** DL 모델을 위한 특성 추출. `Keras`의 `Embedding` 레이어를 사용하여 단어를 밀집 벡터(Dense Vector)로 표현합니다.

### 5.2. 머신러닝 (Shallow Learning) 모델

`Scikit-learn` 라이브러리를 기반으로 하며, **TF-IDF**로 벡터화된 데이터를 입력으로 받습니다.

  * **로지스틱 회귀 (Logistic Regression):**
      * 선형 모델 기반의 이진 분류기. 빠르고 해석이 용이하며 강력한 베이스라인 성능을 제공합니다.
  * **결정 트리 (Decision Tree):**
      * 트리 구조를 기반으로 데이터를 분류. 과적합(Overfitting) 경향이 있으나 직관적입니다.
  * **랜덤 포레스트 (Random Forest):**
      * (앙상블 섹션에서 다룰 수 있으나, 단일 모델로도 분류) 배깅(Bagging) 기반의 앙상블 모델. 결정 트리의 과적합을 완화합니다.

### 5.3. 딥러닝 (Deep Learning) 모델

`TensorFlow/Keras`를 기반으로 하며, **Word Embedding**을 입력으로 받습니다.

  * **1D CNN (Convolutional Neural Network):**
      * 텍스트의 지역적 특징(Local Feature)을 추출하는 데 효과적입니다. (예: "정말 재밌")
      * 합성곱 필터(Filter)가 텍스트를 스캔하며 특징 맵(Feature Map)을 생성하고, 이를 풀링(Pooling)하여 최종적으로 감성을 분류합니다.
        
### 5.4. 앙상블 (Ensemble) 기법

여러 개의 기본 모델(Base Models)을 결합하여 단일 모델보다 강력한 성능을 도출합니다.

  * **Voting:**
      * 여러 모델(예: 로지스틱 회귀, CNN)의 예측 결과를 다수결(Hard Voting) 또는 확률 평균(Soft Voting)으로 합산하여 최종 예측을 결정합니다.
  * **Stacking:**
      * 1단계 모델들의 예측 결과를 다시 학습 데이터로 사용하여, 2단계 모델(Meta-learner)이 최종 예측을 학습하는 방식입니다.

## 6\. 실험 과정 및 결과

### 6.1. 평가 지표

본 데이터셋은 레이블(0, 1)이 비교적 균형 잡혀 있으므로, **정확도(Accuracy)**를 메인 지표로 사용합니다.

$$\text{Accuracy} = \frac{\text{True Positives (TP)} + \text{True Negatives (TN)}}{\text{Total Data}}$$

### 6.2. 모델별 성능 비교

`ratings_test.txt` (50,000개)에 대한 최종 성능 비교표입니다. (아래 표는 아직 placeholder입니다다)

| 모델 (Model) | 특성 추출 (Feature Extraction) | 정확도 (Accuracy) | F1-Score | 비고 (Notes) |
| :--- | :--- | :---: | :---: | :--- |
| **Baseline (ML)** | | | | |
| 로지스틱 회귀 | TF-IDF (n-gram=1,3) | 85.1% | 0.850 | 하이퍼파라미터 튜닝 (C=1) |
| 결정 트리 | TF-IDF (n-gram=1,3) | 72.5% | 0.723 | 과적합 경향 확인 |
| 랜덤 포레스트 | TF-IDF (n-gram=1,3) | 82.0% | 0.819 | |
| **Deep Learning** | | | | |
| 1D CNN | Keras Embedding (100d) | 84.8% | 0.847 | 10 Epochs, Dropout=0.5 |
| **Ensemble** | | | | |
| 앙상블 (Voting) | (LR + CNN) Soft Vote | **85.9%** | **0.858** | 가장 우수한 성능 |

## 7\. 성능 개선 시도

단순 모델 적용을 넘어, 다음과 같은 성능 향상 기법을 실험합니다.

1.  **하이퍼파라미터 튜닝 (Hyperparameter Tuning):**

      * **ML:** `GridSearchCV`를 사용하여 로지스틱 회귀의 `C` (규제 강도), 랜덤 포레스트의 `n_estimators` (트리 개수) 등 최적의 파라미터를 탐색합니다.
      * **DL:** CNN의 `filter_size`, `embedding_dim`, `dropout_rate` 등을 조절하며 성능 변화를 관찰합니다.

2.  **특성 공학 (Feature Engineering):**

      * **n-gram:** TF-IDF 적용 시, `(1, 1)` (unigram) 뿐만 아니라 `(1, 2)` (bigram) 또는 `(1, 3)` (trigram)을 함께 사용하여 "정말 재미"와 같은 연어(collocation) 정보를 포착합니다.

3.  **사전 학습된 임베딩 (Pre-trained Embedding) 활용:**

      * Keras Embedding 레이어를 (from scratch) 학습하는 대신, `FastText` 등 대규모 한국어 코퍼스로 미리 학습된 Word Embedding을 로드하여 CNN의 입력으로 사용합니다. (성능 향상 기대)

## 8\. 결론 및 고찰

  * **성능 요약:** (예시) 본 실험 결과, TF-IDF와 로지스틱 회귀의 조합이 간단함에도 불구하고 85% 이상의 강력한 베이스라인 성능을 보였습니다. 1D CNN 모델 역시 유사한 성능을 달성했으며, 이 두 모델을 Soft Voting으로 앙상블했을 때 가장 높은 85.9%의 정확도를 달성했습니다.
  * **모델별 특성:** 로지스틱 회귀는 n-gram을 통해 문맥 정보를 일부 포착했고, CNN은 단어 순서와 지역적 특징을 학습하는 데 강점을 보였습니다.
  * **한계 및 향후 과제 (Future Work):**
      * **단순 앙상블:** 현재는 Voting에 그쳤으나, Stacking 기법을 도입하여 성능을 추가로 개선할 수 있습니다.
      * **최신 모델 도입:** 성능 향상을 위해 `LSTM`, `GRU`와 같은 RNN 계열 모델이나, `KoBERT`, `KcBERT`와 같은 **Transformer** 기반의 사전 학습 모델(PLM)을 적용해볼 수 있습니다.

## 9\. 프로젝트 실행 방법

본 프로젝트는 Google Colab 환경에서 가장 쉽게 재현할 수 있습니다.

1.  **Repository 클론:**

    ```bash
    git clone https://github.com/[Your-Username]/[Your-Repo-Name].git
    cd [Your-Repo-Name]
    ```

2.  **(Optional) 데이터 다운로드:**

      * `data/` 디렉토리에 NSMC 데이터가 포함되어 있지 않다면, [원본 링크](https://github.com/e9t/nsmc)에서 `ratings_train.txt`와 `ratings_test.txt`를 다운로드하여 해당 위치에 저장합니다.

3.  **(Optional) 필요 라이브러리 설치:**

    ```bash
    pip install -r requirements.txt
    ```
    
4.  **노트북 순차 실행:**

      * `notebooks/` 디렉토리의 Jupyter Notebook을 **1번부터 4번까지** 순서대로 실행합니다.

-----
