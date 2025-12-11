# AI+X: 딥러닝 2025-2 기말 프로젝트

**정보시스템학과 2024062042 김규민 oplgk0576@hanyang.ac.kr** <br> **정보시스템학과 2023092606 송정빈 happysongjb@hanyang.ac.kr**

-----

## 한국어 영화 리뷰 데이터를 활용한 감성 분석: 성능 비교 및 개선 연구

## **Sentiment Analysis on Korean Movie Reviews: <br> Performance Comparison and Improvement**

본 프로젝트는 **[NSMC (Naver Sentiment Movie Corpus)](https://github.com/e9t/nsmc)** 데이터셋을 활용하여 한국어 텍스트의 긍정/부정 감성을 분류하는 다양한 모델링 방법론을 비교 분석합니다. 고전적인 머신러닝 기법부터 딥러닝 모델까지 단계적으로 적용하며, 전처리 수준과 토큰화 단위가 모델 성능에 미치는 영향을 실증적으로 규명하고자 합니다. <br> 본 문서는 **Google Colab** 환경에서 누구나 실험 결과를 재현할 수 있도록 작성했습니다.

-----

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [데이터셋 및 환경](#2-데이터셋-및-환경)
3. [방법론 및 실험 설계](#3-방법론-및-실험-설계)
    - [3.1. 통계 기반 머신러닝 (Baseline)](#31-통계-기반-머신러닝-baseline)
        - [3.1.1. TF-IDF 기반 모델링](#311-tf-idf-기반-모델링)
        - [3.1.2. Word2Vec 임베딩 모델링](#312-word2vec-임베딩-모델링)
        - [3.1.3. 결정 트리 (Decision Tree)](#313-결정-트리-decision-tree)
    - [3.2. 딥러닝 모델 (1D-CNN)](#32-딥러닝-모델-1d-cnn)
        - [3.2.1. Morpheme-level 1D-CNN (형태소 단위)](#321-morpheme-level-1d-cnn-형태소-단위)
        - [3.2.2. Syllable-level 1D-CNN (음절 단위)](#322-syllable-level-1d-cnn-음절-단위)
        - [3.2.3. Jamo-level 1D-CNN (자소 단위)](#323-jamo-level-1d-cnn-자소-단위)
        - [3.2.4. EDA-based 1D-CNN (EDA 기반 전처리)](#324-eda-based-1d-cnn-EDA-전처리)
    - [3.3. 딥러닝 모델 (LSTM)](#33-딥러닝-모델-lstm)
        - [3.3.1. Morpheme-level LSTM (형태소 단위)](#331-morpheme-level-lstm-형태소-단위)
        - [3.3.2. EDA-based LSTM (EDA 기반 전처리)](#332-eda-based-LSTM-EDA-전처리)
    - [3.4. 대형 언어 모델 (Transformer)](#34-대형-언어-모델-Transformer)
        - [3.4.1. KLUE/RoBERTa Fine-tuning](#341-RoBERTa-Fine-tuned)
4. [전처리 파이프라인 상세](#4-전처리-파이프라인-상세-preprocessing-strategy)
5. [최종 성능 평가 및 분석](#5-최종-성능-평가-및-분석)
    - [5.1. 예측 결과 저장](#51-예측-결과-저장)
    - [5.2. 성능 지표 계산 및 시각화](#52-성능-지표-계산-및-시각화)
6. [실험 결과](#6-실험-결과-experimental-results)
7. [결과 분석 및 고찰](#7-결과-분석-및-고찰-discussion)
8. [관련 연구 및 사용 도구](#8-관련-연구-및-사용-도구-related-works)
    - [8.1. 관련 연구](#81-관련-연구)
    - [8.2. 사용 도구 및 라이브러리](#82-사용-도구-및-라이브러리)
9. [결론](#8-결론-conclusion)


-----


## 1\. 프로젝트 개요

감성 분석(Sentiment Analysis)은 텍스트에 내재된 주관적 의견을 식별하는 과제입니다. 한국어는 교착어 특성상 조사와 어미 처리가 중요하기 때문에, 이번 프로젝트에서는 다양한 방법론을 적용하여 여러 방식의 모델 구축과, **"전처리(Preprocessing)와 토큰화(Tokenization)가 성능에 미치는 영향"** 에 대한 고찰 또한 다루고자 합니다.

-----

## 2\. 데이터셋 및 환경

* **데이터셋:** **[NSMC (Naver Sentiment Movie Corpus)](https://github.com/e9t/nsmc)**
- **데이터 구조:** 각 파일은 `id`, `document`, `label`의 세 가지 컬럼으로 구성됩니다.
    - `id`: 네이버에서 제공하는 리뷰 ID
    - `document`: 실제 리뷰 텍스트
    - `label`: 리뷰의 감성 클래스 라벨 (0: 부정, 1: 긍정)
    - 각 컬럼은 탭(Tab)으로 구분됩니다 (`.tsv` 형식이지만 편의상 `.txt` 확장자 사용).
- **데이터 규모:** 총 200,000개의 리뷰
    - `ratings.txt`: 전체 200,000개 리뷰 통합본
    - `ratings_test.txt`: 테스트용으로 분리된 50,000개 리뷰
    - `ratings_train.txt`: 학습용 150,000개 리뷰
- **데이터 특징:**
    - 모든 리뷰는 140자 이내입니다.
    - 각 감성 클래스는 균등하게 추출되었습니다.
        - 부정 리뷰 100,000개 (기존 평점 1-4점)
        - 긍정 리뷰 100,000개 (기존 평점 9-10점)
        - 중립 리뷰(기존 평점 5-8점)는 제외되었습니다.

* **실험 환경 (Experimental Environment):**
  - **플랫폼:** Google Colab (GPU T4 런타임)
  - **Python 버전:** Python 3.12.12
  - **주요 라이브러리:** `scikit-learn`, `konlpy`, `gensim`, `jamo`, `torch`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `tqdm`, `transformers`, `wordcloud`
  - **라이브러리 설치 방법:** 본 프로젝트는 단일 `.ipynb` 환경에서 수행할 예정입니다. 비교적 큰 데이터셋을 다루는 실험이기 때문에, KoNLPy의 Okt.morphs 대신 **Mecab** 형태소 분석기를 사용하며, 필요한 라이브러리 설치 코드는 아래와 같습니다.

  ```python
  # 1. Mecab 설치 (한국어 형태소 분석기)
  # C++ 기반으로, Java 기반의 KoNLPy 대비 매우 빠른 속도
  !pip install python-mecab-ko

  # 2. 공통 라이브러리 설치
  # jamo: 자소 분리 / gensim: Word2Vec / torch: 딥러닝
  !pip install jamo gensim torch scikit-learn pandas numpy matplotlib seaborn tqdm transformers wordcloud
  ```

-----

## 3\. 방법론 및 실험 설계

저희는 전처리 수준과 모델의 복잡도가 성능에 미치는 영향을 다각도로 분석하기 위해 다음과 같이 실험군을 세분화하여 비교해 보았습니다.

### 3.1. 통계 기반 머신러닝 (Baseline)

가장 기본적인 모델인 로지스틱 회귀(Logistic Regression)와 결정 트리(Decision Tree)를 사용하여, **텍스트 전처리(형태소 분석 및 불용어 제거)의 유무**가 성능에 미치는 영향을 직접적으로 비교하는 파트입니다.

#### 3.1.1. TF-IDF 기반 모델링
단어의 빈도를 가중치로 사용하는 TF-IDF 벡터화 방식
*   **Exp 1-1. TF-IDF (Raw):** 형태소 분석 없이 띄어쓰기(Whitespace) 기준으로 토큰화하며, 불용어 처리를 하지 않은 순수 데이터.
*   **Exp 1-2. TF-IDF (Preprocessed):** **`Mecab`** 분석기를 사용하여 조사/어미 등을 정교하게 분리하고 불용어를 제거한 데이터.
*   **Exp 1-3. TF-IDF (EDA):** **`Mecab`** 분석기를 사용하여 조사/어미 등을 정교하게 분리하고, EDA 분석 결과를 반영하여 감정 기호를 보존하며, 도메인 불용어를 제거한 데이터.
*   **[사용 라이브러리]**
    *   **전처리:** `mecab.MeCab` (Exp 1-2 & Exp 1-3)
    *   **특성 추출:** `sklearn.feature_extraction.text.TfidfVectorizer`
    *   **모델:** `sklearn.linear_model.LogisticRegression`

#### 3.1.2. Word2Vec 임베딩 모델링
단어의 의미 정보를 반영하는 Word2Vec 임베딩을 적용한 후, 문장 내 단어 벡터들의 평균(Mean Pooling)값을 입력으로 사용하는 방식
*   **Exp 2-1. Word2Vec (Raw):** 전처리 없는 띄어쓰기 기준 토큰의 임베딩 학습.
*   **Exp 2-2. Word2Vec (Preprocessed):** 형태소 분절 및 불용어 제거를 수행한 토큰의 임베딩 학습.
*   **Exp 2-3. Word2Vec (EDA):** EDA 분석 결과를 바탕으로 형태소 분절 및 불용어 제거를 엄밀하게 수행한 토큰의 임베딩 학습.
*   **[사용 라이브러리]**
    *   **전처리:** `mecab.MeCab` (Exp 2-2 & Exp 2-3)
    *   **임베딩:** `gensim.models.Word2Vec`
    *   **연산:** `numpy` (평균 풀링 / Mean Pooling)
    *   **모델:** `sklearn.linear_model.LogisticRegression`

#### 3.1.3. 결정 트리 (Decision Tree)
희소(Sparse)한 텍스트 데이터에서 트리 모델이 갖는 한계를 확인하고, 전처리가 트리의 복잡도 완화에 미치는 영향을 확인하고자 포함한 방식
*   **Exp 3-1. Decision Tree (Raw):** 전처리 없는 고차원 데이터 학습.
*   **Exp 3-2. Decision Tree (Preprocessed):** 핵심 형태소만 남긴 데이터 학습.
*   **Exp 3-3. Decision Tree (Preprocessed):** EDA 기반으로 전처리를 거친 데이터 학습.  
*   **[사용 라이브러리]**
    *   **전처리:** `mecab.MeCab` (Exp 3-2 & Exp 3-3)
    *   **특성 추출:** `sklearn.feature_extraction.text.TfidfVectorizer`
    *   **모델:** `sklearn.tree.DecisionTreeClassifier`

### 3.2. 딥러닝 모델 (1D-CNN)

딥러닝 모델에서는 **"입력 토큰 단위"** 를 핵심 변수로 설정하여, 한국어의 언어적 특성에 가장 적합한 표현 방식을 탐구하고자 했습니다. CNN 모델은 `PyTorch`를 사용하여 구현하며, 동일한 아키텍처(Kernel sizes, Filters )를 공유합니다.

#### 3.2.1. Morpheme-level 1D-CNN (형태소 단위)
*   **특징:** 의미의 최소 단위인 형태소를 입력으로 사용 - 문법적 구조를 가장 잘 반영하는 정석적인 방법
*   **입력:** `Mecab`을 통해 분절된 형태소 시퀀스.
*   **[사용 라이브러리]**
    *   **전처리:** `mecab.MeCab`
    *   **모델:** `torch.nn` (임베딩, Conv1d, 선형 레이어)
    *   **최적화:** `torch.optim`

#### 3.2.2. Syllable-level 1D-CNN (음절 단위)
*   **특징:** '글자(Character)' 단위로 문장을 쪼개어 입력 - OOV(미등록 단어 / Out of Vocabulary) 문제가 거의 발생하지 않으며, 오탈자에 비교적 강건할 것으로 예상
*   **입력:** 한국어 음절 시퀀스 (예: "영화" -> "영", "화")
*   **[사용 라이브러리]**
    *   **전처리:** Python 기본 문자열 처리 (슬라이싱)
    *   **모델:** `torch.nn` (임베딩, Conv1d, 선형 레이어)
    *   **최적화:** `torch.optim`

#### 3.2.3. Jamo-level 1D-CNN (자소 단위)
*   **특징:** 한글의 자모(초성, 중성, 종성)를 분리하여 입력 - 음절보다 더 작은 단위의 패턴을 학습하여, 신조어나 비표준어(초성체 등)가 많은 리뷰 데이터에서 잠재력을 가질것으로 예상
*   **입력:** `jamo` 라이브러리의 `j2hcj` 함수를 활용한 자소 시퀀스 (예: "영화" -> "ㅇ", "ㅕ", "ㅇ", "ㅎ", "ㅘ")
*   **[사용 라이브러리]**
    *   **전처리:** `jamo.h2j`, `jamo.j2hcj`
    *   **모델:** `torch.nn` (임베딩, Conv1d, 선형 레이어)
    *   **최적화:** `torch.optim`

#### 3.2.4. EDA-based 1D-CNN (EDA 기반)
*   **특징:** 형태소 분석기를 이용하여 분리된 단어들을 대상으로 추가적인 불용어 처리 과정을 거치고, 의미를 포함하는 주요 품사만을 선별하여 적용
*   **입력:** `Mecab`을 통해 분절된 형태소 시퀀스.
*   **[사용 라이브러리]**
    *   **전처리:** `mecab.MeCab` + `extra_stopwords` + `redundancy_removal`
    *   **모델:** `torch.nn` (임베딩, Conv1d, 선형 레이어)
    *   **최적화:** `torch.optim`

### 3.3. 딥러닝 모델 (LSTM)

순차 데이터(Sequential Data) 처리에 특화된 RNN 계열의 LSTM을 사용하여, 문맥의 장기 의존성(Long-term Dependency)을 학습하고자 했습니다. CNN이 지역적 특징(Local Feature) 추출에 강하다면, LSTM은 문장 전체의 흐름을 파악하는 데 강점이 있다는 차이가 있습니다.

#### 3.3.1. Morpheme-level LSTM (형태소 단위)
*   **특징:** 3.2.1의 CNN 모델과 동일한 '형태소 전처리 데이터'를 사용하여, 아키텍처(CNN vs LSTM)에 따른 성능 차이를 공정하게 비교하고자 했습니다. 정교한 형태소 분절과 불용어 제거를 통해 핵심 의미 단위의 시퀀스를 학습
*   **입력:** `Mecab`을 통해 분절된 형태소 시퀀스.
*   **[사용 라이브러리]**
    *   **전처리:** `mecab.MeCab`
    *   **모델:** `torch.nn` (임베딩, LSTM, 선형 레이어, 드롭아웃)
    *   **시퀀스 처리:** `torch.nn.utils.rnn` (`pad_sequence`, `pack_padded_sequence`, `pad_packed_sequence`)
        *   *참고: 가변 길이 시퀀스의 효율적 학습을 위해 패딩(Padding) 및 패킹(Packing) 기법을 적용했습니다.*
    *   **최적화:** `torch.optim`

#### 3.3.2. EDA-based LSTM (형태소 단위)
*   **특징:** 3.2.1의 CNN 모델과 동일한 '형태소 전처리 데이터'를 사용하여, 아키텍처(CNN vs LSTM)에 따른 성능 차이를 공정하게 비교 - EDA 기반으로 한 더 유의한 형태소 분절과 추가적인 불용어 제거를 통해 핵심 의미 단위의 시퀀스를 학습
*   **입력:** `Mecab`을 통해 분절된 형태소 시퀀스.
*   **[사용 라이브러리]**
    *   **전처리:** `mecab.MeCab` + `extra_stopwords` + `redundancy_removal`
    *   **모델:** `torch.nn` (임베딩, LSTM, 선형 레이어, 드롭아웃)
    *   **시퀀스 처리:** `torch.nn.utils.rnn` (`pad_sequence`, `pack_padded_sequence`, `pad_packed_sequence`)
        *   *참고: 가변 길이 시퀀스의 효율적 학습을 위해 패딩(Padding) 및 패킹(Packing) 기법을 적용했습니다*
    *   **최적화:** `torch.optim`


-----

## 4\. 전처리 파이프라인 상세 (Preprocessing Strategy)

전처리에 앞서 데이터셋에 대한 탐색적 분석을 진행하였습니다.

### Step 1: 라벨 분포 확인 (Label Distribution)
> **분석 목적:** 데이터 불균형(Imbalance)으로 인한 모델 성능 왜곡 방지
*   **분포 확인:** 학습 데이터의 라벨(긍정/부정) 분포를 시각화하여 데이터의 균형을 점검했습니다.
*   **분석 결과:** 부정(0)이 **50.1%**, 긍정(1)이 **49.9%** 로 완벽에 가까운 균형을 이루고 있음을 확인했습니다.
*   **결론:** 별도의 **오버샘플링(Oversampling)** 이나 **언더샘플링** 기법을 적용할 필요가 없다고 판단했습니다.

<img alt="data_label_distribution.png" src="https://github.com/restvatikan/AIX-DL-2025-2/blob/main/images/data_label_distribution.png?raw=true" data-hpc="true" class="Box-sc-62in7e-0 eLrlvS">

### Step 2: 리뷰 길이 분석 (Length Analysis)
> **분석 목적:** 적절한 패딩(Padding) 길이 설정 및 데이터 특성 파악
*   **길이 분포:** 대다수의 리뷰가 **20-40자 내외** 에 분포하고 있으며, 140자를 꽉 채운 리뷰는 소수임을 확인했습니다.
*   **라벨별 비교:** 긍정 리뷰와 부정 리뷰 간의 길이 차이는 통계적으로 유의미하게 나타나지 않았습니다.
*   **전처리 전략:** 딥러닝 모델의 `max_len` 설정 시 **140자**를 기준으로 하되, 패딩을 통해 길이를 맞추는 전략이 유효함을 확인했습니다.

<img alt="data_length_distribution.png" src="https://github.com/restvatikan/AIX-DL-2025-2/blob/main/images/data_length_distribution.png?raw=true" data-hpc="true" class="Box-sc-62in7e-0 eLrlvS">

### Step 3: WordCloud 및 도메인 불용어 식별
> **분석 목적:** 감성 분류 결정 단어 식별 및 노이즈(Noise) 제거
*   **분석 방법:** `Mecab`을 활용하여 명사, 형용사 등을 추출하고 워드클라우드를 생성했습니다.
*   **감성 어휘 식별:**
    *   **부정(Negative):** '지루', '쓰레기', '최악', '재미없' 등 직관적인 부정 어휘 등장.
    *   **긍정(Positive):** '최고', '감동', '인생', '눈물' 등 감정적 찬사 어휘가 두드러짐.
*   **도메인 불용어 처리:** '영화', '평점', '배우', '감독', '관람' 등의 단어는 긍/부정 양쪽 모두에서 빈번하게 등장하는 **'도메인 중립 불용어'** 로 지정하였습니다.  
*   **결론:** EDA 전처리 파이프라인에서 해당 단어들을 제거하여 모델이 실제 감성 어휘에 집중하도록 유도했습니다.

**[부정(Negative) 리뷰 주요 단어]**
<img alt="word_label_0.png" src="https://github.com/restvatikan/AIX-DL-2025-2/blob/main/images/word_label_0.png?raw=true" data-hpc="true" class="Box-sc-62in7e-0 eLrlvS">

**[긍정(Positive) 리뷰 주요 단어]**
<img alt="word_label_1.png" src="https://github.com/restvatikan/AIX-DL-2025-2/blob/main/images/word_label_1.png?raw=true" data-hpc="true" class="Box-sc-62in7e-0 eLrlvS">

**이후, 모델과 실험 목적에 따라 총 4가지의 상이한 전처리 파이프라인을 적용했습니다.**

### 파이프라인 A: 최소 전처리 (Raw 데이터)
> **적용 대상:** Exp 1-1, 2-1, 3-1 (Raw 비교군) 및 3.2.2 (Syllable CNN)
*   가장 기초적인 정제만 수행하여 데이터 본연의 노이즈를 보존했습니다.
1.  **Data Cleaning:** 결측치(Null) 및 중복 데이터 제거.
2.  **Regex Cleaning:** 한글, 영어, 숫자, 공백을 제외한 특수문자 제거 (`[^ㄱ-ㅎ가-힣0-9a-zA-Z ]`).
3.  **Tokenization:** 공백(띄어쓰기) 기준 분절 (Syllable CNN의 경우 음절 단위로 분절).

<br> <br>

### 파이프라인 B: 언어학적 전처리 (형태소 분석)
> **적용 대상:** Exp 1-2, 2-2, 3-2 (Preprocessed 비교군) 및 **3.2.1 (Morpheme CNN), 3.3.1 (Morpheme LSTM)**
*   한국어 문법 지식을 활용하여 의미 단위로 데이터를 압축했습니다.
1.  **Basic Cleaning:** 파이프라인 A와 동일 (결측치 및 특수문자 제거).
2.  **형태소 분석:** **`Mecab` (python-mecab-ko)** 사용.
    *   **형태소 단위 분절:** 문장을 형태소 단위로 정밀하게 분해 (예: '재밌었다' -> '재밌', '었', '다').
    *   *Note: Okt와 달리 인위적인 원형 복원(Stemming)을 수행하지 않고, 형태소 본연의 의미를 보존하여 학습에 활용했습니다.*
3.  **불용어 제거:** 조사, 접속사 등 의미 기여도가 낮은 불용어 리스트 정의 및 제거.
4.  **필터링:** 길이가 1 이하인 토큰 제거.

### 파이프라인 C: Sub-character 전처리 (자모 단위)
> **적용 대상:** 3.2.3 (Jamo CNN)
*   한글의 구성 원리를 활용하여, 자모 단위로 데이터를 분해했습니다.
1.  **Basic Cleaning:** Pipeline A와 동일.
2.  **자모 분리:** `jamo` 패키지 활용.
    *   `h2j()`: 한글 음절을 초/중/종성으로 분리합니다 (조합형 자모).
    *   `j2hcj()`: 분리된 자모를 '한글 호환 자모'로 변환하여 학습 가능한 시퀀스로 생성했습니다.
    *   실제 작업에서는 `j2hcj(h2j("한글텍스트"))` 의 형태로 조합하여 사용할 예정입니다.

### 파이프라인 D: EDA 기반 맞춤형 정제 (EDA)
> **적용 대상:** Exp 1-2, 2-2, 3-2 (EDA 비교군), **3.2.4 (EDA CNN), 3.3.2 (EDA LSTM)** 및 **3.4.1 (EDA BERT)**
*   형태소 분석기를 이용하여 분리된 단어들을 대상으로 추가적인 불용어 처리 과정을 거치고, 의미를 포함하는 주요 품사만을 선별하여 적용했습니다.
1.  **Basic Cleaning:** Pipeline A와 동일.
2.  **EDA 분석 결과 활용:** WordCloud 및 빈도 분석 결과 반영
    *   `감정 기호 보존`: !, ? 와 같은 문장 부호와 ㅋㅋ, ㅎㅎ 등의 자음 표현이 감정 분류에 중요한 단서임을 확인하여, 정규식에서 이를 삭제하지 않고 보존([^가-힣a-zA-Z0-9\s!?.ㅋㅎ])했습니다.
    *   `도메인 불용어 제거`: WordCloud 분석 결과 영화, 배우, 감독, 평점 등은 긍정/부정 리뷰 모두에서 빈번하게 등장하는 중립적 단어임을 확인하여 불용어로 지정, 제거했습니다.
    *   품사 필터링: 조사(J), 어미(E) 등 감정 분석에 불필요한 품사를 선별적으로 제거하여, 내용어(content word)만을 남겼습니다.

### 데이터 효율성을 위한 절차 (Pickle 캐싱)
실험 도중 일어날 수 있는 Google Colab의 런타임 초기화 문제에 대응하고 실험 효율성을 높이기 위해, 각 파이프라인을 거친 데이터는 **`pickle`** 형식으로 로컬에 캐싱(Caching)했습니다.
*   **목적:** 형태소 분석(파이프라인 B) 및 자모 분리(파이프라인 C) 과정을 매 실험마다 반복하지 않고, 저장된 리스트 객체를 즉시 로드하여 학습 시간을 단축하고자 했습니다.
*   **파일 포맷:** `train_morphs.pkl`, `train_jamo.pkl` 등.

-----

## 5\. 최종 성능 평가 및 분석

모든 모델의 학습이 완료된 후, 테스트용 데이터(50,000개)에 대한 객관적인 성능 비교를 위해 통일된 평가 프로세스를 진행했습니다.

### 5.1. 예측 결과 저장
각 모델(3.1 ~ 3.3)로 테스트 데이터셋에 대한 추론을 수행하고, 예측된 라벨(0 또는 1)을 텍스트 파일로 저장하고, 이를 통해 실험 코드를 다시 돌리지 않고도 결과를 영구적으로 보존하고 분석하도록 했습니다.
*   **Output Format:** `prediction_[ModelName].txt` (각 행에 0 또는 1 기록)
*   **[사용 라이브러리]**
    *   `torch` (Inference mode: `with torch.no_grad():`)
    *   `sklearn` (Model `predict` method for ML models)
    *   `numpy` (Array handling)

### 5.2. 성능 지표 계산 및 시각화
저장된 예측 파일들과 정답 라벨(`ratings_test.txt`)을 로드하여 최종 성능을 계산했습니다. 단순 정확도(Accuracy)뿐만 아니라, 모델이 어떤 클래스를 헷갈려하는지 파악하기 위해 혼동 행렬(Confusion Matrix)을 시각화 하였습니다.
1.  **정확도 계산:** 전체 테스트 데이터 중 올바르게 분류한 비율 계산.
2.  **혼동 행렬(Confusion Matrix):** True Positive, True Negative, False Positive, False Negative 분포 확인.
*   **[사용 라이브러리]**
    *   **Metrics:** `sklearn.metrics.accuracy_score`, `sklearn.metrics.confusion_matrix`
    *   **데이터 처리:** `pandas` (결과 집계), `numpy`
    *   **시각화:** `matplotlib.pyplot`, `seaborn` (히트맵 시각화)
-----

## 6\. 실험 결과 (Experimental Results)

총 15가지의 실험군에 대하여 테스트 데이터(50,000개)를 기반으로 측정한 정확도(Accuracy) 순위는 다음과 같습니다. 먼저, 딥러닝 모델과 통계 기반 모델, 그리고 전처리 방법에 따라 성능 차이가 뚜렷하게 나타났습니다. SOTA의 성능을 보이는 LLM 모델의 경우 그래프와 혼동 행렬 시각화에는 첨부하지 않았습니다만 `Val Acc: 0.9008`의 성능으로 가장 높게 나타났습니다.

| Rank | Model Name | Accuracy | Category | Input Unit |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **LSTM_EDA** | **0.8508** | **Deep Learning** | **형태소 (Morpheme)** |
| 2 | CNN_EDA | 0.8403 | Deep Learning | 형태소 (Morpheme) |
| 3 | CNN_Syllable | 0.8393 | Deep Learning | 음절 (Character) |
| 4 | LSTM_Morpheme | 0.8274 | Deep Learning | 형태소 (Morpheme) |
| 5 | CNN_Jamo | 0.8213 | Deep Learning | 자소 (Jamo) |
| 6 | TFIDF_Pre | 0.8179 | Machine Learning | 형태소 (Morpheme) |
| 7 | CNN_Morpheme | 0.8152 | Deep Learning | 형태소 (Morpheme) |
| 8 | TFIDF_EDA | 0.8055 | Machine Learning | 형태소 (Morpheme) |
| 9 | W2V_EDA | 0.8053 | Machine Learning | 형태소 (Morpheme) |
| 10 | W2V_Pre | 0.7817 | Machine Learning | 형태소 (Morpheme) |
| 11 | TFIDF_Raw | 0.7790 | Machine Learning | 어절 (Raw) |
| 12 | W2V_Raw | 0.7174 | Machine Learning | 어절 (Raw) |
| 13 | DT_Pre | 0.6591 | Machine Learning | 형태소 (Morpheme) |
| 14 | DT_EDA | 0.6477 | Machine Learning | 형태소 (Morpheme) |
| 15 | DT_Raw | 0.5584 | Machine Learning | 어절 (Raw) |


### 성능 비교 그래프
<img alt="model_accuracy_comparison.png" src="https://github.com/restvatikan/AIX-DL-2025-2/blob/main/images/model_accuracy_comparison.png?raw=true" data-hpc="true" class="Box-sc-62in7e-0 eLrlvS">

### 혼동 행렬 (Confusion Matrix) 시각화
상위 모델들은 긍정(1)과 부정(0)을 균형 있게 예측한 반면, 하위 모델(특히 Decision Tree Raw)은 특정 클래스로 예측이 편향되는 경향을 보였습니다.
<img alt="confusion_matrices.png" src="https://github.com/restvatikan/AIX-DL-2025-2/blob/main/images/confusion_matrices.png?raw=true" data-hpc="true" class="Box-sc-62in7e-0 eLrlvS">

*(실험 코드 실행 시 `model_accuracy_comparison.png` 및 `confusion_matrices.png` 파일이 생성됩니다.)*

-----

## 7\. 결과 분석 및 고찰 (Discussion)

실험 결과를 바탕으로 전처리의 영향과 모델 아키텍처의 특성을 다음과 같이 분석하였습니다.

### 7.1. 전처리 전략의 영향 (EDA vs General Preprocessing)
*   **딥러닝(DL)에서의 EDA 효과 입증:** 본 실험의 가장 두드러진 결과는 **`LSTM_EDA` (0.8508)** 와 **`CNN_EDA` (0.8403)** 가 나란히 1, 2위를 차지했다는 점입니다. 일반 형태소 분석(Morpheme) 데이터셋을 사용했을 때보다 약 **2.3%p** 이상의 성능 향상이 있었습니다.
*   '영화', '배우'와 같은 **도메인 중립 불용어**를 제거하여 모델이 감성 표현에 집중하도록 유도하고, 'ㅋㅋ', '!!'와 같은 **감정 기호(Emoticon)** 를 보존하여 구어체의 감성 신호를 증폭시킨 EDA 전략이 전처리를 하지 않거나, 데이터셋에 특화시키지 않은 일반적인 전처리 과정에 비해 딥러닝 모델 학습에 더욱 유효했음을 시사합니다.
*   **머신러닝(ML)과 딥러닝의 차이:** 반면, 통계 기반의 `TFIDF` 모델에서는 `TFIDF_Pre`(6위)가 `TFIDF_EDA`(8위)보다 소폭 높은 성능을 보였습니다. 희소한 벡터 공간(Sparse Vector Space)을 사용하는 머신러닝 모델에서는, 불용어를 과감하게 제거하는 EDA 방식이 오히려 결정 경계(Decision Boundary)를 형성하는 데 필요한 미세한 문맥 정보(Context)까지 축소시켰을 가능성이 있습니다. 즉, **딥러닝은 "핵심 정보의 밀도"를 선호하고, 전통적 머신러닝은 "풍부한 어휘 정보"를 선호**하는 경향성에서 이러한 형태가 비롯된 것이 아닐까 예상해보게 되었습니다.

### 7.2. 모델 아키텍처 비교 (LSTM vs CNN)
*   **Sequence 모델의 승리:** 동일한 EDA 전처리 데이터를 사용했을 때, **`LSTM_EDA` (0.8508)** 가 `CNN_EDA` (0.8403)보다 우수한 성능을 보였습니다. 이는 문장의 지역적 특징(Local Feature)을 포착하는 CNN보다, 문장 전체의 흐름과 장기 의존성(Long-term Dependency)을 학습하는 **LSTM**이 올바른 전처리가 병행되었을 때, 감성 분석에 조금 더 적합함을 보여줬습니다.
*   **음절 단위(Syllable) CNN의 효율성:** 비록 1위는 아니지만, **`CNN_Syllable` (83.93%)** 은 형태소 분석기를 사용한 대다수의 모델(`LSTM_Morpheme`, `TFIDF_Pre` 등)을 상회하며 전체 3위를 기록했습니다.
    *   **오탈자와 신조어**가 빈번한 리뷰 데이터의 특성상, 잘못된 형태소 분석으로 인한 오류 전파를 막고 글자 자체의 패턴을 학습하는 방식이 매우 강력한(Robust) 대안이 될 수 있음을 증명합니다. 전처리 과정 대비 성능 효율면에서는 가장 우수한 모델이라 할 수 있었습니다.

### 7.3. 자소(Jamo) 단위의 잠재력
추가로 `CNN_Jamo`(82.13%) 모델 역시 상위권에 위치했습니다. 이는 초성체(예: 'ㅎㅎ', 'ㅋㅋ')나 변형된 자모 표현이 많은 인터넷 구어체 텍스트에서 자소 단위 분해가 유의미한 특징을 추출할 수 있음을 보여줬습니다.

### 7.4. Baseline 및 LLM과의 비교
*   **TF-IDF의 저력:** 단순 선형 모델인 `TFIDF_Pre`(81.79%)가 딥러닝 모델인 `CNN_Morpheme`보다 높은 성능을 기록했습니다. 이는 감성 분석 태스크에서 특정 키워드(긍/부정 단어)의 존재 여부로 판단하는 것도 여전히 유의함을 알 수 있었습니다.
*   **SOTA 모델 (LLM):** 본문 표에는 포함되지 않았으나, 추가 실험으로 진행한 `RoBERTa Fine-tuning` 모델은 약 **90.08%** 의 정확도를 기록했습니다. 이는 충분한 컴퓨팅 자원이 있다면 사전 학습된(Pre-trained) 대형 언어 모델을 사용하는 것이 성능 측면에서 가장 확실한 방법임을 재확인시켜 줍니다. 하지만, 시간적 측면에서나 비용적 측면에서나 접근성이 뛰어난 `LSTM_EDA` 모델이 85%대의 준수한 성능을 낸다는 점은 실무적 관점에서 매우 유의미하다고 판단됐습니다.
-----

## 8\. 관련 연구 및 사용 도구 (Related Works)

### 8.1. 관련 연구

참고 문헌 (Reference) 및 참고 사항
> Mao, Y., Liu, Q., & Zhang, Y. (2024). Sentiment analysis methods, applications, and challenges: <br> A systematic literature review. Journal of King Saud University – Computer and Information Sciences, 36, 102048. https://doi.org/10.1016/j.jksuci.2024.102048

*   실험의 주요 지표였던 Accuracy 외에도 논문에서 소개한 F1-score 등 다양한 지표로의 확장과 구체적인 분석이 가능하도록 **'혼동 행렬(Confusion Matrix)'** 을 함께 기록하였습니다. 
*   또한, 논문에서 제시한 감성 분석 기술의 발전 역사를 반영하여, 기술의 고도화가 실제 성능에 미치는 영향을 단계적으로 확인하고자 **[DT → CNN/LSTM → RoBERTa]** 순으로 모델을 구성해 보았습니다.
*   마지막으로, 데이터셋 선정에 있어서는 리뷰 논문에서 주로 다루어진 영어 텍스트 기반의 벤치마크 대신 **한국어 데이터셋**을 채택하여 차별화를 두었습니다. 영어와는 문법적 구조와 언어 체계가 판이한 한국어에서는 어떠한 절차로 감성분석을 진행해야 하는지 탐구하고자 했습니다. 즉, 논문을 통해 기본적인 아이디어를 얻으면서도, 새로운 영역으로의 확장해보기 위해서 한국어 데이터셋을 활용한 실험을 설계했습니다.

### 8.2. 사용 도구 및 라이브러리
본 연구의 데이터 전처리, 모델 구현 및 성능 평가를 위해 사용된 주요 도구와 라이브러리는 아래와 같습니다.

| Tool/Library | Purpose | Reference |
| :--- | :--- | :--- |
| **NSMC** | Dataset (Korean Movie Reviews) | [e9t/nsmc](https://github.com/e9t/nsmc) |
| **PyTorch** | Deep learning framework (CNN, LSTM) | [pytorch.org](https://pytorch.org) |
| **Transformers** | Pre-trained LLM (RoBERTa) | [Hugging Face](https://huggingface.co) |
| **Mecab (ko)** | Morphological Analysis | [python-mecab-ko](https://pypi.org/project/python-mecab-ko/) |
| **Jamo** | Hangul Jamo Decomposition | [pypi/jamo](https://pypi.org/project/jamo/) |
| **Gensim** | Word2Vec Embedding | [radimrehurek.com](https://radimrehurek.com/gensim/) |
| **Scikit-learn** | Machine Learning (TF-IDF, DT, LR) | [scikit-learn.org](https://scikit-learn.org) |
| **Pandas / NumPy** | Data Processing & Manipulation | [pandas.pydata.org](https://pandas.pydata.org) |
| **Matplotlib / Seaborn** | Result Visualization | [matplotlib.org](https://matplotlib.org) |
| **Google Colab** | Experiment Platform (GPU T4) | [research.google.com](https://colab.research.google.com) |
| **Python 3.12** | Programming Language | [python.org](https://www.python.org) |

-----
## 9\. 결론 (Conclusion)

본 프로젝트를 통해 저희는 단순 통계 모델부터 딥러닝 모델까지 다양한 모델이 감성분석 작업에 있어서 어떠한 차이점을 가지며, 전처리 수준이 각 성능에 미치는 영향을 탐구해 볼 수 있었습니다. 이후 아래와 같은 결론을 도출해보았습니다.

1.  **최적의 전처리 전략:** 딥러닝 모델을 활용할 경우, 단순한 형태소 분석을 넘어 도메인 특화 불용어 제거와 감정 기호 보존을 수행하는 **EDA(Exploratory Data Analysis) 기반 전처리**가 크게 유의했습니다. 전처리 방식의 변화만으로 약 2% 이상의 성능 향상을 이끌어낼 수 있었습니다. 즉, 원본 데이터인 "한국어 영화 리뷰 데이터셋(NSMC)"의 특징을 분석하고, 그에 적합한 전처리 방법론을 설계하는 것이 중요했던 것 같습니다.
2.  **모델 선정 가이드라인:**
    *   **성능 지향:** 컴퓨팅 자원이 충분하다면 **LLM (BERT/RoBERTa)** 을, 제한된 환경에서는 **EDA 전처리를 거친 LSTM** 모델을 추천할 수 있을것 같습니다.
    *   **가성비 및 속도 지향:** 형태소 분석기의 로딩 및 연산 비용을 절감하고 싶다면, **음절(Character) 단위의 1D-CNN**이 가장 합리적인 선택이라고 할 수 있습니다. 복잡한 전처리 없이도 상위권(Rank 3)의 성능이 나온점에서, 데이터에 대한 사전 분석이 어려울 때 적용하기에는 제일 좋은 모델이라고 생각했습니다.
3.  **머신러닝의 한계와 필수 조건:** 전통적인 머신러닝(TF-IDF 등)을 사용할 경우 **형태소 분석은 선택이 아니라 필수** 였습니다. 전처리 없는 Raw 데이터 사용 시, 딥러닝 모델에 비해 성능 하락폭이 매우 컸습니다.

결론적으로, 한국어 텍스트 감성 분석은 단순히 복잡한 모델을 사용하는 것보다, **데이터의 특성(구어체, 오타 등)을 고려한 세밀한 전처리(Preprocessing)와 적절한 토큰화 단위(Tokenization Unit)의 선정**이 성능을 결정짓는 핵심 요인임을 알 수 있었습니다.


-----
## Appendix
-----
## 역할 분배 (Role)

### 김규민
*   Code A (EDA 및 결과 분석 및 시각화 / 전처리 구현)
*   Github (Commit only)
*   Documentation A 
*   video recording

### 송정빈
*   Code B (ML, DL 등 모델 구현 / 평가 함수 구현)
*   Github (Repositary management / file management)
*   Documentation B
*   Literature / Theory Investigation
-----

Task 4 Link
*   딥러닝 발표 영상입니다. [(https://youtu.be/uKS9zgCs-88)]
