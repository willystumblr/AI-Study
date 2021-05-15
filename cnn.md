# CNN

## Introduction

### Definition

시각 뉴런 =&gt; 국소적 영역, 단순한 패턴\(단순세포\) + 넓은 영역, 복잡한 패턴\(복잡세포\) = layer

> CNN is a **Deep Learning algorithm** which can take in an input **image**, assign importance \(learnable weights and biases\) to various aspects/objects in the image and be able to differentiate one from the other.

"이미지의 공간 정보를 유지한 상태로 학습이 가능한 모델"

### Properties

* 각 레이어의 입출력 데이터의 형상 유지
* 이미지의 공간 정보를 유지하면서 인접 이미지와의 특징을 효과적으로 인식
* 여러 개의 필터로 이미지의 특징 추출 및 학습
* 추출한 이미지의 특징을 수집/강화: Pooling layer
* 일반 인공 신경망과 비해 학습 파라미터가 적음\(필터를 여러번 적용, 가중치를 공유하기 때문\)

### Terms

* Convolution\(합성곱\): 어떤 함수의 다른 함수에 대한 일치 정도를 Feature Map으로 나타내는 연산
* 채널\(Channel\): convolution에 포함되는 matirces의 depth\(RGB image =&gt; 3\) 필터의 개수와 같다.
* 필터\(Filter\)\(=Kernel\): 일치의 기준이 되는 함수
* 스트라이드\(Strid\): 필터가 이동하는 단위
* 패딩\(Padding\): 이미지의 외곽에 지정된 픽셀만큼  값을 채워 넣는 것
* 피처 맵\(Feature Map\) 또는 액티베이션 맵\(Activation Map\): 합성곱 연산의 결과\(일치 정도\)를 나타낸 map
* 풀링\(Pooling\) 레이어: 출력 데이터를 입력으로 받아서 출력 데이터\(Activation Map\)의 크기를 줄이거나 특정 데이터를 강조하여 나타내는 방법 - Max/Average

## Process of CNN

### Overview

* input image
* Convolutional Neural Network
* Output label \(image class\)

#### Steps

* Convolution
* ReLU layer
* Pooling
* Flatening
* Full Connection

### Step 1-1: Convolution

#### 요소들

* 입력 이미지
* "Feature detector" = 필터
* Feature map

#### 과정

* 이미지의 경계가 되는 왼쪽 위 코너부터 시작해서 필터의 크기에 해당되는 이미지의 부분과 필터의 일치 정도를 곱으로 표현, 그 결과를 feature map의 왼쪽 위 cell에 삽입
* Stride만큼 이동, 같은 연산을 하여 feature map의 cell에 값을 삽입; 반복
* 한 행에 대해 연산이 끝나면 다음 행으로 이동하여 반복

![&#xD569;&#xC131;&#xACF1; &#xC5F0;&#xC0B0; &#xC2DC;&#xC791; &#xB2E8;&#xACC4;](.gitbook/assets/image%20%2815%29.png)

![&#xD569;&#xC131;&#xACF1; &#xC5F0;&#xC0B0; &#xB9C8;&#xC9C0;&#xB9C9; &#xB2E8;&#xACC4;](.gitbook/assets/image%20%2816%29.png)

### Step 1-2: ReLU\(**Rectified Linear Unit**\) layer

이미지의 비선형성을 추가하기 위해 사용하는 활성화 함수 중 하나로, ReLU, Leaky ReLU, Randomized Leaky ReLU의 세 종류가 있음.



