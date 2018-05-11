# Hierarchical Attention Networks

[Hierarchical Attention Networks for Document Classification (Yang et al., 2016)](http://www.aclweb.org/anthology/N16-1174)를 텐서플로우로 구현.

## Dataset

* [Yelp 2018](https://www.yelp.com/dataset/challenge) 리뷰 데이터 중 100만건



## Code Description

* `preprocessing.ipynb`: 데이터 전처리
* `contraction.py`: 영어 축약어 딕셔너리 [출처](https://github.com/dipanjanS/text-analytics-with-python/blob/master/Chapter-5/contractions.py)
* `stopwords.py`: 불용어 리스트
* `utils.py`: TFRecord 파싱 함수
* `model.py`: 모델 클래스
* `train.py`: 트레이닝, 라벨 예측



## Result

* Test accuracy(10% of data) : 69.5% (4 epochs) 