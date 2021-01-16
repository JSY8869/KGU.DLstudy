# KGU.DLstudy
경기대학교 전자공학과 Deep Learning study 그룹입니다.
---

### CHAPTER 1

[Back propagation](https://www.notion.so/Back-propagation-31ccb32936824c31925b0cbb016ba25f)

[Vanishing Problem](https://www.notion.so/Vanishing-Problem-b959ba565e664ab59d938425f1ed353e)

---

### CHAPTER 2

[Overfitting(과적합)](https://www.notion.so/Overfitting-6e5af44f8be04f17af4d18c1fc347983)

[regularization(정규화)](https://www.notion.so/regularization-59c028e9392048a19074a624e8829d1c)

[batch size, epoch](https://www.notion.so/batch-size-epoch-a67754ec5d2f40ca9e22443a5a1073a9)

---

### CHAPTER 3

[CNN](https://www.notion.so/CNN-1ff67526c7854118a492d5e9e61b039c)

---

### CHAPTER 4

[Word Embedding](https://www.notion.so/Word-Embedding-b30f757b1902453fbc40aad9ff76f0ee)

[선형, 로지스틱, 소프트맥스 회귀](https://www.notion.so/8813c1d764ff43a0a895c90127d8669e)

[Word 2 Vec](https://www.notion.so/Word-2-Vec-6d331d5b7b5d407ca5118abc1821763f)

[Bag of Words](https://www.notion.so/Bag-of-Words-8faaffaafd8c4c6299623bc9f410ec74)

[n-gram](https://www.notion.so/n-gram-0f8ea5193e05494787ca52f0cceca8eb)

---

### CHAPTER 5

[RNN](https://www.notion.so/RNN-29a1f810ca84456c96efed4c7a132fe2)

---

### CHAPTER 6


[강화학습(RL)](https://www.notion.so/RL-80352a3992fe40c7b6689ee4c5c0579d)

---

[https://ebbnflow.tistory.com/120](https://ebbnflow.tistory.com/120) - 딥러닝 코딩 기초 

# **신경망 구현 순서[¶](https://datascienceschool.net/view-notebook/51e147088d474fe1bf32e394394eaea7/#%EC%8B%A0%EA%B2%BD%EB%A7%9D-%EA%B5%AC%ED%98%84-%EC%88%9C%EC%84%9C)**

Keras 를 사용하면 다음과 같은 순서로 신경망을 구성할 수 있다.

1. `Sequential` 모형 클래스 객체 생성
2. `add` 메서드로 레이어 추가.
    - 입력단부터 순차적으로 추가한다.
    - 레이어는 출력 뉴런 갯수를 첫번째 인수로 받는다.
    - 최초의 레이어는 `input_dim` 인수로 입력 크기를 설정해야 한다.
    - `activation` 인수로 활성화함수 설정
3. `compile` 메서드로 모형 완성.
    - `loss`인수로 비용함수 설정
    - `optimizer` 인수로 최적화 알고리즘 설정
    - `metrics` 인수로 트레이닝 단계에서 기록할 성능 기준 설정
4. `fit` 메서드로 트레이닝
    - `nb_epoch` 로 에포크(epoch) 횟수 설정
    - `batch_size` 로 배치크기(batch size) 설정
    - `verbose`는 학습 중 출력되는 문구를 설정하는 것으로, 주피터노트북(Jupyter Notebook)을 사용할 때는 `verbose=2`로 설정하여 진행 막대(progress bar)가 나오지 않도록 설정한다.

    ```python
    def inference(x):
        # 모델을 정의

    def loss(hypothesis, y):
        # 오차함수를 정의

    def training(loss):
        # 학습 알고리즘을 정의

    if __name__ == '__main__':
        # 1. 데이터를 준비한다
        # 2. 모델을 설정한다
            hypothesis = inference(x)
            loss = loss(hypothesis, y)

        # 3. 모델을 학습시킨다
            train = training(loss)

        # 4. 모델을 평가한다
    ```
