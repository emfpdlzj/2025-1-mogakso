# CH4 - TinyML 'Hello World' 시작하기 : 모델 구축과 훈련

## 4.1 만들고자 하는 시스템
사인함수의 결과를 시간에 따라 기록하여 얻은 그래프 사인파를 학습->  
x라는 값이 들어왔을 때 사인 함수의 결괏값인 y를 예측할 수 있는 모델 학습
[참고링크](https://github.com/yunho0130/tensorflow-lite)

## 4.2 머신러닝 도구
주피터 노트북 + 파이썬

- 주피터 노트북: 클릭 한 번으로 문서 작업, 그래픽, 코드를 함께 실행할 수 있는 특수 문서 형식
- 텐서플로: 머신러닝 모델을 구축, 훈련, 평가, 배포하기 위한 도구 모음. 원래 구글에서 개발됐고 이젠 전 세계 수천 명의 참여자가 구축하고 유지관리하는 오픈소스 프로젝트. 머신러닝에 가장 널리 사용되는 프레임워크. 대부분 파이썬 라이브러리의 형태로 텐서플로 사용. 
- Keras: 딥러닝 네트워크를 쉽게 구축하고 훈련시킬 수 있는 텐서플로의 고급 API

## 4.3 모델 구축하기
[노트북 링크](https://oreil.ly/NN6Mj)

#### 4.3.1. 종속성 라이브러리 가져오기
$$$$ 4.3.2. 데이터 생성
```
# 아래의 값만큼 데이터 샘플을 생성할 것이다.
SAMPLES = 1000

# 시드 값을 지정하여 이 노트북에서 실행할 때마다 다른 랜덤 값을 얻게 한다.
# 어떤 숫자든 사용할 수 있다.
SEED=137

# 사인파 진폭의 범위인 0~2π 내에서 균일하게 분포된 난수 집합을 생성한다.
np.random.seed(SEED)
tf.random.set_seed(SEED) # 노트북 실행마다 다른 값

#0~2pi 내에서 사인 값 생성
x_values=np.random.uniform(low=0, high=2*math.pi, size=SAMPLES)
#값 순서를 섞어줌
np.random.shuffle(x_values)
y_values=np.sin(x_values) #사인 값 계산
#데이터를 그래프로 그림. 'b.': 파란색
plt.plot(x_values, y_values, 'b.')
plt.show()
```
랜덤 값 추가하여 노이즈 만들어보기
```
# Add a small random number to each y value
y_values += 0.1 * np.random.randn(*y_values.shape)

# Plot our data
plt.plot(x_values, y_values, 'b.')
plt.show()
```
#### 4.3.3. 데이터 분할  
검증을 위해 데이터의 20%, 테스트를 위해 20%, 모델 학습을 위해 나머지 60% 사용.
```
# We'll use 60% of our data for training and 20% for testing. The remaining 20%
# will be used for validation. Calculate the indices of each section.
TRAIN_SPLIT =  int(0.6 * SAMPLES)
TEST_SPLIT = int(0.2 * SAMPLES + TRAIN_SPLIT)

# Use np.split to chop our data into three parts.
# The second argument to np.split is an array of indices where the data will be
# split. We provide two indices, so the data will be divided into three chunks.
x_train, x_validate, x_test = np.split(x_values, [TRAIN_SPLIT, TEST_SPLIT])
y_train, y_validate, y_test = np.split(y_values, [TRAIN_SPLIT, TEST_SPLIT])

# Double check that our splits add up correctly
assert (x_train.size + x_validate.size + x_test.size) ==  SAMPLES

# Plot the data in each partition in different colors:
plt.plot(x_train, y_train, 'b.', label="Train")
plt.plot(x_validate, y_validate, 'y.', label="Validate")
plt.plot(x_test, y_test, 'r.', label="Test")
plt.legend()
plt.show()
```
#### 4.4.4. 기본모델 정의  
- 입력값을 사용하여 숫자 출력값을 예측하는 모델 -> regression
- 케라스를 사용하여 sequential 모델 생성
- 첫 번째 레이어에는 단일 입력(x값)과 16개의 뉴런이 있음. 이것은 **Dense**레이어로, 예측할 때 입력이 뉴런의 모든 단일 뉴런으로 전달됨 의미. 다음 각 뉴런은 어느 정도 activated. 정도는 훈련 중 학습된 가중치, 편향, activation function을 기준으로 함. 

활성화함수로는 ReLU함수(rectified linear unit)사용

optimizer인수는 훈련 중 네트워크가 입력을 모델링하도록 조정하는 알고리즘 지정

loss argument:훈련과정에서 네트워크 예측이 실젯값에서 얼마나 멀리 떨어져 있는지 계산하기 위해 사용할 방법 지정. 이 방법을 loss fuction이라 함. 
- mse(mean squared error):평균 제곱 오차법
- mae(mean absolute error):평균 절대 오차법

``` 
# We'll use Keras to create a simple model architecture
from tensorflow.keras import layers
model_1 = tf.keras.Sequential()

# First layer takes a scalar input and feeds it through 8 "neurons". The
# neurons decide whether to activate based on the 'relu' activation function.
model_1.add(layers.Dense(16, activation='relu', input_shape=(1,)))

# Final layer is a single neuron, since we want to output a single value
model_1.add(layers.Dense(1))

# Compile the model using the standard 'adam' optimizer and the mean squared error or 'mse' loss function for regression.
model_1.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

#요약된 모델설계 출력
model_1.summary()
```

## 4.4 모델 학습시키기
```
# Train the model on our training data while validating on our validation set
history_1 = model_1.fit(x_train, y_train, epochs=1000, batch_size=16,
                        validation_data=(x_validate, y_validate))
```
- fit()의 처음 두가지 인수: x_train, y_train: 훈련 데이터의 x,y값
- epochs: 전체 훈련 데이터셋이 훈련 중 네트워크를 통해 몇 번이나 실행될 것인지 지정함. 
- batch_size: 정확도를 측정하고 가중치와 편향을 업데이트 하기 전에 네트워크에 공급할 훈련 데이터의 수를 지정함. 16또는 32로 시작해서 효과적인값을 실험해보는게 좋음. 64, 32, 16으로 실험해본 결과 본 코드에서는 16이 가장 좋았다..
- validation_data : 검증 데이터셋을 지정하는 인수. 

#### 4.4.1 훈련 지표
- loss:손실 함수의 출력값이다. 작을수록 좋다. 
- mae:훈련 데이터의 평균 오차.
- val_loss: 검증 데이터에 대한 손실 함수의 출력.
- val_mae:검증 데이터의 평균 오차. 크면 과적합 되고 있다는 뜻

평균적으로 훈련데이터가 검증데이터보다 오류가 낮다.

#### 4.4.2 히스토리 개체 그래프로 나타내기
```
# Draw a graph of the loss, which is the distance between
# the predicted and actual values during training and validation.
loss = history_1.history['loss']
val_loss = history_1.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'g.', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
```
목표는 모델이 더이상 개선되지 않거나 훈련 손실이 검증 손실보다 작을 때 훈련을 중단하는 것이다. 훈련 손실이 더 작아진다는 것은 검증 데이터보다 훈련 데이터를 더 잘 예측하는 방법을 배워서 새로운 데이터가 들어와도 더 이상 일반화시키기 어려워짐을 의미한다.

처음 몇 에폭 동안 손실이 급격히 떨어지기 때문에 그래프의 나머지 부분을 읽기가 매우 어렵다. 다음 셀을실행하여 처음 100개의 에폭을 건너뛰어보자.
```
# Exclude the first few epochs so the graph is easier to read
SKIP = 100

plt.plot(epochs[SKIP:], loss[SKIP:], 'g.', label='Training loss')
plt.plot(epochs[SKIP:], val_loss[SKIP:], 'b.', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
```
600에폭까지는 손실이 감소 -> 굳이 1000만큼 학습시킬 필요 없다는 뜻 
```
#예측에서 오차를 측정하는 또 다른 방법인 평균 절대 오차 그래프를 그려보자.
mae=history_1.history['mae']
val_mae=history_1.history['val_mae']

plt.plot(epochs[SKIP:], mae[SKIP:], 'g.', label='Training MAE')
plt.plot(epochs[SKIP:], val_mae[SKIP:], 'b.', label='Validation MAE')
plt.title('Training and validation mean absolute error')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.show()
```
몇 가지 단서
- 평균적으로 훈련 데이터가 검증 데이터보다 오류가 낮다. -> 네트워크가 과적합됐거나 훈련데이터를 너무 학습하여 새로운 데이터 예측 불가 

훈련데이터에 대한 네트워크 예측을 기댓값과 비교 
```
# 모델을 사용하여 검증 데이터로부터 예측값 생성
predictions = model_1.predict(x_train)

# 테스트 데이터와 함께 예측값을 그래프로 표현
plt.clf()
plt.title('Training data predicted vs actual values')
plt.plot(x_test, y_test, 'b.', label='Actual')
plt.plot(x_train, predictions, 'r.', label='Predicted')
plt.legend()
plt.show()
```
모델을 더 크게 만들면 성능을 향상시킬 수 있을것임

#### 4.4.3 모델 개선하기 
네트워크를 더 크게 만드는 쉬운 방법 -> 다른 뉴런레이어 추가  
이전과 같은 방식으로 모델을 재정의하지만, 중간에 16개의 뉴런을 추가해보자. 
```
model_2 = tf.keras.Sequential()

# First layer takes a scalar input and feeds it through 16 "neurons". The
# neurons decide whether to activate based on the 'relu' activation function.
model_2.add(layers.Dense(16, activation='relu', input_shape=(1,)))

# The new second and third layer will help the network learn more complex representations
model_2.add(layers.Dense(16, activation='relu'))

# Final layer is a single neuron, since we want to output a single value
model_2.add(layers.Dense(1))

# Compile the model using the standard 'adam' optimizer and the mean squared error or 'mse' loss function for regression.
model_2.compile(optimizer='rmsprop', loss="mse", metrics=["mae"])

#모델 요약
model_2.summary()
```

```
# Train the model on our training data while validating on our validation set
history_2 = model_2.fit(x_train, y_train, epochs=600, batch_size=16,
                        validation_data=(x_validate, y_validate))
```

val_loss가 0.17->0.01, val_mae가 0.32->0.08로 크게 개선됨 
```
# Draw a graph of the loss, which is the distance between
# the predicted and actual values during training and validation.
loss = history_2.history['loss']
val_loss = history_2.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'g.', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

그래프 분석 결과
- 훈련보다 검증의 지표가 훨씬 더 좋음. 네트워크가 과적합되지 않았음.
- 전체 손실과 평균 절대 오차가 이전 네트워크보다 훨씬 나아짐

#### 4.4.4 테스트
모델이 검증 데이터에 과적합하면 테스트 데이터셋의 지표가 검증 데이터셋보다 훨씬 더 나쁠것이라고 예상가능. 
```
# Calculate and print the loss on our test dataset
loss = model_2.evaluate(x_test, y_test)

# Make predictions based on our test dataset
predictions=model_2.predict(x_test)

# Graph the predictions against the actual values
plt.clf()
plt.title('Comparison of predictions and actual values')
plt.plot(x_test, y_test, 'b.', label='Actual values')
plt.plot(x_test, predictions, 'r.', label='TF predicted')
plt.legend()
plt.show()
```
과적합이 있는듯 하다

# 4.5 텐서플로 라이트용 모델 변환

- tensorflowlite converter: tensorflow모델을 메모리가 제한된 장치에서 사용하기 위해 특수한 포맷으로 변경함. 모델크기를 줄인 뒤 빠른실행을 위한 최적화도 적용가능
- tensorflowlite interprete: 주어진 장치에 가장 효율적인 연산을 사용하여 적절히 변환된 텐서플로 라이트 모델을 실행함 

- quantization: 모델 가중치와 편향은 기본적으로 32비트 부동소수점숫자임. 양자화를 사용해 이 숫자의 정밀도를 8비트 정수에 맞추면 모델 크기가 4배 줄어듬. 정확도의 손실을 최소화함.

```
#양자화 없이 모델을 텐서플로 라이트로 변환
converter=tf.lite.TFLiteConverter.from_keras_model(model_2)
tflite_model=converter.convert()

#모델을 디스크에 저장
open("sine_model.tflite", "wb").write(tflite_model)

#양자화하여 모델을 텐서플로 라이트 형식으로 변환
converter=tf.lite.TFLiteConverter.from_keras_model(model_2)
#양자화를 포함한 기본 최적화 연산 수행 
converter.optimizations=[tf.lite.Optimize.DEFAULT]
#평가 데이터의 x값을 대표 데이터셋으로 제공하는 생성 함수를 정의하고 변환기에 사용
def representative_dataset_generator():
  for value in x_test:
    #각 스칼라값은 반드시 리스트로 쌓여 있는 2차원 배열 안에 있어야 함
    yield [np.array(value, dtype=np.float32, ndmin=2)]
converter.representative_dataset = representative_dataset_generator
#모델변환
tflite_model=converter.convert()

#모델을 디스크에 저장
open("sine_model_quantized.tflite", "wb").write(tflite_model)
```
양자화된 모델이 가능한 한 효율적으로 실행되게 만들려면 모델이 학습한 데이터셋의 전체 입력값 범위를 나타내는 숫자의 집합인 representative dataset을 제공해야함.

케라스모델보다 tf interpreter가 조금 더 복잡함
1. 인터프리터 객체 인스턴스화
2. 모델에 메모리를 할당하는 메서드 호출
3. 입력 텐서에 입력값 작성
4. 모델 호출
5. 출력 텐서에서 출력값 읽기

```
#각 모델 인터프리터 인스턴스화.
sine_model= tf.lite.Interpreter('sine_model.tflite')
sine_model_quantized=tf.lite.Interpreter('sine_model_quantized.tflite')

#각 모델에 메모리 할당
sine_model.allocate_tensors()
sine_model_quantized.allocate_tensors()

#입력과 결과 텐서 인덱스 가져오기
sine_model_input_index = sine_model.get_input_details()[0]["index"]
sine_model_output_index = sine_model.get_output_details()[0]["index"]
sine_model_quantized_input_index=sine_model_quantized.get_input_details()[0]["index"]
sine_model_quantized_output_index=sine_model_quantized.get_output_details()[0]["index"]

#결과저장용 배열 생성
sine_model_predictions=[]
sine_model_quantized_predictions=[]

#각 값에 대해 각 모델의 인터프리터를 실행하고 결과를 배열에 저장
for x_value in x_test:
  #현재 x값을 감싸고 있는 2차원 텐서 생성
  x_value_tensor=tf.convert_to_tensor([[x_value]], dtype=np.float32)
  #값을 입력 텐서에 쓰기
  sine_model.set_tensor(sine_model_input_index, x_value_tensor)

  #추론 실행
  sine_model.invoke()
  #예측값을 결과 텐서에서 읽기
  sine_model_predictions.append(sine_model.get_tensor(sine_model_output_index)[0])
  #양자화된 모델에 같은 작업 실시
  sine_model_quantized.set_tensor(sine_model_quantized_input_index, x_value_tensor)
  sine_model_quantized.invoke()
  sine_model_quantized_predictions.append(sine_model_quantized.get_tensor(sine_model_quantized_output_index)[0])

#데이터가 어떻게 정렬되는지 확인
plt.clf()
plt.title('Comparison of various models against actual values')
plt.plot(x_test, y_test, 'bo', label='Actual')
plt.plot(x_test, predictions, 'ro', label='Original predictions')
plt.plot(x_test, sine_model_predictions, 'bx', label='Lite predictions')
plt.plot(x_test, sine_model_quantized_predictions, 'gx', label='Quantized Lite predictions')
plt.legend()
plt.show()
```
사이즈비교해보기
```
# Define paths to model files
import os
basic_model_size=os.path.getsize("sine_model.tflite")
print("Basic model is %d bytes" % basic_model_size)
quantized_model_size=os.path.getsize("sine_model_quantized.tflite")
print("Quantized model is %d bytes" % quantized_model_size)
difference=basic_model_size-quantized_model_size
print("Difference is %d bytes" % difference)
```
더 복잡한 모델은 더 많은 가중치를 가지므로 양자화로 인해 공간이 훨씬 더 절약된다. 
### 4.5.1 C파일로 변환하기
모델을 애플리케이션에 포함할 수 있는 C소스 파일로 변환해보자.

지금까지 텐서플로의 파이썬 API는 인터프리터 생성자를 사용했다. 이는 디스크에서 모델 파일을 읽을 수 있음을 의미한다. 

그러나 대부분의 마이크로 컨트롤러에는 파일 시스템이 없다. 설사 있다고 해도 디스크에서 모델을 로드하는 데 필요한 추가 코드는 제한된 공간 때문에 낭비된다. 그 대신 훌륭한 방식으로 바이너리에 포함시켜 메모리에 직접 로드할 수 있는 C소스 파일 형식으로 모델을 제공할 수 있다.

파일에서 모델은 바이트 배열로 정의되며 ** xxd라는 편리한 유닉스 도구**가 있어 주어진 파일을 필요한 형식으로 변환 가능하다. 
```
#xxd 사용 불가시 설치
!apt-get qq install xxd
#파일을 c 소스 코드로 저장
!xxd -i sine_model_quantized.tflite > sine_model_quantized.cc
#소스 파일 출력
!cat sine_model_quantized.cc
```
## 4.6 마치며