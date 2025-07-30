# CH5 - TinyML 'HelloWorld': 애플리케이션 구축

입력데이터 -> [ 전처리 -> TF lite interpreter -> 후처리 -> 출력처리] -> 장치 출력

애플리케이션 코드 
- 전처리 : 모델에 적합하게 입력을 변환
- TF lite interpreter: 모델 실행 <-> 모델:데이터를 바탕으로 예측 하도록 훈련됨
- 후처리: 모델의 출력을 해석하고 판단을 내림
- 출력 처리: 장치의 자원을 사용하여 예측에 따른 반응을 수행

## 5.1 테스트 작성
tensorflow-lite/tensorflow/lite/micro/examples/hello_world/helo_world_test.cc

#### 5.1.1 종속성 불러오기
#include 지시문: C++코드가 의존하는 다른 코드를 지정하는 방법. 
```cpp
// #include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/examples/hello_world/sine_model_data.h"
//xxd를 사용하여 훈련, 변환하고 C++코드로 바꾼 사인모델.
//#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
//all_ops_resolver.h 대신 micro_mutalbel_op_resolver.h 사용함
#include "tensorflow/lite/micro/micro_error_reporter.h"
//디버깅을 위해 오류와 출력을 기록하는 클래스
#include "tensorflow/lite/micro/micro_interpreter.h"
//모델을 실행할 마이크로컨트롤러용 텐서플로 라이트
#include "tensorflow/lite/micro/testing/micro_test.h"
//테스트 작성을 위한 간단한 프레임워크. 이 파일을 실행하면 테스트가 이루어짐
#include "tensorflow/lite/schema/schema_generated.h"
//sine_model_data.h의 모델 데이터를 이해하는 데 사용되는 텐서플로 라이트 플랫버퍼 데이터 구조를 정의하는 스키마 
#include "tensorflow/lite/version.h"
//스키마의 현재 버전번호. 모델이 호환 가능한 버전으로 정의되어있는지 확인가능

```
#### 5.1.2 테스트 설정
매크로 : C++에서 코드 덩어리에 이름을 붙여서 다른곳에 포함하여 재사용 가능하도록 저으이 한 코드덩어리. 
```cpp
TF_LITE_MICRO_TESTS_BEGIN //매크로, micro_test.h에 있다.

TF_LITE_MICRO_TEST(LoadModelAndPerformInference) { //매크로, micro_test.h에 있다.
```
#### 5.1.3 데이터 기록 준비 
tflite::는 네임스페이스 접두어로, C++코드 구성에 도움이 된다. 
```cpp
tflite::MicroErrorReporter micro_error_reporter; //인스턴스 정의. 디버그 정보를 기록하는 메커니즘. 
  tflite::ErrorReporter* error_reporter = &micro_error_reporter; 
  //ErrorReporter클래스의 서브 클래스이다. 디버그 정보 출력을 위해 쓰인다.
  //텐서플로 라이트에서 제공하는 디버그 로깅 메커니즘 이다. 
  //재정의 되지 않은 메소드를 인스턴스를 생성하여, 처리한다. 
```

#### 5.1.4 모델 매핑하기 
데이터 정렬
- 프로세서는 데이터가 메모리에 정렬되어 있을 때 가장 효율적으로 데이터를 읽을 수 있다. 즉, 단일 작업에서 프로세서가 읽을 수 있는 경계와 겹치지 않도록 데이터 구조가 저장되는 것이 바람직하다. 이 매크로를 지정하면 가능한 경우 최적의 읽기 성능을 위해 모델 데이터가 올바르게 정렬된다. 
```cpp
//모델을 사용 가능한 데이터 구조에 매핑한다.
//복사나 파싱을 포함하지 않는 가벼운 작업이다. 
  const tflite::Model* model = ::tflite::GetModel(g_sine_model_data); //데이터 모델 배열을 가져와 전달.
  //g_sine_model_data는 DATA_ALIGN_ATTRIBUTE매크로를 정렬을 위해 참고하고 있다. 
  if (model->version() != TFLITE_SCHEMA_VERSION) { //모델의 버전 번호를 검색하는 메서드
  //숫자가 일치하면 모델이 호환되는 버전의 텐서플로 라이트로 변환된다. 
  //일치하지 않으면 진행되기는 하나 경고를 기록한다. 
    TF_LITE_REPORT_ERROR(error_reporter, //error_report(포인터)의 report()메소드가 경고를 기록함. 
                         "Model provided is schema version %d not equal "
                         "to supported version %d.\n",
                         model->version(), TFLITE_SCHEMA_VERSION);
  }
```
#### 5.1.5 AllOpsResolver 생성하기 
```cpp
  // 필요한 모든 operation 구현을 가져온다. 
  tflite::ops::micro::AllOpsResolver resolver;
  ```

#### 5.1.6 텐서 아레나 정의하기
텐서 아레나: 모델의 입력, 출력, 중간 텐서를 저장하는 데 사용되는 메모리 영역 

모델 아키텍처마다 크기와 개수, 입력, 출력, 중간 텐서가 다르므로 필요한 메모리 양을 알기는 어려움. 마이크로컨트롤러는 RAM이 제한되어 있으므로 가능한 한 텐서 아레나를 작게 유지하여 나머지 프로그램을 위한 공간을 확보해야함. 

 n*1024로 배열의 크기를 표히사혹, n의 값을 바꿔나가면 된다. 상당히 높은 크기부터 시작하여 제대로 작동하는지 확인하고, 모델이 제대로 실행되지 않을 때 까지 숫자를 줄인다. 마지막으로 작동한 숫자가 최적의 값이다.
 ```cpp
 // 입력, 출력, 중간 배열에 사용할 메모리 영역을 생성한다.
  //모델 최솟값을 찾으려면 시행착오가 필요하다. 
  const int tensor_arena_size = 2 * 1024;
  uint8_t tensor_arena[tensor_arena_size];
 ```

 #### 5.1.7 인터프리터 생성하기
 ```cpp
  // 모델을 실행하기 위한 인터프리터를 빌드한다. micro tfl의 핵심이다. 
  tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                       tensor_arena_size, error_reporter);

  // 모델의 텐서에 대한 tensor_arena의 메모리를 할당한다. 추론하기 전 반드시 호출해야한다. 
  //모델이 정의한 모든 텐서를 확인한 후, tensor_arena에서 각 텐서로 메모리를 할당한다. 
  TF_LITE_MICRO_EXPECT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
  ```

#### 5.1.8 입력 텐서 검사
인터프리터를 만든 후에는 모델 정보를 제공해야 한다. 

앞으로 텐서를 많이 사용할 것이기 때문에 이 코드로 TfLiteTensor 구조체의 작동 방식에 익숙해지는 것이 좋다 .

어서션 코드 (assertion code)
- TF_LITE_MICRO_EXPECT(x) : x가 true로 평가되는지 확인
- TF_LITE_MICRO_EXPECT_EQ(x,y) : x와 y가 같은지 확인
- TF_LITE_MICRO_EXPECT_NE(x,y) : x와 y가 같지 않은지 확인
- TF_LITE_MICRO_EXPECT_NEAR(x,y,epsilon):숫자값에 대해 x와 y의 차이가 epsilon보다 작거나 같은지 확인한다.
- TF_LITE_MICRO_EXPECT_GT(x,y):숫자값에 대해 x가 y보다 큰지 확인한다.
- TF_LITE_MICRO_EXPECT_LT(x,y):숫자값에 대해 x가 y보다 작은지 확인한다.
- TF_LITE_MICRO_EXPECT_GE(x,y): 숫자값에 대해 x가 y보다 크거나 같은지 확인한다.
- TF_LITE_MICRO_EXPECT_LE(x,y): 숫자값에 대해 x가 y보다 작거나 같은지 확인한다. 

모든 텐서는 형태를 가진다. 모델에는 스칼라 값으로 입력하지만, 케라스 레이어가 입력을 받아들이는 고유한 방식에 맞추려면 입력값을 하나의 숫자를 포함하는 2D텐서로 감싸서 제공해야한다. 

ex) ```cpp [[0]]```

```cpp
 //모델의 입력 텐서에 대한 포인터 얻기 
  //모델은 여러개의 입력 텐서를 가질 수 있으므로 원하는 텐서를 지정하는 input()메서드에 인덱스를 전달해야한다. 
  TfLiteTensor* input = interpreter.input(0); //모델의 입력 텐서가 하나이므로 인덱스는 0이다. 
  //텐서플로 라이트에서 텐서는 TfLiteTensor 구조체로 표시된다. 

  // 입력이 예상하는 속성을 갖는지 확인한다.
  //변수 값에 대한 assertion코드를 작성하여 특정한 값을 예상하고 있음을 증명가능하다. 
  TF_LITE_MICRO_EXPECT_NE(nullptr, input); 
  // "dims"속성은 텐서 모양을 알려준다. 각 차원마다 원소는 하나다.
  //입력은 한 개의 요소를 포함하는 2D 텐서이므로 dims의 크기는 2이다. 
  TF_LITE_MICRO_EXPECT_EQ(2, input->dims->size);
  // 각 원소의 값은 해당 텐서의 길이를 제공한다.
  // 두 개의 차원에 단일 원소 텐서(하나가 다른 하나에 포함됨)를 갖는지 확인한다. 
  TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[1]); 
  // The input is a 32 bit floating point value
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteFloat32, input->type); 
  //type변수는 텐서의 자료형을 알려준다. 
```

#### 5.1.9 입력에 대해 추론 실행 
TfLitePtrUnion 유니언은 동일한 위치의 메모리에 서로 다른 자료형을 저장할 수 있는 특수한 C++ 자료형이다. 주어진 텐서는 다양한 유형의 데이터(ex:부동소수점 숫자, 정수, 부울)중 하나를 포함할 수 있으므로 유니언은 이를 저장하는 데 도움이 되는 완벽한 자료형이다. 

구조는 다음과 같다.
```cpp
typedef union {
  int* i32;
  int64_t* i64;
  float* f;
  char* raw;
  const char* raw_const;
  uint8_t* uint8;
  bool* b;
  int16_t* i16;
  TfLiteComplex64* c64;
  int8_t* int8;
} TfLitePtrUnion;
```
  이전과 마찬가지로 interpreter, alocateTensor()를 호출하면 텐서가 데이터를 저장하도록 할당된 메모리 블록을 가리키는 적절한 포인터가 생성된다. 각 텐서에는 특정 데이터 유형이 있으므로 해당 유형 포인터만 설정된다. 

복잡한 입력:  
 모델의 입력이 여러 값으로 구성된 벡터의 경우, 메모리의 이어지는 위치에 추가해야 한다.   
다음은 숫자 1,2,3을 포함하는 벡터의 예다.  
[1,2,3]  
TfLiteTensor에서 이러한 값을 설정하는 방법은 다음과 같다.  

```cpp
//세 개의 원소를 가지는 벡터
input->data.f[0]=1.;
input->data.f[1]=2.;
input->data.f[2]=3.;
```
다음과 같이 여러 벡터로 구성된 행렬은 어떻게 해야할까?
[[1,2,3]
    [4,5,6]]

TfLiteTensor에서 이것을 설정하기 위해서는 왼쪽에서 오른쪽, 위에서 아래 순서로 값을 할당한다. 구조를 2차원에서 1차원으로 줄이는 이러한 과정을 '평탄화'flattening라고 한다. 
```cpp
//여섯개의 원소를 가지는 벡터
input->data.f[0]=1.;
input->data.f[1]=2.;
input->data.f[2]=3.;
input->data.f[3]=4.;
input->data.f[4]=5.;
input->data.f[5]=6.;
```
TfLiteTensor 구조체는 실제 차원의 레코드를 가지고 있기 때문에, 메모리가 평평한 구조를 가지고 있음에도 메모리의 어느 위치가 다차원 형태 원소에 해당하는지 알 수 있다. 

```cpp
  // Provide an input value
  input->data.f[0] = 0.;
  //데이터를 저장하기 위해 적절한 포인터를 사용하였다. 
  //0.은 0.0의 약어이다. 

  // 입력값으로 모델을 실행하고 성공여부를 확인
  TfLiteStatus invoke_status = interpreter.Invoke(); //추론 실행
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status); //추론 성공여부 
```

#### 5.1.10 출력 읽기
입력과 마찬가지로 모델 출력은 TfLiteTensor를 통해 접근할 수 있으며 쉽게 포인터를 얻을 수 있다.

원하는 값과 동일한 값이 아닌 가까운 값을 확인하는 이유
1. 모델은 실제 사인 값에 근접할 뿐 정확하게 맞지 않을 수 있다.
2. 컴퓨터의 부동소수점 계산에는 기본적으로 오차가 있다. 오류는 컴퓨터마다 다를 수 있다. 
```cpp
// Obtain a pointer to the output tensor and make sure it has the
  // properties we expect. It should be the same as the input tensor.
  TfLiteTensor* output = interpreter.output(0);
  TF_LITE_MICRO_EXPECT_EQ(2, output->dims->size); //테스트를 위해 출력 텐서의 크기, 차원, 자료형이 다음과 같은지 확인한다.
  TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[1]);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteFloat32, output->type);

  // 텐서의 출력값을 획득.
  float value = output->data.f[0];
  // 출력값과 예상 값의 오차가 0.05 범위에 있는지 확인한다. 
  TF_LITE_MICRO_EXPECT_NEAR(0., value, 0.05);

  // 몇 가지 값에 대해 추론을 추가로 실행하고 결과를 확인한다. 
  //동일한 입력과 출력 텐서 포인터의 활용 방식에 주목해야한다. 
  input->data.f[0] = 1.; 
  invoke_status = interpreter.Invoke();
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);

  value = output->data.f[0];
  TF_LITE_MICRO_EXPECT_NEAR(0.841, value, 0.05);

  input->data.f[0] = 3.;
  invoke_status = interpreter.Invoke();
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);

  value = output->data.f[0];
  TF_LITE_MICRO_EXPECT_NEAR(0.141, value, 0.05);

  input->data.f[0] = 5.;
  invoke_status = interpreter.Invoke();
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);

  value = output->data.f[0];
  TF_LITE_MICRO_EXPECT_NEAR(-0.959, value, 0.05);
}

TF_LITE_MICRO_TESTS_END //매크로를 사용하여 테스트의 끝을 나타낸다. 
```
#### 5.1.11 테스트 실행하기 
임베디드 애플케이션을 구축하기 위한 좋은 워크플로는 일반적인 개발 시스템에서 실행할 수 있는 테스트에서 최대한 많은 로직을 작성하는 것이다. 항상 실제 하드웨어를 실행해야 하지만 로컬에서 테스트할수록 편해진다. 

###### 5.1.11.1 코드 돌려보기 

###### 5.1.11.2 Make를 사용하여 실행하기 
- make라는 프로그램을 사용하여 테스트를 할 것이다. 
- make는 소프트웨어 빌드 작업 자동화 도구이다. 
- 개발자는 특수한 언어로 makefile이라는 파일을 작성해서 make에 코드를 빌드하고 실행하는 방법을 지시한다.
- 마이크로컨트롤러용 텐서플로 라이트의 makefile은 micro/tool/make/makefile에 정의되어있다.

먼저 git으로 내려받은 tf디렉터리의 루트에서 실행해야한다. 사용할 makefile을 지정하고 빌드하려는 구성요소인 타겟을 지정한다.

```
cd /Users/emfpdlzj/Desktop/ML-study/TinyML/tensorflow-lite
gmake -f tensorflow/lite/micro/tools/make/Makefile test_hello_world_test
```
이 코드를 실행시 마이크로컨트롤러에 포팅되지 않았지만 코드는 추론을 성공적으로 실행하게 된다. 

## 5.2 프로젝트 파일 구조 
애플리케이션 복잡한 구조이다. 살펴보자

애플리케이션 루트는 tensorflow/lite/micro/examples/hello_world에 있으며 다음 파일들이 포함되어있다.
- BUILD: 기본 애플리케이션 바이너리. 이전에 수행한 테스트를 포함하여 애플리케이션의 소스 코드를 사용하여 빌드할 수 있는 다양한 항목을 나열하는 파일이다.
- Makefile.inc: 이전에 실행한 테스트와 기본 애플리케이션 바이너리인 hello_world를 포함하여 애플리케이션 내의 빌드 대상에 대한 정보가 포함된 makefile이다. 일부 소스파일을 정의한다. 
- README.md: 애플리케이션 빌드와 실행에 대한 지시사항을 포함하는 텍스트 파일
- constants.h, constants.cc: 프로그램 동작을 정의하는 데 중요한 영향을 미치는 다양한 상수(프로그램 수명 동안 변경되지 않는 변수)를 포함하는 파일 싸이다.
- create_sine_model_ipynb: 이번 장에서 사용된 주피터 노트북이다.
- hello_world_test.cc: 모델을 사용하여 추론을 실행하는 테스트
- main.cc: 애플리케이션이 장치에 배포되면 장치에서 가장 먼저 실행되는 프로그램 진입점
- main_functions.h, main_functions.cc: 프로그램에 필요한 모든 초기화를 수행하는 setup()함수와 프로그램의 핵심 로직을 포함하고 루프에서 상태 머신을 무한히 순환하게 설게된 loop()함수를 정의하는 파일쌍. 이 함수들은 프로그램이 시작될 때 main.cc에 의해 호출됨
- output_handler.h, output_handler.cc : 추론이 실행될 때 마다 추력을 표시하는 데 사용할 수 있는 함수를 정의하는 파일 쌍이다. output_handler.cc의 기본 구현은 결과를 화면에 출력한다. 이 구현을 재정의하면 다른 장치에서 다른 작업을 수행할수도 있다.
- output_handler_test.cc : output_handler.h와 output_handler.cc의 코드가 올바르게 작동하는지 증명하는 테스트다. 
- sine_mode_data.h, sine_model_data.cc: 이 장의 첫 부분에서 xxd를 사용하여 내보낸 모델의 데이터 배열을 정의하는 파일쌍이다.

## 5.3 소스 코드 살펴보기
대부분의 일을 처리하는 main_function.cc로 시작하여 다른 파일로 뻗어나가보자. 

#### 5.3.1 main_functions.cc
변수들은 네임스페이스로 묶여있다. 즉, main_function.cc의 어느 곳에서나 접근할 수 있지만 프로젝트 내의 다른 파일에서는 접근할 수 없다. ->두 개의 서로 다른 파일이 동일한 이름의 변수를 정의할 때 생기는 문제를 방지할 수 있다.

setup()은 프로그램이 처음 시작될 때 호출되지만 그 이후에는 다시 호출되지 않는다. 이 함수는 추론을 시작하기 전에 수행해야 하는 모든 일회성 작업을 수행하기 위해 필요하다. 

setup()의 첫 부분은 로깅 설정, 모델 로드, 인터프리터 설정, 메모리 할당이다. 이후 입출력 텐서 모두에 대한 포인터를 가져온다. 마지막으로 함수를 종료하기 위해 inference_dount 변수에 0을 할당한다. 

이제 애플리케이션 로직을 정의해야 한다.

loop()함수에 넣은 코드는 무한히 반복해서 실행된다. 이 코드는 루프에서 실행되므로 시간이 지남에 따라 사인 값 수열이 생성된다. HandlerOutput()을 호출하여 결과를 출력한다.

> 상수 앞에 k를 붙이는것은 C++의 코딩 컨벤션이다. 이를 통해 상수를 쉽게 식별할 수 있다. 

> static_cast<float>()은 정수값을 부동소수점 숫자로 변환하는데 사용된다. c++에서 두 정수를 나누면 그 결과는 정수가 된다. x값이 소수 부분을 포함하는 부동 소수점 숫자가 되려면 숫자를 부동소수점으로 변환해야한다. 

#### 5.3.2 output_handler.cc의 출력 처리 
코드는 매우 간단하다. ErrorReporter 인스턴스를 사용하여 x값과 y값을 기록한다.

배포하려는 각 개별 플랫폼에 대한 output_handler.cc의 커스텀 버전을 제공하면, 플랫폼의 API를 사용하여 LED를 켜는 등 출력을 제어할 수 있다. 

#### 5.3.3 main_functions.cc 정리
loop()에서 마지막으로 inference_count 카운터를 증가시킨다. 이 값이 kInferencesPerCycle에 정의된 사이클당 최대 추론 수에 도달하면 이를 0으로 재설정한다. 
```cpp
  // Increment the inference_counter, and reset it if we have reached
  // the total number per cycle
  inference_count += 1;
  if (inference_count >= kInferencesPerCycle) inference_count = 0;
```
이 코드는 루프가 반복될 때 x값을 단계적으로 이동시키고, 범위 끝에 도달하면 다시 0으로 바꾸는 효과가 있다. 

loop함수는 실행될 때 마다 새로운 x값이 게산되고 추론이 실행되며 결과는 HandleOutput()으로 출력된다. loop()가 계속 호출되면 0에서 2pi범위의 x값 진행에 대해 추론을 실행한 다음 반복한다.

#### 5.3.4 main.cc 이해하기 
C++ 표준은 모든 C++ 프로그램에 main()이라는 전역 함수가 포함되도록 지정한다. 이 함수는 프로그램이 시작될 때 실행되는 함수로, main.cc 파일에 정의되어 있다. main()함수의 존재 덕분에 main.cc가 프로그램의 진입점 역할을 하게 된다.
main()의 코드는 마이크로컨트롤러가 시작될 때마다 실행된다.

main()을 실행하면 setup()함수가 호출되어 한 번만 작업을 수행한다. 그런 다음 while루프를 시작하여 반복적으로 loop()함수를 계속 호출한다. 

 이 루프는 무한히 실행된다. 단일 실행 스레드를 차단하므로, 종료할 방법이 없다. 그러나 마이크로컨트롤러용 소프트웨어를 작성할 때 이런 방식의 무한 루프는 실제로 매우 일반적이다. 멀티태스킹이 없고 하나의 애플리케이션만 실행되므로 루프가 계속 진행되는것은 문제가 되지 않는다. 마이크로컨트롤러는 전원에 연결되어 있는 한 계속 추론하고 데이터를 출력한다.

 ### 5.3.5 애플리케이션 실행하기
 ```
#  mac OS (arm)
 tensorflow/lite/micro/tools/make/gen/osx_arm64/bin/hello_world

# mac Os (intel)
  tensorflow/lite/micro/tools/make/gen/osx_x86_64/bin/hello_world
# linux
tensorflow/lite/micro/tools/make/gen/linux_x86_64/bin/hello_world

#window 
  tensorflow/lite/micro/tools/make/gen/windows_x86_64/bin/hello_world
  ```
실행 결과는 output_handler.cc의 HandleOutput() 함수로 작성된 로그다. 추론당 하나의 로그가 있으며 x_valuesms 2pi에 도달할 때 까지 점차 증가하다가 0으로 감소하고 다시 시작한다. 

## 5.4
마치며