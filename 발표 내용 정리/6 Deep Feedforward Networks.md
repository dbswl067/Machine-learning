# 6 Deep Feedforward Networks
딥 피드 포워드 네트워크는 인공신경망으로 피드 포워드 신경망(feedforward neural network)으로 불리기도 한다.

Deep feedforward network는 흔히 말하는 Multi-layer perceptron, 어떤 function f* 에 근사하는 함수를 찾아가는 모형이다.

## feedforward
feedforward라고 부르는 이유는, 학습의 정보의 흐름을 보게 된다면 데이터 x로 부터 시작돼서, 함수 f를 정의하고, 정의된 함수를 통해 output 인 y^ 를 산출하게 된다.
이 과정에서 feedback connection이 부제하게 된다. 이 feedback connection은 output이 모델에 스스로 feedback을 보내는 연결관계이다.

## Network
네트워크 모델은 일반적으로 directed acyclic graph로 표현이 된다. 에를 들면, 

f(x) = f3(f2(f1(x))) 

이러한 체인구조가 될 것이다. 이 케이스에서 f1은 첫 번째 레이어, f2는 두 번째 레이어가 될 것으로 예상된다.
이러한 함수들의 레이어 개수를 depth이라고 하며, deep learning의 용어가 여기서 나오게 된다.
그리고 f3인 마지막 레이어는 흔히 output 레이어라고 불리게 된다.