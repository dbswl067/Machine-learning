# 6 Deep Feedforward Networks
딥 피드 포워드 네트워크는 인공신경망으로 피드 포워드 신경망(feedforward neural network)으로 불리기도 한다.

Deep feedforward network는 흔히 말하는 Multi-layer perceptron, 어떤 function f* 에 근사하는 함수를 찾아가는 모형이다.

![피드 포워드 신경망](https://wikidocs.net/images/page/24987/mlp_final.PNG)
## feedforward
feedforward라고 부르는 이유는, 학습의 정보의 흐름을 보게 된다면 데이터 x로 부터 시작돼서, 함수 f를 정의하고, 정의된 함수를 통해 output 인 y^ 를 산출하게 된다.
이 과정에서 feedback connection이 부제하게 된다. 이 feedback connection은 output이 모델에 스스로 feedback을 보내는 연결관계이다.

## Network
네트워크 모델은 일반적으로 directed acyclic graph로 표현이 된다. 에를 들면, 

f(x) = f3(f2(f1(x))) 

이러한 체인구조가 될 것이다. 이 케이스에서 f1은 첫 번째 레이어, f2는 두 번째 레이어가 될 것으로 예상된다.
이러한 함수들의 레이어 개수를 depth이라고 하며, deep learning의 용어가 여기서 나오게 된다.
그리고 f3인 마지막 레이어는 흔히 output 레이어라고 불리게 된다.

## Neural
일반적으로는 우리는 neuroscience로 부터 이 용어를 쉽게 볼 수 있다. 각각 뉴럴넷의 hidden layer는 vector의 형태로 표현이 된다. 이 vector는 뉴런의 역할로 해석이 가능하다.

## Output

![피드포워드 신경망 언어 모델](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FdoUNv3%2FbtqAQ4myJ61%2FVi4iCDghoKtiFR1BebLULk%2Fimg.png)

최종 아웃풋은 label과 일치하게끔 학습이 진행되거나 다른 task에 쓰이는데, 중간 레이어들의 아웃풋들은 직접적으로 쓰이지 않기 때문에 이 레이어들을 hidden layer이라고도 한다. 각 레이어를 이루는 벡터의 차원은 width라고 한다. 레이어를 나타내는 벡터의 요소 하나하나가 신경계의 뉴런과도 같다.

feedforward 네트워크를 이해하는 것은 linear model과 이의 한계를 어떻게 극복했는지에 대해 알아보는 것으로부터 시작할 수 있다. Linear regression이나 logistic regression 같은 linear model은 효율적으로 최적화 할 수 있다.

그러나 linear function을 사용하기에 model capacity에서 한계가 있어, 입력변수 둘 이상의 상호작용을 이해하지 못한다.

따라서 linear function 대신 nonlinear function을 사용하여 nonlinear transformation ϕ(x)를 사용한다.
즉, ϕ 를 이용해서 feature x 를 새로운 representation으로 사용하는 것이다.

## Reference
- [딥 뉴럴 네트워크](https://seungheondoh.netlify.app/blog/mlp)
- [인공 신경망 훑어보기](https://wikidocs.net/24987)
- [딥러닝 피드 포워드](https://m.blog.naver.com/PostView.nhn?blogId=beyondlegend&logNo=221373971859&proxyReferer=https:%2F%2Fwww.google.com%2F)