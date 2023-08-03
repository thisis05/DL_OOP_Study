#!/usr/bin/env python
# coding: utf-8

# # Chapter 5. | 오차역전파법 (Backpropagation)

# ##### 여태까지 가중치는 수치 미분을 통해서 구했지만 오래걸린다는 단점이 있기 때문에 Backpropagation 을 통해서 가중치를 구해볼 것입니다. 구하는 방법은 총 2가지 입니다
# - 수식을 통해서 구하는 것
# - 계산 그래프
# 
# 이 장에서는 계산 그래프 (시각적으로 조금 더 잘 보이는) 방법을 주로 다룹니다.

# ## 5.1 계산 그래프 (Computational Graph)
# 계산 과정을 그래프로 나태낸 것

# ### 5.1.1 계산 그래프를 풀다
# ##### 문제 1: 현빈 군은 슈퍼에서 1개에 100원인 사과를 2개 샀습니다. 이때 지불금액을 구하세요. 단, 소비세가 10% 부과됩니다.<br>
# ![Screenshot 2023-08-03 at 8.32.02 AM.png](attachment:32361316-a083-474e-b021-043af92131e6.png)![Screenshot 2023-08-03 at 8.31.57 AM.png](attachment:334d425b-c317-4128-8b5c-2425ca4f0ee4.png)
# <br>각 동그라미는 노드로 해석하면 되며 노드 안에는 연산이 들어갑니다. 그림 1에서는 노드 안에 연산 값이 있지만 오로지 연산만을 노드로 바라볼 수 있으며 이것이 그림 2가 보여주는 바입니다.

# ##### 문제 2: 현빈 군은 슈퍼에서 사과를 2개, 귤을 3개 샀습니다. 사과는 1개에 100원, 귤은 1개 150원입니다. 소비세가 10%일 때 지불 금액을 구하세요.<br>
# ![Screenshot 2023-08-03 at 8.34.22 AM.png](attachment:d64d158d-d12a-4893-a69c-807a177949a1.png)
# <br>문제 1의 확장판이라고 생각하시면 되며 여기에서는 더하기 연산도 들어간다는 차이가 있습니다. 한 마디로 다른 노드가 추가됐다고 생각하면 됩니다.

# ##### 이러한 것들을 계산 그래프라고 불리며 왼쪽에서 오른쪽으로 진행되기 때문에 순전파 (Forward Propagation) 이라고 불립니다. 
# ##### 하지만 저희가 관심있는 것은 이것의 반대인 역전파 (Backward Propagation) 입니다. 

# ### 5.1.2 국소적 계산
# <br>
# 
# ![Screenshot 2023-08-03 at 8.48.24 AM.png](attachment:dc212002-0aa0-4a21-b87c-289a9dc67238.png)
# <br>국소적 계산이란 전체에 어떤 일이 벌어지든 상관없이 자신과 관계된 정보만으로 결과를 출력하는 것을 말합니다. 위의 예시에서는 다른 과일들이 어떠한 복잡한 계산을 하는 것과 사과 2개를 구입하는 계산 과정이 있습니다. 저희가 관심 있는 것은 사과를 포함해 여러 식품을 구하는 것이기 때문에 오로지 **4000 + 200** 에만 집중하면 되며 이것이 바로 국소적 계산입니다.
# <br><br>
# 이것이 중요한 이유는 제아무리 복잡한 계산이라도 국소적 계산으로 바라본다면 이해를 하기가 편하기 때문입니다.

# ### 5.1.3 왜 계산 그래프로 푸는가?
# - 5.1.2 에서도 알 수 있듯이 계산 그래프의 가장 큰 장점은 국소적 계산입니다. 복잡한 계산을 계산 그래프를 통해서 국소적 계산을 보여줌으로 쉽게 와닿을 수 있다는 장점이 있습니다.
# - 중간 계산 결과를 모두 보관할 수 있습니다. 소비세 이전에 금액은 얼마고, 사과를 더하기 전에 값은 얼마인지를 알 수 있다는 것입니다.
# - 가장 중요한 이유는 역전파를 통해 **미분**을 효율적으로 계산할 수 있다는 점입니다.
# #### 역전파 
# 예시로 문제 1을 다시 갖고 오겠습니다. **사과 가격을 올린다면 최종 금액에 어떠한 영향을 끼치는지**는 **사과 가격에 대한 지불 금액의 미분**을 구하는 문제에 해당합니다.<br>
# 즉 사과 값을 x, 지불 금액을 L 이라고 할 때 $\frac{\delta L}{\delta x}$를 구하는 것입니다.<br>
# ![Screenshot 2023-08-03 at 9.00.22 AM.png](attachment:951de5f3-2a67-45f0-a4a9-08c2accee3fd.png)
# <br>계산 그래프에서 알 수 있듯이 사과 가격에 대한 지불 금액의 미분 값은 2.2, 즉 사과가 1원 오르면 최종 금액은 2.2원 오릅니다.

# ## 5.2 연쇄 법칙 (Chain Rule)
# 기존에는 왼쪽에서 오른쪽으로 갔으며 국소적 계산을 이용했지만 역전파는 이와 반대인 **국소적인 미분**을 오른쪽에서 왼쪽으로 전달하며 이것을 전달하는 원리가 바로 **연쇄 법칙** 입니다.

# #### 5.2.1 계산 그래프의 역전파
# ![Screenshot 2023-08-03 at 9.15.49 AM.png](attachment:41dd60df-6cf8-4737-8755-7f5f8892a8a1.png)<br>
# 역전파의 계산 절차는 그림과 같이 신호 E에 노드의 국소적 미분인 $\frac{\delta y}{\delta x}$를 곱한 후 다음 노드로 이 값을 전송하는 것입니다.<br>
# 위의 그림은 $y = f(x)$ 이란 과정을 그린 것이며 국소적 미분은 결국 $\frac{\delta y}{\delta x}$이며 이것을 E 값에 곱하고 다음 노드로 전송합니다.<br>
# 예시로 $y = x^2$이면 $\frac{\delta y}{\delta x} = 2x$이므로 이 값을 E에 곱해서 다음 노드로 넘깁니다.

# ### 5.2.2 연쇄 법칙 이란?
# ##### 정의 : **합성 함수의 미분은 합성 함수를 구성하는 각 함수의 미분의 곱으로 나타낼 수 있다.**
# 합성 함수란 여러 함수로 구성된 함수입니다. 예를 들어서 $z = (x+y)^2$ 이란 함수를 $z = t^2$,  $t = x + y$으로 나눌 수 있다는 것입니다. 연쇄 법칙은 이를 활용해서 다음과 같이 표현이 가능합니다.<br>
# $$\frac{\delta z}{\delta x} = \frac{\delta z}{\delta t} \frac{\delta t}{\delta x}$$
# $$\frac{\delta z}{\delta t} = 2t$$
# $$\frac{\delta t}{\delta x} = 1$$
# $$\frac{\delta z}{\delta x} = \frac{\delta z}{\delta t} \frac{\delta t}{\delta x} = 2t \cdot 1 = 2(x+y)$$

# ### 5.2.3 연쇄 법칙과 계산 그래프
# ![Screenshot 2023-08-03 at 9.30.53 AM.png](attachment:6c784432-f318-4539-af32-886c3ea453ed.png)
# <br>이것이 결국에는 $\frac{\delta z}{\delta x}$으로 성립이 되기 때문에 'x에 대한 z미분'이 됩니다. 즉 역전파는 연쇄 법칙이 하는 것과 똑같다고 생각하면 됩니다.
# ![Screenshot 2023-08-03 at 9.33.52 AM.png](attachment:4d07d629-a5c3-42c5-986d-b181c371ac8f.png)

# ## 5.3 역전파
# 여기서부터는 '+' 와 'X' 등의 연산으로 예로 들어 역전파의 구조를 설명합니다.<br>
# ### 5.3.1 덧셈 노드의 역전파
# $z = x + y$ 에서 $$\frac{\delta z}{\delta x} =  \frac{\delta z}{\delta y} = 1$$
# 오른쪽 (상류) 에서 $\frac{\delta L}{\delta z}$값이 전해진다고 가정을 하면 계산 그래프는 다음과 같습니다.
# ![Screenshot 2023-08-03 at 9.41.06 AM.png](attachment:7f58a2b8-e2ef-43c0-9955-2c4172500155.png)<br>
# ![Screenshot 2023-08-03 at 9.44.02 AM.png](attachment:291e2f9b-8f7f-45b3-b05a-c5d40990b886.png)

# ### 5.3.2 곱셈 노드의 역전파
# $z = x\cdot y$ 에서의 미분은 다음과 같습니다. $$\frac{\delta z}{\delta x} = y$$$$\frac{\delta z}{\delta y} = x$$
# ![Screenshot 2023-08-03 at 9.46.48 AM.png](attachment:6f0c2d5e-6a95-416b-85de-38198063b705.png)
# <br>
# ![Screenshot 2023-08-03 at 9.47.10 AM.png](attachment:865d6904-09e7-4159-9839-f8cf1a752ac3.png)
# <br>오른쪽 그림의 경우 상류에서 **위쪽**으로의 계산은 $1.3 \cdot 5 = 6.5$, **아래쪽**으로의 계산은 $1.3 \cdot 10 = 13$입니다.

# ### 5.3.3 사과 쇼핑의 예
# ![Screenshot 2023-08-03 at 9.49.41 AM.png](attachment:e85de640-51c8-4cd9-a905-1d04193e6b91.png)
# <br>
# ![Screenshot 2023-08-03 at 9.49.56 AM.png](attachment:0565d6b4-242a-413b-adff-195a2273fa83.png)

# ## 5.4 단순한 계층 구하기
# 여기서부터는 파이썬!<br>
# 곱셈 노드 --> 'MulLayer'<br>
# 덧셈 노드 --> 'AddLayer'<br>
# ### 5.4.1 곱셈 계층
# forward() --> 순전파<br>
# backward() --> 역전파<br>

# In[63]:


class MulLayer:
    def __init(self):
        self.x = None
        self.y = None
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        
        return out
    def backward(self, dout):
        dx = dout * self.y # x 와 y를 바꾼다
        dy = dout * self.x

        return dx, dy


# init() --> 변수 초기화<br>
# forward() --> x 와 y를 받아서 곱해서 반환<br>
# backward() --> 미분 (dout) 에 순전파 때의 값을 '서로 바꿔' 곱한 후 하류로 흘립니다.<br>
# 예시로는 사과 2개 구입하는 상황을 이용하겠습니다.<br>
# ![Screenshot 2023-08-03 at 9.56.06 AM.png](attachment:0bb790e2-c911-43e2-ae26-133746cd4885.png)

# In[64]:


apple = 100
apple_num = 2
tax = 1.1

mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)

print(price)


# In[65]:


dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print(dapple, dapple_num, dtax)


# ### 5.4.2 덧셈 계층

# In[66]:


class AddLayer:
    def __init__(self):
        pass
    def forward(self, x, y):
        out = x + y
        return out
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy


# 곱하기 1이기 때문에 바꾸는 그런 것 없이 간단하게 dout 만 return 합니다.<br>
# 예시로는 다음과 같은 상황을 이용하겠습니다.<br>
# ![Screenshot 2023-08-03 at 10.00.35 AM.png](attachment:788610a0-8372-4507-ba9b-50318f53dde8.png)

# In[67]:


apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

#계층들
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()


# In[68]:


# 순전파
apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)
all_price = add_apple_orange_layer.forward(apple_price, orange_price)
price = mul_tax_layer.forward(all_price, tax)
print (price)


# In[69]:


#역전파
dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice)
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
print(dapple_num, dapple, dorange, dorange_num, dtax)


# ## 5.5 활성화 함수 계층 구현하기
# 여기서부터는 계산 그래프를 신경망에 적용하는 것이며, ReLU 와 Sigmoid 도 활용됩니다.

# ### 5.5.1 ReLU 계층
# ReLU 의 수식 :
# $$y = x, (x > 0)$$
# $$y = 0, (x \le 0)$$
# x 에 대한 y의 미분:
# $$\frac{\delta y}{\delta x} = 1, (x > 0)$$
# $$\frac{\delta y}{\delta x} = 0, (x \le 0)$$
# ![Screenshot 2023-08-03 at 10.11.39 AM.png](attachment:c6d7be2c-45db-40ff-960b-485555a957d0.png)

# In[70]:


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


# Relu 클래스는 mask 라는 변수를 가지며 mask 는 True/False 으로 구성이 되어 있으며 순전파 입력인 x의 원소 값이 0 이하인 인덱스는 True, 그 외에는 False 를 유지하므로 예시와 같은 array 에서는 다음과 같이 출력이 됩니다.

# In[71]:


import numpy as np
x = np.array([[1.0, -0.5], [-2.0, 3.0]])
print(x)


# In[72]:


mask = (x <= 0)
print(mask)


# ### 5.5.2 Sigmoid 계층
# 시그모이드 함수는 다음과 같습니다.
# $$y = \frac{1}{1 + exp(-x)}$$
# 이것을 계산 그래프로 나타내면 다음과 같습니다.<br>
# ![Screenshot 2023-08-03 at 10.16.41 AM.png](attachment:4ca57649-a156-420d-8d12-b5d9460660ed.png)
# <br>
# 오른쪽에서 왼쪽으로 차근차근 하나씩 살피겠습니다.
# ##### **1 단계**
# '/' 노드, $y = \frac{1}{x}$ 노드 이며 미분하면 다음과 같습니다.
# $$\frac{\delta y}{\delta x} = -\frac{1}{x^2} = -y^2$$
# ##### **2 단계**
# '+' 노드, 이전과 같이 그냥 $\cdot 1$ 을 해주면 됩니다.
# ##### **3 단계**
# 'exp' 노드, $y = exp(x)$ 노드이며 미분하면 다음과 같습니다.
# $$\frac{\delta y}{\delta x} = exp(x)$$
# ##### **4 단계**
# 'X' 노드, 서로를 바꿔서 곱하는 개념이며 여기에서는 -1를 곱하면 됩니다.<br><br>
# **최종 계산 그래프:**
# ![Screenshot 2023-08-03 at 10.21.29 AM.png](attachment:21078b80-6cc3-4653-a9c0-0b8cd684e832.png)
# ![Screenshot 2023-08-03 at 10.21.40 AM.png](attachment:579450a4-5c79-4605-95a1-285719c416fd.png)

# 시그모이드의 최종 공식, 즉 순전파의 출력 (y) 만으로의 계산:
# ![Screenshot 2023-08-03 at 10.24.42 AM.png](attachment:dc2a43a0-8117-4ffd-8909-bbea5881340a.png)
# ![Screenshot 2023-08-03 at 10.25.43 AM.png](attachment:f0645259-9b5d-47b7-a4a5-5f20db17fedc.png)

# 위의 계산을 코드를 구현하겠습니다.

# In[73]:


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1/(1+np.exp(-x))
        self.out = out
        return out
    
    def backward(self, dout):
        dx = dout * (1.0 - self.out)*self.out
        return dx


# ## 5.6 Affine / Softmax 계층 구현하기
# ### 5.6.1 Affine 계층
# ##### 어파인 변환 (Affine Transformation):
# 신경망의 순전파 때 수행하는 행렬의 곱을 기하학에서는 어파인 변환이라고 불립니다.

# np.dot()에 대한 복습 (행렬들의 곱)

# In[74]:


X = np.random.rand(2) # 입력
W = np.random.rand(2, 3) # 가중치
B = np.random.rand(3) # 편향
print("X's shape :", X.shape)
print("W's shape :", W.shape)
print("B's shape :", B.shape)

Y = np.dot(X, W) + B
print(Y)


# ![Screenshot 2023-08-03 at 10.31.42 AM.png](attachment:b4839989-f677-468b-ae77-a74182b074eb.png)
# **주의점** : 변수들이 행렬입니다!

# 행렬을 이용하기 때문에 역전파의 계산법도 살짝 달라지게 됩니다. 하지만 선형대수의 고수들인 여러분들에게는 쉽다고 생각하겠습니다 (저는 참고로 선대 개못함~).
# $$\frac{\delta L}{\delta \textbf{X}} = \frac{\delta L}{\delta \textbf{Y}} \cdot \textbf{W}^T$$
# $$\frac{\delta L}{\delta \textbf{W}} = \textbf{X}^T \cdot \frac{\delta L}{\delta \textbf{Y}}$$
# ![Screenshot 2023-08-03 at 10.34.43 AM.png](attachment:64718a50-d0ef-45a5-b50b-261f57021953.png)
# 헷갈린 만한거 : **X** 와 $\frac{\delta L}{\delta \textbf{X}}$ 은 같은 형상일까?<br>
# ![Screenshot 2023-08-03 at 10.39.02 AM.png](attachment:122c2bc5-143c-4345-9dc0-f490dd80775a.png)

# ### 5.6.2 배치용 Affine 계층
# 위의 경우 입력 데이터로 **X** 하나만을 고려한 것이며 여기서부터는 데이터 N개를 묶어서 순전파하는 경우를 살피겠습니다. 이 경우를 배치용 Affine 게층이라고 불립니다.<br>
# ![Screenshot 2023-08-03 at 10.40.22 AM.png](attachment:cd3d9704-cb3a-4c59-8e5c-fb74b9768291.png)

# (N, ?) 의 형태를 따른 것 이외에는 차이가 크게 없지만 편향에 대해서 조심을 해야 합니다.<br>
# 예시로 N = 2로 한 경우, 편향은 그 두 데이터 각각에 더해집니다. 구체적으로는 :

# In[75]:


X_dot_W = np.array([[0, 0, 0], [10, 10, 10]])
B = np.array([1, 2, 3])
print(X_dot_W)


# In[76]:


print(X_dot_W + B)


# 순전파의 편향 덧셈은 각각의 데이터에 더해지므로 역전파 때는 각 데이터의 역전파 값이 편향의 원소에 모여야 합니다.

# In[77]:


dY = np.array([[1, 2, 3], [4, 5, 6]])
print(dY)


# In[78]:


dB = np.sum(dY, axis = 0)
print(dB)


# 이제는 Affine 자체를 구현하겠습니다.

# In[79]:


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(selfx.T, dout)
        self.db = np.sum(dout, axis = 0)
        return dx


# ### 5.6.3 Softmax-with-Loss 계층
# 소프트맥스 함수는 입력 값을 정규화하여 출력합니다.
# ![Screenshot 2023-08-03 at 10.47.01 AM.png](attachment:e736c210-cee8-4b94-8e0c-b01d56ed00ba.png)
# 위의 그림처럼 소프트맥스 함수가 언제 사용되는지 알 수가 있으며, 이제는 소프트맥스 계층을 구현할 것이며 보기 편하게 다음을 참조하면 될 것 같습니다.
# ![Screenshot 2023-08-03 at 10.48.34 AM.png](attachment:c2f035f2-adde-47cf-a2de-b821274bbe2e.png)
# 

# Softmax 계층은 (a1, a2, a3)를 정규화하여 (y1, y2, y3)를 출력합니다.<br>
# Cross Entropy Error 계층은 (y1, y2, y3) 과 정답 레이블 (t1, t2, t3) 를 받으며 이로 인한 손실인 L 을 출력합니다.

# 역전파 결과에 대해서입니다.<br>
# Softmax 게층의 역전파는 (y1 - t1, y2 - t2, y3 - t3) 이라는 결과를 내놓고 있으며 이것은 Softmax 계층의 출력과 정답 레이블의 차분인 것입니다. 

# 구체적인 예시 :<br>
# 정답 레이블이 (0, 1, 0)일 때 Softmax 계층이 (0.3, 0.2, 0.5)를 출력했다고 가정<br>
# Softmax 계층의 역전파는 (0.3, -0.8, 0.5)라는 커다란 오차를 전파하게 되므로 Softmax 계층들의 **앞 계층**들은 그 큰 오차로부터 큰 깨달음을 얻게 됩니다.

# In[80]:


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss
    def backward(self, dout = 1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx


# ## 5.7 오차역전파법 구현하기
# 여기에서는 지금까지 구현한 계층들을 조합해서 신경망을 구축하는 것입니다.
# 
# ### 5.7.1 신경망 학습의 전체 그림
# ##### **전체**
# 신경망에는 적응 가능한 가중치와 편향이 있고, 이 가중치와 편향을 훈련 데이터에 적응하도록 조정하는 과정을 '학습'이라 합니다. 신경망 학습은 다음과 같이 4단계로 수행합니다.
# ##### **1단계 - 미니배치**
# 훈련 데이터 중 일부를 무작위로 가져옵니다. 이렇게 선별한 데이터를 미니배치라 하며, 그 미니배치의 손실 함수 값을 줄이는 것이 목표입니다.
# ##### **2단계 - 기울기 산출**
# 미니배치의 손실 함수 값을 줄이기 위해 각 가중치 매개변수의 기울기를 구합니다. 기울기는 손실 함수의 값을 가장 적게 하는 방향을 제시합니다.
# ##### **3단계 - 매개변수 갱신**
# 가중치 매개변수를 기울기 방향으로 아주 조금 갱신합니다.
# ##### **4단계 - 반복**
# 1~3 단계를 반복합니다.

# ### 5.7.2 오차역전파법을 적용한 신경망 구현하기
# TwoLayerNet --> 2층 신경망 클래스<br>
# 다음은 해당 클래스의 변수들과 메서드들에 대한 설명입니다.
# ![Screenshot 2023-08-03 at 11.03.30 AM.png](attachment:6f762de2-7e46-43a3-8bfe-20856113882c.png)

# In[81]:


import sys, os
sys.path.append(os.pardir)
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        #가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # 계층 생성
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
            return x
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis = 1)
        if t.ndim != 1:
            t = np.argmax(t, axis = 1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads

    def gradient(self, x, t):
        self.loss(x, t)
        
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine1'].dW
        grads['b2'] = self.layers['Affine1'].db

        return grads


# ### 5.7.3 오차역전파법으로 구한 기울기 검증하기
# 수치 미분 방법으로 여태까지 기울기를 구했지만 매우 느리다는 단점이 있기 때문에 오차역전파법을 통해서 구현을 하겠습니다.<br>
# 수치 미분이 있는 이유는 오차역 전파법을 정확하게 구현했는지 확인하기 위해 필요합니다.<br><br>
# 수치 미분의 이점은 구현하기 쉽다는 것이기 때문에 버그가 숨어있기가 어려운 반면에 오차역전파법은 구현하기 복잡하기 때문에 실수가 자주 생깁니다. 그러므로 수치 미분의 결과와 오차역 전파법의 결과를 비교하여 오차역전파법을 제대로 구현했는지 검증을 합니다. 이와 같은 두 방식으로 구한 기울기가 일치한지를 확인하는 작업을 **기울기 확인** (Gradient Check) 이라고 합니다.

# 다음 코드에서는 각 가중치 매개변수의 차이의 절댓값을 구하고, 이를 평균한 값이 오차가 됩니다.<br>
# 즉 출력으로는 수치 미분과 오차역전파법 방법을 사용해서 나온 값들의 차이라고 보면 됩니다.

# In[82]:


import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

# 각 가중치의 절대 오차의 평균을 구한다.
for key in grad_numerical.keys():
    diff = np.average( np.abs(grad_backprop[key] - grad_numerical[key]) )
    print(key + ":" + str(diff))


# ### 5.7.4 오차역 전파법을 사용한 학습 구현하기
# 마지막으로는 오차역 전파법을 통해서 신경망 학습을 구현하는 것입니다. 이전 장과 똑같으며 그저 오차역 전파법을 이용했다는 차이가 있습니다.

# In[83]:


# coding: utf-8
import sys, os
sys.path.append(os.pardir)

import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 기울기 계산
    #grad = network.numerical_gradient(x_batch, t_batch) # 수치 미분 방식
    grad = network.gradient(x_batch, t_batch) # 오차역전파법 방식(훨씬 빠르다)
    
    # 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)


# ## 5.8 정리
# ![Screenshot 2023-08-03 at 11.45.53 AM.png](attachment:3e38813a-8f38-4faa-910a-3465b157a461.png)

# # Chapter 6. | 학습 관련 기술들

# ## 6.1 매개변수 갱신
# **최적화** (Optimization) : 매개 변수의 최적값을 찾는 문제들을 최적화라 부릅니다.<br>
# **확률적 경사 하강법** (SGD) : 여태까지 최적의 매개변수 값을 매개변수의 기울기 (미분)을 이용해서 기울어진 방향으로 매개변수 값을 갱신하는 일의 반복을 통해서 최적의 값에 다가가는 방법.

# ### 6.1.1 모험가 이야기
# 색다른 모험가가 있습니다. 광활한 메마른 산맥을 여행하면서 날마다 깊은 골짜기를 찾아 발걸음을 옮깁니다. 그는 전설에 나오는 세상에서 가장 깊고 낮은 골짜기, '깊은 곳'을 찾아가려 합니다. 그것이 그의 여행 목적이죠. 게다가 그는 엄격한 '제약' 2개로 자신을 옭아맸습니다. 하나는 지도를 보지 않을 것, 또 하나는 눈가리개를 쓰는 것입니다. 지도도 없고 보이지도 않으니 가장 낮은 골짜기가 광대한 땅 어디에 있는지 알 도리가 없죠. 그런 혹독한 조건에서 이 모험가는 어떻게 '깊은 곳'을 찾을 수 있을까요? 어떻게 걸음을 옮겨야 효율적으로 '깊은 곳'을 찾아낼 수 있을까요?
# <br><br>
# 광대하고 복잡한 지형을 지도도 없이 눈을 가린 채로 '깊은 곳을 찾지 않으면 안 됩니다. 그러므로 단서인 땅의 '기울기'를 이용해야 합니다. 지금 서 있는 장소에서 가장 크게 기울어진 방향으로 가는 것이 SGD 의 전략입니다.

# ### 6.1.2 확률적 경사 하강법 (SGD)
# $$\textbf{W} \leftarrow \textbf{W} - \eta \frac{\partial L}{\partial \textbf{W}}$$
# $\textbf{W}$ : 갱신할 가중치 매개변수<br>
# $\frac{\partial L}{\partial \textbf{W}}$ : $\textbf{W}$에 대한 손실 함수의 기울기<br>
# $\eta$ : 학습률 (Learning Rate)

# In[84]:


class SGD:
    def __init__(self, lr = 0.01):
        self.lr = lr
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


# SGD 가 코드에서 사용되는 때는 최적화 때 사용되며 예시는 다음과 같으며, 코드는 어디에서 작동되는지에 대한 예시므로 실행하지 마세요!
network = TwoLayerNet(...)
optimizer = SGD()
for i in range(10000):
    ...
    x_batch, t_batch = get_mini_batch(...)
    grads = network.gradient(x_batch, t_batch)
    params = network.params
    optimizer.update(params, grads)
    ...
# ### 6.1.3 SGD의 단점
# 여기에서는 다음 함수의 최솟값을 구하는 문제를 생각해보겠습니다.
# $$f(x, y) = \frac{1}{20} x^2 + y^2$$
# 다음은 함수의 그래프와 그 등고선을 보여주고 있습니다.
# ![Screenshot 2023-08-03 at 12.06.27 PM.png](attachment:bd3d8092-84cf-4bde-bd41-219536c30a79.png)
# <br>
# 기울기는 y 축 방향은 크고 x축 방향은 작다는 것이 특징입니다.<br>
# 최솟값이 되는 장소는 $(x, y) = (0, 0)$ 이지만 다음 그림은 기울기 대부분이 (0, 0) 방향을 가리키고 있지 않습니다.
# <br>
# ![Screenshot 2023-08-03 at 12.06.43 PM.png](attachment:519e6090-9e91-40e0-bbf7-3b1f959751e1.png)
# 이러한 특징을 가지고 초깃값 $(x, y) = (-7.0, 2.0)$ 을 가지면서 SGD를 적용시키면 다음과 같이 결과가 나옵니다.
# ![Screenshot 2023-08-03 at 12.11.47 PM.png](attachment:7e55b7e0-76c9-4084-96fb-19425f8a9c1f.png)
# 지그재그로 움직이기 때문에 상당히 비효율적인 움직임을 보입니다. 원인은 기울어진 방향이 본래의 최솟값과 다른 방향을 가리켜서라는 점입니다.
# ##### 비등방성 (Anisotropy) 함수 : 방향에 따라 성질, 혹은 기울기가 달라지는 함수)
# 결론적으로 SGD 의 단점은 비등방성 함수에서는 탐색 경로가 비효율적이라는 것입니다.

# ### 6.1.4 모멘텀 (Momentum)
# 모멘텀은 운동량이란 뜻이며 물리에서 자주 사용됩니다. 사용할 수식들은 다음과 같습니다.
# $$\textbf{v} \leftarrow \alpha \textbf{v} - \eta \frac{\partial L}{\partial \textbf{W}}$$
# $$\textbf{W} \leftarrow \textbf{W} + \textbf{v}$$
# $\textbf{W}$ : 갱신할 가중치 매개변수<br>
# $\frac{\partial L}{\partial \textbf{W}}$ : $\textbf{W}$ 에 대한 손실 함수의 기울기<br>
# $\eta$ : 학습률<br>
# $\textbf{v}$ : 물리에서 말하는 속도 (Velocity)에 해당합니다.
# $\alpha \textbf{v}$ : 물체가 아무런 힘을 받지 않을 때 서서히 하강시키는 역할이며 $\alpha$는 0.9 등의 값으로 설장합니다.
# ![Screenshot 2023-08-03 at 12.17.51 PM.png](attachment:7e131322-8146-4f1e-b309-d606e939b0bf.png)
# 

# In[86]:


class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
        
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():                                
                self.v[key] = np.zeros_like(val)
                
        for key in params.keys():
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key] 
            params[key] += self.v[key]



# 위의 예시에서 SGD 말고 Momentum 을 적용시키면 다음과 같은 결과가 나옵니다.
# ![Screenshot 2023-08-03 at 12.20.19 PM.png](attachment:b4749079-10d0-4ddd-8840-6e3853e8f24c.png)
# SGD 에 비해서 지그재그 정도가 덜하며 이에 대한 원인은 x축의 힘은 아주 작지만 방향은 변하지 않아서 한 방향으로 일정하게 가속하기 때문입니다.<br><br>
# 다르게 보면 y축의 힘은 크지만 위 아래로 번갈아 받아서 상층하여 y축 방향의 속도는 안정적이지 않습니다.<br><br>
# 하지만 그래도 SGD 보다는 지그재그 정도가 훨씬 적은 모습을 보이고 있습니다.

# ### 6.1.5 AdaGrad
# 신경망 학습에서는 학습률 값이 매우 중요합니다.<br><br>
# 이 값이 너무 크면 발산하여 학습이 제대로 안됩니다.<br>
# 반면에 너무 작으면 학습 시간이 너무 길어집니다.<br><br>
# 이러한 문제들을 감안해서 학습률을 정하는 기술이 바로 **학습률 감소** (Learning Rate Decay) 입니다.<br>
# 학습을 진행하면서 학습률을 점차 줄여가는 방법으로, 처음에는 크게 학습하다가 조금씩 작게 학습한다는 방법이 신경망 학습에서 자주 쓰이는 방법입니다.
# ##### Ada Grad
# 학습률을 서서히 낮추는 가장 간단한 방법은 매개변수 '전체'의 학습률 값을 일괄적으로 낮추는 것이며 이것을 더욱 발전한 것이 AdaGrad 입니다. AdaGrad 은 **각각의 매개변수에 맞춰진 값**들을 만들어주며 개별 매개변수에 적응적으로 학습률을 조정하면서 학습을 진행합니다.<br><br>
# 수식은 다음과 같습니다.
# $$\textbf{h} \leftarrow \textbf{h} + \frac{\partial L}{\partial \textbf{W}} \odot \frac{\partial L}{\partial \textbf{W}}$$
# $$\textbf{W} \leftarrow \textbf{W} - \eta \frac{1}{\sqrt{\textbf{h}}} \frac{\partial L}{\partial \textbf{W}}$$
# $\textbf{W}$ : 갱신할 가중치 매개변수<br>
# $\frac{\partial L}{\partial \textbf{W}}$ : $\textbf{W}$ 에 대한 손실 함수의 기울기<br>
# $\eta$ : 학습률<br>
# $\odot$ : 행렬의 원소별 곱셈<br>
# $\textbf{h}$ : 기존 기울기 값을 제곱하여 더하는 것<br><br>
# $\textbf{h}$ 의 공식에서 알 수 있듯이 결국 AdaGrad는 과거의 기울기를 제곱하여 계속 더해가는 것이기 때문에 학습을 진행할수록 갱신 강도가 약해집니다.<br><br>
# 다음은 AdaGrad의 구현 코드입니다.

# In[87]:


class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None
        
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
            
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


# 마지막 줄에 **1e-7** 이 있는 것은 0으로 나누는 사태를 막아주는 역할입니다.<br><br>
# 위의 예시를 AdaGrad 을 이용해서 최적화를 하게 된다면 결과는 다음과 같습니다.
# ![Screenshot 2023-08-03 at 12.38.25 PM.png](attachment:2f96f3ce-3eeb-49ad-946b-5f4fa355f1d5.png)
# 그림에서도 보이듯이 최솟값을 향해서 가장 효율적으로 움직이는 것을 보여줍니다. y 축 방향은 기울기가 커서 처음에는 크게 움직이지만 그 큰 움직임과 비례해서 작아지는 모습을 보입니다.

# ### 6.1.6 Adam
# 모멘텀과 AdaGrad 를 융합한게 Adam 이라고 생각하면 됩니다. 구현 코드는 다음과 같습니다.

# In[88]:


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        
        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         
        
        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)


# 위의 예시 문제를 풀면 다음과 같습니다.
# ![Screenshot 2023-08-03 at 12.43.14 PM.png](attachment:e39a2ef5-6a5d-4384-a9c8-a4f1daace2ff.png)
# 교재에서는 큰 설명은 없으며 다음은 ChatGPT 에서 검색해서 나온 결과입니다.
# ![Screenshot 2023-08-03 at 12.48.20 PM.png](attachment:e584f93b-f28d-4312-9a38-f606dcc694fc.png)

# ### 6.1.7 어느 갱신 방법을 이용할 것인가?

# In[89]:


import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from common.optimizer import *
def f(x, y):
    return x**2 / 20.0 + y**2


def df(x, y):
    return x / 10.0, 2.0*y

init_pos = (-7.0, 2.0)
params = {}
params['x'], params['y'] = init_pos[0], init_pos[1]
grads = {}
grads['x'], grads['y'] = 0, 0

optimizers = OrderedDict()
optimizers["SGD"] = SGD(lr=0.95)
optimizers["Momentum"] = Momentum(lr=0.1)
optimizers["AdaGrad"] = AdaGrad(lr=1.5)
optimizers["Adam"] = Adam(lr=0.3)
idx = 1

for key in optimizers:
    optimizer = optimizers[key]
    x_history = []
    y_history = []
    params['x'], params['y'] = init_pos[0], init_pos[1]    
    for i in range(30):
        x_history.append(params['x'])
        y_history.append(params['y'])
        
        grads['x'], grads['y'] = df(params['x'], params['y'])
        optimizer.update(params, grads)
    x = np.arange(-10, 10, 0.01)
    y = np.arange(-5, 5, 0.01)
    
    X, Y = np.meshgrid(x, y) 
    Z = f(X, Y)
    
    # 외곽선 단순화
    mask = Z > 7
    Z[mask] = 0
    
    # 그래프 그리기
    plt.subplot(2, 2, idx)
    idx += 1
    plt.plot(x_history, y_history, 'o-', color="red")
    plt.contour(X, Y, Z)
    plt.ylim(-10, 10)
    plt.xlim(-10, 10)
    plt.plot(0, 0, '+')
    #colorbar()
    #spring()
    plt.title(key)
    plt.xlabel("x")
    plt.ylabel("y")
    
plt.show()


# 결국에 네개 중에서 어떤게 가장 나은 방법이라는 정답은 없으며, 이것은 하이퍼파라미터를 어떻게 설정하느냐, 혹은 풀어야 할 문제가 무엇이냐에 따라서 달라집니다.

# ### 6.1.8 MNIST 데이터셋으로 본 갱신 방법 비교

# In[90]:


# coding: utf-8
import os
import sys
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.util import smooth_curve
from common.multi_layer_net import MultiLayerNet
from common.optimizer import *


# 0. MNIST 데이터 읽기==========
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

train_size = x_train.shape[0]
batch_size = 128
max_iterations = 2000


# 1. 실험용 설정==========
optimizers = {}
optimizers['SGD'] = SGD()
optimizers['Momentum'] = Momentum()
optimizers['AdaGrad'] = AdaGrad()
optimizers['Adam'] = Adam()
#optimizers['RMSprop'] = RMSprop()

networks = {}
train_loss = {}
for key in optimizers.keys():
    networks[key] = MultiLayerNet(
        input_size=784, hidden_size_list=[100, 100, 100, 100],
        output_size=10)
    train_loss[key] = []    


# 2. 훈련 시작==========
for i in range(max_iterations):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    for key in optimizers.keys():
        grads = networks[key].gradient(x_batch, t_batch)
        optimizers[key].update(networks[key].params, grads)
    
        loss = networks[key].loss(x_batch, t_batch)
        train_loss[key].append(loss)
    
    if i % 100 == 0:
        print( "===========" + "iteration:" + str(i) + "===========")
        for key in optimizers.keys():
            loss = networks[key].loss(x_batch, t_batch)
            print(key + ":" + str(loss))


# 3. 그래프 그리기==========
markers = {"SGD": "o", "Momentum": "x", "AdaGrad": "s", "Adam": "D"}
x = np.arange(max_iterations)
for key in optimizers.keys():
    plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 1)
plt.legend()
plt.show()


# AdaGrad 가 가장 빨라 보이지만 이것은 하이퍼파라미터인 학습률 (Learning Rate)과 신경망의 구조 (층, 깊이) 에 따라 결과가 달라집니다. 하지만 일반적으로는 SGD 가 가장 느리며 가끔씩은 SGD 만큼이나 높은 최종 좡확도를 보일 때도 있습니다.

# ## 6.2 가중치의 초깃값

# ### 6.2.1 초깃값을 0으로 하면?
# **가중치 감소** (Weight Decay) : 가중치 매개변수의 값이 작아지도록 학습하는 방법, 즉 가중치 값을 작게 하여 오버피팅이 일어나지 않게 하는 것.<br>
# ##### 가중치의 초깃값을 0으로 설정하면 학습이 올바로 이뤄지지 않습니다.
# 오차역 전파법에서 모든 가중치의 값이 똑같이 갱신되기 때문입니다. 0을 초깃값으로 하게 된다면 역전파 때 가중치가 모두 똑같이 갱신이 됩니다. 그래서 가중치의 대칭적인 구조를 무너뜨리기 위해서 초깃값을 무작위로 설정해야 합니다.

# ### 6.2.2 은닉층의 활성화값 분포
# 여기에서는 가중치의 초깃값에 따라 은닉층 활성화값들이 어떻게 변화하는지에 대한 실험을 진행할 예정입니다. 가중치의 분포는 정규분포를 이용하였는데 표준편차의 차이에 따라서 활성화값의 분포가 어떻게 바뀌는지 살필 것입니다.

# In[91]:


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
x = np.random.randn(1000, 100)  # 1000개의 데이터
node_num = 100  # 각 은닉층의 노드(뉴런) 수
hidden_layer_size = 5  # 은닉층이 5개
activations = {}  # 이곳에 활성화 결과를 저장

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]
    w = np.random.randn(node_num, node_num) * 1
    a = np.dot(x, w)
    z = sigmoid(a)
    activations[i] = z

# 히스토그램 그리기
for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) + "-layer")
    plt.hist(a.flatten(), 30, range=(0,1))
plt.show()


# 각 층의 활성값들이 0과 1에 치우쳐져 있습니다. 시그모이드 함수의 출력은 0 혹은 1에 가까워지면 미분값이 0에 다가갑니다. 그렇기 때문에 데이터가 0과 1에 치우쳐 분포하게 된다면 역전파의 기울기 값이 점점 작아지다가 사라집니다. 이러한 현상을 **기울기 소실** (Gradient Vanishing) 이라 알려진 문제입니다.

# 이번에는 가중치의 표준편차를 0.01로 바꾸어서 실험을 반복하겠습니다.

# In[92]:


x = np.random.randn(1000, 100)  # 1000개의 데이터
node_num = 100  # 각 은닉층의 노드(뉴런) 수
hidden_layer_size = 5  # 은닉층이 5개
activations = {}  # 이곳에 활성화 결과를 저장

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]
    w = np.random.randn(node_num, node_num) * 0.01
    a = np.dot(x, w)
    z = sigmoid(a)
    activations[i] = z

# 히스토그램 그리기
for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) + "-layer")
    plt.hist(a.flatten(), 30, range=(0,1))
plt.show()


# 이번에는 0과 1로 치우치진 않아서 기울기 소실 문제는 일어나지 않았으나 다수의 뉴런이 거의 같은 값을 출력하고 있으니 뉴런을 여러 개 둔 의미가 없어지게 됐습니다. 이 문제는 **표현력을 제한**한다는 관점에서 문제가 됩니다.

# 다음으로는 **Xavier 초깃값**에 대해서 살피겠습니다.
# ![Screenshot 2023-08-03 at 1.30.38 PM.png](attachment:5014931f-6d4d-44ef-b2ee-71bf6280380d.png)
# 쉽게 말해서 표준편차를 $\frac{1}{\sqrt{n}}$ 인 분포를 사용한 것입니다.

# In[93]:


x = np.random.randn(1000, 100)  # 1000개의 데이터
node_num = 100  # 각 은닉층의 노드(뉴런) 수
hidden_layer_size = 5  # 은닉층이 5개
activations = {}  # 이곳에 활성화 결과를 저장

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]
    w = np.random.randn(node_num, node_num) / np.sqrt(node_num)
    a = np.dot(x, w)
    z = sigmoid(a)
    activations[i] = z

# 히스토그램 그리기
for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) + "-layer")
    plt.hist(a.flatten(), 30, range=(0,1))
plt.show()


# 위의 Histogram 과 같이 넓게 분포됨을 알 수가 있으므로 시그모이드 함수의 표현력에 제한받지 않으면서 학습이 효율적으로 이뤄질 것으로 기대됩니다.

# ### 6.2.3 ReLU를 사용할 때의 가중치 초깃값
# Xavier 초깃값은 활성화 함수가 선형인 것을 전제로 이끈 결과이지만 ReLU를 이용할 때는 ReLU 에 특화된 초깃값을 이용하라고 권장합니다. 이 특화된 초깃값을 **He 초깃값**이라고 불리며 앞 계층의 노드가 n개일 때, 표준편차 $\sqrt{\frac{2}{n}}$ 인 정규분포를 이용합니다. Xavier 초깃값과는 다르게 ReLU는 음의 영역이 0이라서 더 넓게 분포시키기 위해서 2배의 계수가 필요하다고 해석하면 됩니다.
# <br><br>
# 다음은 활성화 함수로 ReLU를 사용한 경우의 가중치 초깃갑에 따른 활성화값 분포 변화입니다.

# ##### 표준편차 = 0.01

# In[94]:


def ReLU(x):
    return np.maximum(0, x)
    
x = np.random.randn(1000, 100)  # 1000개의 데이터
node_num = 100  # 각 은닉층의 노드(뉴런) 수
hidden_layer_size = 5  # 은닉층이 5개
activations = {}  # 이곳에 활성화 결과를 저장

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]
        
    w = np.random.randn(node_num, node_num) * 0.01
    a = np.dot(x, w)
    z = ReLU(a)
    activations[i] = z

# 히스토그램 그리기
for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) + "-layer")
    if i != 0: 
        plt.yticks([], [])
        plt.xlim(0.1, 1)
        plt.ylim(0, 7000)
    plt.hist(a.flatten(), 30, range=(0,1))
plt.show()


# 각 층의 활성화 값들은 아주 작은 값들이며, 역전파 때 가중치의 기울기 역시 작아진다는 뜻입니다. 즉 학습이 거의 이뤄지지 않을 것입니다.

# ##### 표준편차 = Xavier 초깃값

# In[95]:


x = np.random.randn(1000, 100)  # 1000개의 데이터
node_num = 100  # 각 은닉층의 노드(뉴런) 수
hidden_layer_size = 5  # 은닉층이 5개
activations = {}  # 이곳에 활성화 결과를 저장

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]
        
    w = np.random.randn(node_num, node_num) / np.sqrt(node_num)
    a = np.dot(x, w)
    z = ReLU(a)
    activations[i] = z

# 히스토그램 그리기
for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) + "-layer")
    if i != 0: 
        plt.yticks([], [])
        plt.xlim(0.1, 1)
        plt.ylim(0, 7000)
    plt.hist(a.flatten(), 30, range=(0,1))
plt.show()


# 층이 깊어지면서 치우침이 조금씩 커집니다. 이것은 학습할 때 '기울기 소실' 문제를 일으킵니다.

# ##### 표준편차 = He 초깃값

# In[96]:


x = np.random.randn(1000, 100)  # 1000개의 데이터
node_num = 100  # 각 은닉층의 노드(뉴런) 수
hidden_layer_size = 5  # 은닉층이 5개
activations = {}  # 이곳에 활성화 결과를 저장

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]
        
    w = np.random.randn(node_num, node_num) / np.sqrt(node_num) * 2
    a = np.dot(x, w)
    z = ReLU(a)
    activations[i] = z

# 히스토그램 그리기
for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) + "-layer")
    if i != 0: 
        plt.yticks([], [])
        plt.xlim(0.1, 1)
        plt.ylim(0, 7000)
    plt.hist(a.flatten(), 30, range=(0,1))
plt.show()


# 모든 층에서 균일하게 분포되어있으며 층이 깊어져도 분포가 균일하게 유지되기에 역전파 떄도 적절한 값이 나올거라고 기대할 수 있습니다.

# ##### 결론
# ReLU --> He 초깃값
# Sigmoid / tanh 등의 S 자 모양 곡선일 때 --> Xavier 초깃값

# ### 6.2.4 MNIST 데이터셋으로 본 가중치 초깃값 비교
# 이번에는 실제 데이터를 이용해서 가중치의 초깃값이 학습에 얼마나 영향을 주는지 보겠습니다.

# In[97]:


sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from dataset.mnist import load_mnist
from common.util import smooth_curve
from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD

# 0. MNIST 데이터 읽기==========
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

train_size = x_train.shape[0]
batch_size = 128
max_iterations = 2000


# 1. 실험용 설정==========
weight_init_types = {'std=0.01': 0.01, 'Xavier': 'sigmoid', 'He': 'relu'}
optimizer = SGD(lr=0.01)

networks = {}
train_loss = {}
for key, weight_type in weight_init_types.items():
    networks[key] = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100],
                                  output_size=10, weight_init_std=weight_type)
    train_loss[key] = []


# 2. 훈련 시작==========
for i in range(max_iterations):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    for key in weight_init_types.keys():
        grads = networks[key].gradient(x_batch, t_batch)
        optimizer.update(networks[key].params, grads)
    
        loss = networks[key].loss(x_batch, t_batch)
        train_loss[key].append(loss)
    
    if i % 100 == 0:
        print("===========" + "iteration:" + str(i) + "===========")
        for key in weight_init_types.keys():
            loss = networks[key].loss(x_batch, t_batch)
            print(key + ":" + str(loss))


# 3. 그래프 그리기==========
markers = {'std=0.01': 'o', 'Xavier': 's', 'He': 'D'}
x = np.arange(max_iterations)
for key in weight_init_types.keys():
    plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 2.5)
plt.legend()
plt.show()


# std = 0.01 : 활성화값의 분포가 0 근처로 밀집해서 학습이 거의 이루어지지 않습니다.<br>
# std = Xavier 초깃값 : 기울기 소실 문제 때문에 He 초깃값보다는 학습이 느린 모습.<br>
# std = He 초깃값 : 학습이 가장 순조롭게 이루어지고 있습니다.

# ## 6.3 배치 정규화 (Batch Normalization)
# 아이디어 : 각 층의 활성화값 분포가 적당히 퍼지면서 학습이 수월하게 수행된다면 각 층이 활성화를 적당히 퍼뜨리도록 '강제'시키는 것

# ### 6.3.1 배치 정규화 알고리즘

# 주목 받는 이유들 :
# 
# - 학습을 빨리 진행할 수 있다 (학습 속도 개선)
# - 초깃값에 크게 의존하지 않는다
# - 오버피팅을 억제한다
# 
# 원리 : 각 층에서의 활성화 값이 적당히 분포되도록 조정하는 것
# ![Screenshot 2023-08-03 at 1.57.48 PM.png](attachment:44bf29cc-5a0b-4adb-bd98-8d8cb430792d.png)
# 배치 정규화는 학습 시 미니배치를 단위로 평균이 0, 분산이 1이 되도록 정규화 합니다. 수식으로는 다음과 같습니다.
# $$\mu_B \leftarrow \frac{1}{m}\sum^m_{i=1} x_i$$
# $$\sigma_B^2 \leftarrow \frac{1}{m}\sum^m_{i=1} (x_i - \mu_B)^2$$
# $$\hat{x}_i \leftarrow \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
# $B = {x_1, x_2, ..., x_m}$ : m개의 입력데이터의 집합인 미니배치<br>
# $\mu_B$ : 평균<br>
# $\sigma_B^2$ : 분산<br>
# $\epsilon$ : 0으로 나누는 사태를 예방하는 역할로, 보통 10e-7 등의 값을 넣습니다<br><br>
# 추가적으로 정규화된 데이터에 고유한 확대 (Scale) 와 이동 (Shift) 변환을 수행합니다. 수식으로는 다음과 같습니다.
# $$y_i \leftarrow \gamma \hat{x}_i + \beta$$
# $\gamma$ : 확대 역할, (처음엔 1)<br>
# $\beta$ : 이동 역할, (처음엔 0)<br>
# 학습하면서 둘다 적합한 값으로 조정해갑니다.<br><br>
# 계산 그래프를 통해서 다음과 같이 보입니다.
# ![Screenshot 2023-08-03 at 2.06.23 PM.png](attachment:7525ac7b-10f4-4c45-9a00-aa34228e7ab6.png)

# ### 6.3.2 배치 정규화의 효과
# MNIST 데이터셋을 이용해서 배치 정규화 계층을 이용한 것과 안한 것의 학습 진도 차이를 보여주겠습니다.

# In[98]:


from dataset.mnist import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.optimizer import SGD, Adam

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 학습 데이터를 줄임
x_train = x_train[:1000]
t_train = t_train[:1000]

max_epochs = 20
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.01


def __train(weight_init_std):
    bn_network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100], output_size=10, 
                                    weight_init_std=weight_init_std, use_batchnorm=True)
    network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100], output_size=10,
                                weight_init_std=weight_init_std)
    optimizer = SGD(lr=learning_rate)
    
    train_acc_list = []
    bn_train_acc_list = []
    
    iter_per_epoch = max(train_size / batch_size, 1)
    epoch_cnt = 0
    
    for i in range(1000000000):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
    
        for _network in (bn_network, network):
            grads = _network.gradient(x_batch, t_batch)
            optimizer.update(_network.params, grads)
    
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            bn_train_acc = bn_network.accuracy(x_train, t_train)
            train_acc_list.append(train_acc)
            bn_train_acc_list.append(bn_train_acc)
    
            print("epoch:" + str(epoch_cnt) + " | " + str(train_acc) + " - " + str(bn_train_acc))
    
            epoch_cnt += 1
            if epoch_cnt >= max_epochs:
                break
                
    return train_acc_list, bn_train_acc_list


# 그래프 그리기==========
weight_scale_list = np.logspace(0, -4, num=16)
x = np.arange(max_epochs)

for i, w in enumerate(weight_scale_list):
    print( "============== " + str(i+1) + "/16" + " ==============")
    train_acc_list, bn_train_acc_list = __train(w)
    
    plt.subplot(4,4,i+1)
    plt.title("W:" + str(w))
    if i == 15:
        plt.plot(x, bn_train_acc_list, label='Batch Normalization', markevery=2)
        plt.plot(x, train_acc_list, linestyle = "--", label='Normal(without BatchNorm)', markevery=2)
    else:
        plt.plot(x, bn_train_acc_list, markevery=2)
        plt.plot(x, train_acc_list, linestyle="--", markevery=2)

    plt.ylim(0, 1.0)
    if i % 4:
        plt.yticks([])
    else:
        plt.ylabel("accuracy")
    if i < 12:
        plt.xticks([])
    else:
        plt.xlabel("epochs")
    plt.legend(loc='lower right')
    
plt.show()


# 실선 : 배치 정규화를 사용한 경우<br>
# 점선 : 배치 정규화를 사용하지 않은 경우<br><br>
# 배치 정규화를 사용할 때 학습 진도가 훨씬 빠르다는 것을 보여주고 있으며 가중치 초깃값에 크게 의존도 안해도 된다.

# ## 6.4 바른 학습을 위해
# **오버피팅 (Overfitting)**: 신경망이 훈련데이터에만 지나치게 적응되어 그 외의 데이터에는 제대로 대응하지 못하는 상태

# ### 6.4.1 오버피팅 (Overfitting)
# 다음과 같은 상황에 오버피팅이 발생합니다
# - 매개변수가 많고 표현력이 높은 모델
# - 훈련 데이터가 적음
# 
# 여기에서는 두 조건을 충족해서 오버피팅을 일으키며, 그러기 위해서는 60,000개의 데이터 대신에 300개만 사용하였으며 7층 네트워크를 사용해 네트워크의 복잡성을 높이겠습니다.

# In[99]:


from dataset.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD
## 1.) 데이터를 읽는 코드 ##
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)
# 오버피팅을 재현하기 위해 학습 데이터 수를 줄임
x_train = x_train[:300]
t_train = t_train[:300]


# weight decay（가중치 감쇠） 설정
weight_decay_lambda = 0 # weight decay를 사용하지 않을 경우

## 2.) 훈련을 수행하는 코드 ##
network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10,
                        weight_decay_lambda=weight_decay_lambda)
optimizer = SGD(lr=0.01) # 학습률이 0.01인 SGD로 매개변수 갱신

max_epochs = 201
train_size = x_train.shape[0]
batch_size = 100

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)
epoch_cnt = 0

for i in range(1000000000):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grads = network.gradient(x_batch, t_batch)
    optimizer.update(network.params, grads)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print("epoch:" + str(epoch_cnt) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc))

        epoch_cnt += 1
        if epoch_cnt >= max_epochs:
            break


# In[100]:


# 그래프 그리기==========
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()


# 훈련 데이터에서는 거의 1.0 가까운 정확도를 보이고 있지만 실제 시험 데이터과는 매우 큰 차이를 보이므로 정확도가 떨어진 모습을 보이고 있습니다. 이와 같은 현상을 오버피팅이라고 보면 됩니다.

# ### 6.4.2 가중치 감소 (Weight Decay)
# **가중치 감소 (Weight Decay)**: 학습 과정에서 큰 가중치에 대해서는 그에 상응하는 큰 페널티를 부과하여 오버피팅을 억제하는 방법입니다.
# 
# 가중치를 $\textbf{W}$라 하면 
# <br>가중치의 제곱 노름(norm) --> (L2 노름)에 따른 가중치 감소는 $\frac{1}{2}\lambda\textbf{W}^2$
# <br>$\lambda$는 여기에서 정규화의 세기를 조절하는 하이퍼 파라미터이며 크게 설정할수록 큰 가중치에 대한 페널티가 커집니다.
# <br>$\frac{1}{2}$ 은 $\frac{1}{2}\lambda\textbf{W}^2$의 미분 결과인 $\lambda\textbf{W}$를 조정하는 상수
# <br>손실함수에 $\frac{1}{2}\lambda\textbf{W}^2$를 더하면 가중치 감소가 나오게 된다.
# <br><br>**즉** 가중치 감소는 모든 가중치 각각의 손실 함수에 $\frac{1}{2}\lambda\textbf{W}^2$를 더한 것이며,
# <br>가중치의 기울기를 구하는 계산에서는 그동안의 오차역전파법에 따른 결과에 정규화 항을 미분한 $\lambda\textbf{W}$를 더합니다.
# <br><br>L2 노름 : 각 원소들의 제곱들을 더한 것, 쉽게 생각해서 거리 공식

# In[101]:


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

x_train = x_train[:300]
t_train = t_train[:300]

weight_decay_lambda = 0.1

network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10,
                        weight_decay_lambda=weight_decay_lambda)
optimizer = SGD(lr=0.01)

max_epochs = 201
train_size = x_train.shape[0]
batch_size = 100

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)
epoch_cnt = 0

for i in range(1000000000):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grads = network.gradient(x_batch, t_batch)
    optimizer.update(network.params, grads)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print("epoch:" + str(epoch_cnt) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc))

        epoch_cnt += 1
        if epoch_cnt >= max_epochs:
            break


# In[102]:


markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()


# 물론 아직도 테스트 데이터와 차이를 보여주고 있지만 그래도 가중치 감소가 없었을 때에 비해서는 현저히 준 모습을 보이고 있습니다. 즉 오버피팅이 억제되었습니다.

# ### 6.4.3 드롭아웃 (Dropout)
# 신경망 모델이 복잡해지면 위의 가중치 감소만으로 대응하기 어렵기 때문에 드롭아웃을 사용하게 됩니다.
# <br><br>
# 드롭아웃은 뉴런을 임의로 삭제하면서 학습하는 방법이며 훈련 때 은닉층의 뉴런을 무작위로 골라 삭제합니다. 훈련 때는 데이터를 흘릴 때마다 삭제할 뉴런을 무작위로 선택하고 시험 때는 모든 뉴런을 활용합니다.
# ![Screenshot 2023-08-03 at 2.30.29 PM.png](attachment:dc414746-c156-4e6b-a792-0b93f6eb2e4a.png)
# 다음은 드롭아웃을 구현한 것이며 순전파에서 훈련 때 (train_flg = True) 인 곳에서만 계산하게 된다면 시험 때는 단순히 데이터를 흘리기만 하면 됩니다.

# In[103]:


class Dropout:
    def __init__(self, dropout_ratio = 0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None
    def forward(self, x, train_flg = True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)
    def backward(self, dout):
        return dout * self.mask


# **핵심**: self.mask 에 삭제할 뉴런을 False 로 표시한다는 것입니다.

# 다음은 드롭아웃을 사용하지 않을 때의 결과입니다.

# In[104]:


from dataset.mnist import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.trainer import Trainer

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 오버피팅을 재현하기 위해 학습 데이터 수를 줄임
x_train = x_train[:300]
t_train = t_train[:300]

use_dropout = False
dropout_ratio = 0.2

network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100],
                              output_size=10, use_dropout=use_dropout, dropout_ration=dropout_ratio)
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=301, mini_batch_size=100,
                  optimizer='sgd', optimizer_param={'lr': 0.01}, verbose=True)
trainer.train()

train_acc_list, test_acc_list = trainer.train_acc_list, trainer.test_acc_list

# 그래프 그리기==========
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()


# 다음은 드롭아웃을 사용할 때 (0.2로 설정)의 결과입니다.

# In[105]:


from dataset.mnist import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.trainer import Trainer

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 오버피팅을 재현하기 위해 학습 데이터 수를 줄임
x_train = x_train[:300]
t_train = t_train[:300]

use_dropout = True
dropout_ratio = 0.2

network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100],
                              output_size=10, use_dropout=use_dropout, dropout_ration=dropout_ratio)
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=301, mini_batch_size=100,
                  optimizer='sgd', optimizer_param={'lr': 0.01}, verbose=True)
trainer.train()

train_acc_list, test_acc_list = trainer.train_acc_list, trainer.test_acc_list

# 그래프 그리기==========
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()


# 드롭아웃을 사용한 것과 안한 것에는 확실한 차이가 있습니다. 이처럼 드롭아웃은 표현력을 높이면서도 오버피팅을 억제할 수 있습니다.

# ## 6.5 적절한 하이퍼 파라미터 값 찾기
# 하이퍼 파라미터의 예시들: 
# 
# - 각 층의 뉴런 수
# - 배치 크기
# - 매개변수 갱신 시의 학습률
# - 가중치 감소
# - 등등

# ### 6.5.1 검증 데이터
# ##### 하이퍼 파라미터의 성능을 평가할 때는 시험 데이터를 사용하면 안됩니다.
# 시험 데이터를 사용하여 하이퍼파라미터를 조정하면 값이 시험 데이터에 오버피팅 됩니다. 즉 하이퍼파라미터 값이 시험 데이터에만 적합하도록 조정되어 버린다는 것입니다. 이렇게 되면 다른 데이터에는 적응하지 못하니 범용 성능이 떨어지는 모델이 될 수도 있습니다.<br><br>
# 그렇기 때문에 검증 데이터, 즉 하이퍼 파라미터 전용 확인 데이터가 필요합니다.
# ##### 정리 :
# - 훈련 데이터 : 매개변수 학습
# - 검증 데이터 : 하이퍼 파라미터 성능 평가
# - 시험 데이터 : 신경망의 범용 성능 평가
# 
# MNIST 데이터와 같이 훈련 데이터와 시험 데이터만 있는 경우에는 직접 데이터를 분리해야 하며 교재에서는 훈련 데이터 중 20% 정도를 검증 데이터로 먼저 분리합니다.

# In[106]:


(x_train, t_train), (x_test, t_test) = load_mnist()
def shuffle_dataset(x, t):
    """데이터셋을 뒤섞는다.

    Parameters
    ----------
    x : 훈련 데이터
    t : 정답 레이블
    
    Returns
    -------
    x, t : 뒤섞은 훈련 데이터와 정답 레이블
    """
    permutation = np.random.permutation(x.shape[0])
    x = x[permutation,:] if x.ndim == 2 else x[permutation,:,:,:]
    t = t[permutation]

    return x, t
x_train, t_train = shuffle_dataset(x_train, t_train)
validation_rate = 0.20
validation_num = int(x_train.shape[0] * validation_rate)


# ### 6.5.2 하이퍼파라미터 최적화
# '최적 값'이 존재하는 범위를 조금씩 줄여간다는 것이 핵심입니다. 줄이기 위해서는 대략적인 범위를 지정하며, 보통은 $10^{-3} \sim 10^{3}$ 와 같이 10의 거듭제곱 단위 (로그 스케일) 로 지정합니다.<br>Note: 굉장히 오래 걸린다네요 거의 뭐 며칠, 몇 주 이상..., 그래서 세련된 기법인 **베이즈 최적화** 방법도 있다네요.
# <br>다음은 하이퍼 파라미터 최적화에 대한 요약입니다.
# ![Screenshot 2023-08-03 at 2.49.49 PM.png](attachment:bb7b9393-803b-4bfd-8dbb-0ccc45b5a29b.png)

# ### 6.5.3 하이퍼 파라미터 최적화 구현하기
# Weight_Decay : ($10^{-8}, 10^{-4}$)
# <br>Learning Rate 범위 : ($10^{-6}, 10^{-2}$)

# In[107]:


from dataset.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.util import shuffle_dataset
from common.trainer import Trainer

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 결과를 빠르게 얻기 위해 훈련 데이터를 줄임
x_train = x_train[:500]
t_train = t_train[:500]

# 20%를 검증 데이터로 분할
validation_rate = 0.20
validation_num = int(x_train.shape[0] * validation_rate)
x_train, t_train = shuffle_dataset(x_train, t_train)
x_val = x_train[:validation_num]
t_val = t_train[:validation_num]
x_train = x_train[validation_num:]
t_train = t_train[validation_num:]

def __train(lr, weight_decay, epocs=50):
    network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100],
                            output_size=10, weight_decay_lambda=weight_decay)
    trainer = Trainer(network, x_train, t_train, x_val, t_val,
                      epochs=epocs, mini_batch_size=100,
                      optimizer='sgd', optimizer_param={'lr': lr}, verbose=False)
    trainer.train()
    return trainer.test_acc_list, trainer.train_acc_list

# 하이퍼파라미터 무작위 탐색======================================
optimization_trial = 100
results_val = {}
results_train = {}
for _ in range(optimization_trial):
    # 탐색한 하이퍼파라미터의 범위 지정===============
    weight_decay = 10 ** np.random.uniform(-8, -4)
    lr = 10 ** np.random.uniform(-6, -2)
    # ================================================
    val_acc_list, train_acc_list = __train(lr, weight_decay)
    print("val acc:" + str(val_acc_list[-1]) + " | lr:" + str(lr) + ", weight decay:" + str(weight_decay))
    key = "lr:" + str(lr) + ", weight decay:" + str(weight_decay)
    results_val[key] = val_acc_list
    results_train[key] = train_acc_list


# In[108]:


graph_draw_num = 20
col_num = 5
row_num = int(np.ceil(graph_draw_num / col_num))
i = 0

for key, val_acc_list in sorted(results_val.items(), key=lambda x:x[1][-1], reverse=True):
    print("Best-" + str(i+1) + "(val acc:" + str(val_acc_list[-1]) + ") | " + key)

    plt.subplot(row_num, col_num, i+1)
    plt.title("Best-" + str(i+1))
    plt.ylim(0.0, 1.0)
    if i % 5: plt.yticks([])
    plt.xticks([])
    x = np.arange(len(val_acc_list))
    plt.plot(x, val_acc_list)
    plt.plot(x, results_train[key], "--")
    i += 1

    if i >= graph_draw_num:
        break

plt.show()


# 잘될 것 같은 범위들을 좁혀 나가는 과정임을 알 수가 있습니다.

# ### 6.6 정리
# ![Screenshot 2023-08-03 at 2.53.11 PM.png](attachment:5f23d87a-bd10-4b80-8d76-a03d12711410.png)
