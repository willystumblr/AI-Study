---
description: >-
  This page covers the concept of linear regression and its application on
  Boston housing dataset.
---

# Linear Regression on Boston Housing Dataset

## Linear Regression

Linear regression attempts to _model the relationship between two variables by fitting a linear equation to observed data._ One variable is considered to be an explanatory variable, and the other is considered to be a dependent variable. For example, a modeler might want to relate the weights of individuals to their heights using a linear regression model.

Before attempting to fit a linear model to observed data, **a modeler should first determine whether or not there is a relationship between the variables of interest**. This does not necessarily imply that one variable causes the other \(for example, higher SAT scores do not cause higher college grades\), but that there is some significant association between the two variables. A [scatterplot](http://www.stat.yale.edu/Courses/1997-98/101/scatter.htm) can be a helpful tool in determining the strength of the relationship between two variables. If there appears to be no association between the proposed explanatory and dependent variables \(i.e., the scatterplot does not indicate any increasing or decreasing trends\), then fitting a linear regression model to the data probably will not provide a useful model. A valuable numerical measure of association between two variables is the [correlation coefficient](http://www.stat.yale.edu/Courses/1997-98/101/correl.htm), which is a value between -1 and 1 indicating the strength of the association of the observed data for the two variables.

A linear regression line has an equation of the form **Y = a + bX**, where **X** is the explanatory variable and **Y** is the dependent variable. The slope of the line is **b**, and **a** is the intercept \(the value of **y** when **x** = 0\). \(출처: [http://www.stat.yale.edu/Courses/1997-98/101/linreg.htm](http://www.stat.yale.edu/Courses/1997-98/101/linreg.htm)\)

즉, 주어진 데이터를 바탕으로 두 변수 간의 관계를 1차함수 꼴로 나타내어 예측 모델을 만드는 것.   
회귀\(Regression\)이므로 모델을 통해 '값'을 예측함.

### Cost Function

MSE\(Mean Squared Error\)

$$
MSE =  \frac{1}{N} \sum_{i=1}^{n} (y_i - (m x_i + b))^2
$$

To minimize MSE, Gradient Descent is used.

#### Gradient Descent

Consider the 3-dimensional graph below in the context of a cost function. Our goal is to move from the mountain in the top right corner \(high cost\) to the dark blue sea in the bottom left \(low cost\). The arrows represent the direction of steepest descent \(negative gradient\) from any given point–the direction that decreases the cost function as quickly as possible.

![Andrew Ng](.gitbook/assets/image%20%282%29.png)

MSE의 두 변수인 m, b를 기준으로 나타낸 그래프  
**GOAL:** 가장 높이가 낮은\(dark blue\) 곳으로 가는 것  =&gt; Gradient Descent is used!

$$
\begin{split}f'(m,b) =
   \begin{bmatrix}
     \frac{df}{dm}\\
     \frac{df}{db}\\
    \end{bmatrix}
=
   \begin{bmatrix}
     \frac{1}{N} \sum -2x_i(y_i - (mx_i + b)) \\
     \frac{1}{N} \sum -2(y_i - (mx_i + b)) \\
    \end{bmatrix}\end{split}
$$

#### Learning Rate

**The size of these steps** is called the _learning rate_. With a high learning rate we can cover more ground each step, but we risk overshooting the lowest point since the slope of the hill is constantly changing. With a very low learning rate, we can confidently move in the direction of the negative gradient since we are recalculating it so frequently. A low learning rate is more precise, but calculating the gradient is time-consuming, so it will take us a very long time to get to the bottom.

### Gradient Descent in PyTorch

#### Tensor

multidimensional array

![vector, matrix, and tensor](.gitbook/assets/image%20%284%29.png)

> 텐서는 일관된 유형\(`dtype`이라고 불림\)을 가진 다차원 배열입니다. 지원되는 모든 `dtypes`은 [`tf.dtypes.DType`](https://www.tensorflow.org/api_docs/python/tf/dtypes/DType?hl=ko)에서 볼 수 있습니다.
>
> [NumPy](https://numpy.org/devdocs/user/quickstart.html)에 익숙하다면, 텐서는 일종의 `np.arrays`와 같습니다.
>
> 모든 텐서는 Python 숫자 및 문자열과 같이 변경할 수 없습니다. 텐서의 내용을 업데이트할 수 없으며 새로운 텐서를 만들 수만 있습니다.

#### Creation of tensor and initialization

```python
import torch
X = torch.Tensor(2,3) #난수가 들어감. 2X3 tensor
X = torch.tensor([1,2,3],[4,5,6])
```

{% hint style="success" %}
**Q. What is the difference between `torch.Tensor` and `torch.tensor`?**  
Our `torch.Tensor` constructor is overloaded to do the same thing as both `torch.tensor` and `torch.empty`. We thought this overload would make code confusing, so _we split_ `torch.Tensor` into `torch.tensor` and `torch.empty`.
{% endhint %}

```python
import torch
# z= 2*x^2 + 3에서 x 기울기 구하기

x = torch.tensor(data=[2.0,3.0], requires_grad = True)
#텐서를 생성, 기울기를 사용하도록 지정함
y = x**2 
z = 2*x + 3

target = torch.tensor([3.0, 4.0])

loss = torch.sum(torch.abs(z-target))
loss.backward()

print(x.grad) #y.grad z.grad ==> warning!
print(y.retain_grad(), z.retain_grad())
# If you want to obtain non-leaf node's gradient, use .retain_grad()
```

```text
tensor([2., 2.])
None None
```

**Functions**

* `torch.tensor`

  argument: `data`, `dtype`, `device`, `requires_grad`...  
  \*`requires_grad`: 텐서에 대한 기울기를 저장할 것인지 여부를 지정하는 argument   
  =&gt; `True/False`

* `torch.abs(z-target)`: computing absolute value of `z-target`
* `torch.sum`: Returns the sum of all elements in the `input` tensor.
* `loss.backward()`: 연산 그래프 따라가면서 leaf node의 gradient를 계산한다고 함. 
  * leaf node: "`x`"

### Linear Regression Model

```python
import torch
import torch.nn as nn #neural net. model / Linear function 사용
import torch.optim as optim #grad descent 
import torch.nn.init as init #values required for initialization of tensor
```

```python
num_data = 100 #number of data used
num_epoch = 500 #number of repitition of gradient descent

x = init.uniform_(torch.Tensor(num_data, 1), -10, 10)
noise = init.normal_(torch.FloatTensor(num_data, 1), std=1)
y = 2*x+3
y_noise = y + noise
```

* line 4: 100X1 tensor 생성, 이를 `init.uniform_()` 함수 사용해 -10~10까지 균등하게 초기화 \(randomly\)
* line 5: y =&gt; -17≤ $$2x + 3$$ ≤ 23
* line 6, 7: noise - _Gaussian noise_ **Why?** 데이터에 기본적으로 노이즈가 추가되어 있음. =&gt; reality

```python
model = nn.Linear(1,1)
cost_func = nn.L1Loss()
```

> Applies a linear transformation to the incoming data:   
> $$y = xA^T + b$$

* `Linear`: \# of features\(input\), \# of features\(output\), bias 사용 여\(?\)  
  parameter: weight, bias

  --&gt; 1개의 input feature x, 1개의 output features

* `L1Loss()`: `y_noise`와 `model`의 차이

```python
optimizer = optim.SGD(model.parameters(), lr = 0.01)
```

최적화 함수; 

* `model.parameters()`: passing w and b from model
* `lr`: learning rate 

```python
label = y_noise
for i in range(num_epoch) :
    optimizer.zero_grad()
    output = model(x)
    
    loss = cost_func(output, label)
    loss.backward()
    optimizer.step()
    
    if i%10 == 0:
        print(loss.data)

param_list = list(model.parameters())
print(param_list[0].item(), param_list[1].item())
```

* `zero_grad()`: gradient 초기화, 새로운 gradient를 구해야하기 때문

학습이 거듭됨에 따라 오차가 계속 줄어드는 것을 확인할 수 있



