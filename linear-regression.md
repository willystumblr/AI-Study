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

학습이 거듭됨에 따라 오차가 계속 줄어드는 것을 확인할 수 있음. 

## Boston Housing Dataset

{% embed url="https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html" %}

\[설명\] This dataset contains information collected by the U.S Census Service concerning housing in the area of Boston Mass. 

### **Peek dataset & Preprocessing**

```python
from sklearn.datasets import load_boston
import pandas as pd

bos = load_boston()
bos.keys()
```

dict-formatted data; line 5 shows:

`dict_keys(['data', 'target', 'feature_names', 'DESCR'])`

1. data: the content of features, which are what we focus on.
2. target: the price of houses, which are **what we need to predict**.
3. feature\_names: as its name, feature names. storing the meanings of each column respectively.
4. DESCR: the description of this dataset.
5. filename: the path of this dataset storing.

굳이 사이즈를 확인하고 싶으면

```python
bos.data.shape
```

=&gt; \(506, 13\)

```python
df = pd.DataFrame(bos.data)
df.columns = bos.feature_names
df['Price'] = bos.target
df.head()
```

1: `pd.DataFrame`: _Two-dimensional_, size-mutable, potentially heterogeneous tabular data  
2: `feature_names`, i.e. column names goes to `df`'s column  
3: setting `bos.target` as 'Price'  
4: peek the first 5 rows of data by `.head()` after adding a ‘Price’ column to our data.

### **Split training data and testing data**

```python
data = df[df.columns[:-1]]
data = data.apply(
    lambda x: (x - x.mean()) / x.std()
)

data['Price'] = df.Price
```

{% hint style="info" %}
**`lambda` expression**  
In Python, an anonymous function is a [function](https://www.programiz.com/python-programming/function) that is defined without a name.

While normal functions are defined using the `def` keyword in Python, anonymous functions are defined using the `lambda` keyword.

`lambda arguments: expression`

Lambda functions can have any number of arguments but **only one expression**. The expression is evaluated and returned. Lambda functions can be used wherever function objects are required.
{% endhint %}

=&gt; to get the normalized value of each feature 

```python
import seaborn as sns 
import matplotlib.pyplot as plt 

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(df['Price'], bins=30)
plt.show()
```

![](.gitbook/assets/image%20%285%29.png)

### Splitting Data into Test Sets, Training Sets

```python
import numpy as np

X = data.drop('Price', axis=1).to_numpy()
Y = data['Price'].to_numpy()

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
print(X_train.shape)    #(354, 13)
print(X_test.shape)     #(152, 13)
print(Y_train.shape)    #(354,)
print(Y_test.shape)     #(152,)
```

Format data as an array in numpy, divide our data as a training set and a testing set  
=&gt; "drop은 row or column label을 입력하면 그 값들을 지워줍니다. axis = 1은 각 row값을 가져옵니다." --&gt; 'Price' column을 제거한 값을 X에 넣어주겠다는 의, Y는 반면에 'Price'를 부여하겠다는 의미

```python
correlation_matrix = df.corr().round(2)
# annot = True to print the values inside the square
sns.heatmap(data=correlation_matrix, annot=True)
```

* `corr()`: 데이터 프레임의 열 사이의 상관 관계

![](.gitbook/assets/image%20%286%29.png)

## Constructing Linear Regression on PyTorch

```python
import torch

n_train = X_train.shape[0] #354
X_train = torch.tensor(X_train, dtype=torch.float)
X_test = torch.tensor(X_test, dtype=torch.float)
Y_train = torch.tensor(Y_train, dtype=torch.float).view(-1, 1)
Y_test = torch.tensor(Y_test, dtype=torch.float).view(-1, 1)
```

Setting `n_train` as 354, and saving each of the four as tensor

PyTorch allows a tensor to be a `View` of an existing tensor. View tensor shares _the same underlying data with its base tensor._ Supporting `View` avoids explicit data copy, thus allows us to do fast and memory efficient reshaping, slicing and element-wise operations.

```python
w_num = X_train.shape[1]
net = torch.nn.Sequential(
    torch.nn.Linear(w_num, 1)
)

torch.nn.init.normal_(net[0].weight, mean=0, std=0.1)
torch.nn.init.constant_(net[0].bias, val=0)
```

* `torch.nn.Sequential`: 간단히 말하자면 여러 nn.Module을 한 컨테이너에 집어넣고 한 번에 돌리는 방법. `nn.Sequential`은 코드에 적힌 순서대로 값을 전달해 처리한다. 빠르고 간단히 적을 수 있기 때문에 간단한 모델을 구현할 때에 쓰면 된다는 것 같다.
* `torch.nn.Linear`: Applies a linear transformation to the incoming data:  **y = xA^T + b**
* Fills the input Tensor with values drawn from the normal distribution :

$$\mathcal{N}(\text{mean}, \text{std}^2)$$  

```python
datasets = torch.utils.data.TensorDataset(X_train, Y_train)
train_iter = torch.utils.data.DataLoader(datasets, batch_size=10, shuffle=True)

loss = torch.nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
```

* constructing a Dataset of Tensor.
* generate a DataLoder by using this Dataset. `batch_size` is the size of each batch in which data returned. Data will be returned in _random sequence_ if shuffle is True.
* set loss function with mean squared error
* optimize the neural network by stochastic\(확률적, 추츨적\) gradient descent.

```python
num_epochs = 300
for epoch in range(num_epochs):
    for x, y in train_iter:
        output = net(x)
        l = loss(output, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print("epoch {} loss: {:.4f}".format(epoch + 1, l.item()))
```

**Process**

1. Load a batch of data.
2. Predict the batch of the data through _net_.
3. Calculate the loss value by predict value and true value.
4. Clear the grad value optimizer stored.
5. **Backpropagate** the loss value. ==&gt; NN에 나오는 거 같은
6. Update optimizer

