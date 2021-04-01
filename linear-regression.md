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





