# Linear Regresion with multiple features

## Inital data

|                   	| Size  	| # Bedrooms 	| # Floor 	| Age   	| Price (1000s) 	|
|-------------------	|-------	|------------	|---------	|-------	|---------------	|
| ${x_j}/{x^{(i)}}$ 	| $x_1$ 	| $x_2$      	| $x_3$   	| $x_4$ 	| $y$           	|
| $x^{(1)}$         	| 2104  	| 5          	| 1       	| 45    	| 460           	|
| $x^{(2)}$         	| 1416  	| 3          	| 2       	| 40    	| 232           	|
| ...               	| ...   	| ...        	| ...     	| ...   	| ...           	|

- $x_j$ = $j^{th}$ feature
- $n$ = number of features, in the table, $n = 4$
- $\vec x^{(i)}$ = feature of $i^{th}$ training example
- $x^{(i)}_j$ = value of feature $j$ in $i^{th}$ training example

For example:  
$\vec x^{(2)} = [1416, 3, 2, 40]$  
$x^{(2)}_3 = 2$

## Model with multiple features

Previously,  
$f_{w,b}(x) = wx + b$

Now,  
$f_{w,b}(x) = w_1x_1 + w_2x_2 + ... + w_nx_n + b$

With vectorization notation:  
$f_{\vec w,b}(\vec x) = \vec w \cdot \vec x + b$  

Where  
$\vec w = [w_1, w_2, ..., w_n]$ are parameters of the model  
$\vec x = [x_1, x_2, ..., x_n]$ are the $j^{th}$ feature  
$b, w_j, x_j \in \mathbb{R}$ and $\vec w \cdot \vec x^{(i)}$ is dot product

## Gradient descent

The cost function is rewritten in vectorized notation as  
$J(\vec w, b) = \frac{1}{2m} \displaystyle\sum_{i=1}^{m} (f_{\vec w,b}(\vec x^{(i)}) - y^{(i)})^2 $, where $m$ is the number of training examples.

So the gradient descent is rewritten as:  
$w_{j, next} = w_j - \alpha \frac{\partial}{\partial w_j} J(\vec w, b)$  
$b_{next} = b - \alpha \frac{\partial}{\partial b} J(\vec w, b)$

Therefore, for $j$ from $1$ to $n$, we repeat until convergence  
$w_{1, next} = w_1 - \alpha\frac{1}{m} \displaystyle\sum_{i=1}^{m} (f_{\vec w,b}(\vec x^{(i)}) - y^{(i)})x^{(i)}_1 $  

...

$w_{j, next} = w_j - \alpha \frac{1}{m} \displaystyle\sum_{i=1}^{m} (f_{\vec w,b}(\vec x^{(i)}) - y^{(i)})x^{(i)}_n $  

$b_{next} = b - \alpha \frac{1}{m} \displaystyle\sum_{i=1}^{m} (f_{\vec w,b}(\vec x^{(i)}) - y^{(i)}) $

## Feature scaling

Given a training set where:  
$300 \leqslant x_1 \leqslant 2000$  
$0 \leqslant x_2 \leqslant 5$  

When the features have a very different range of values, it can cause the gradient descent to run slow. One option is to feature scale them so they all are in a closer range.

### Option 1 - Divide by $max$

For $x_1$, divide by $2000$, so the range would be  
$0.15 \leqslant x_{1, scaled} \leqslant 1$  

For $x_2$, divide by $5$, so the range would be  
$0 \leqslant x_{2, scaled} \leqslant 1$

### Option 2 - Mean normalization

Calculate the average $\mu_j$  
For $x_1$, $\mu_1 = 600$ and for $x_2$, $\mu_2 = 2.3$.

$x_{1, scaled} = \frac{x_1 - \mu_1}{200 - 300}$  
$x_{2, scaled} = \frac{x_2 - \mu_2}{5 - 0}$  
$-0.18 \leqslant x_{1, scaled} \leqslant 0.82$  
$-0.46 \leqslant x_{2, scaled} \leqslant 0.54$

### Option 3 - Z-score normalization

Calculate standard deviation $\sigma_j$  
For $x_1$, $\sigma_i = 450$ and for $x_2$, $\sigma_i = 1.4$.

Calculate the average $\mu_j$  
For $x_1$, $\mu_1 = 600$ and for $x_2$, $\mu_2 = 2.3$.

$x_{1, scaled} = \frac{x_1 - \mu_1}{\sigma_1}$  
$x_{2, scaled} = \frac{x_2 - \mu_2}{\sigma_2}$  
$-0.67 \leqslant x_{1, scaled} \leqslant 3.1$  
$-1.6 \leqslant x_{2, scaled} \leqslant 1.9$

### Examples

$-1 \leqslant x_{j} \leqslant 1$ OK, no rescaling  
$-2 \leqslant x_{j} \leqslant 2$ OK, no rescaling  
$-0.3 \leqslant x_{j} \leqslant 0.3$ OK, no rescaling  

$0 \leqslant x_{1} \leqslant 3$ Ok, no rescaling  
$-2 \leqslant x_{2} \leqslant 2$ OK, no rescaling  
$-100 \leqslant x_{3} \leqslant 100$ Too large, rescale  
$-0.001 \leqslant x_{4} \leqslant 0.001$ Too small, rescale  
$98.6 \leqslant x_{5} \leqslant 105$ Too small, rescale

