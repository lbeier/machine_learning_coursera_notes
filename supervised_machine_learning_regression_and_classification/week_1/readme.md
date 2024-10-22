# Linear regression

## Terminology

Given the following training set:

|      | Size in m<sup>2</sup> (x) | Price in 1000's (y) |
|------|-----------------|---------------------|
| (1)  | 2104            | 400                 |
| (2)  | 1416            | 232                 |
| ...  | ...             | ...                 |
| (47) | 3210            | 870                 |

$x$ = "input" variable or feature  
$y$ = "output" variable or target  
$m$ = number of training examples

$(x^{(i)}, y^{(i)})$ = i<sup>th</sup> training example  

For example,  
$(x^{(2)}, y^{(2)}) \implies (1416, 232)$

> A training set in supervised learning includes both the input features $(x)$, such as the size of the house and also the output targets $(y)$, such as the price of the house.  
> The output targets are the right answers to the model we'll learn from.  
> To train the model, you feed the training set, both the input features and the output targets to your learning algorithm.  

## One variable model
>
> Supervised learning algorithm will produce some function $f$.  
> The job with $f$ is to take a new input $x$ and output and estimate or a prediction $\hat{y}$.

$x \rightarrow f \rightarrow \hat{y}$

$\hat{y}^{(i)} = f_{w,b}(x^{(i)}) = wx^{(i)} + b$

We need to find the parameters $w$ and $b$ so:  
$\hat{y}^{(i)}$ is close to $y^{(i)}$ for all $(x^{(i)}, y^{(i)})$

### Cost function

$J(w, b) = \frac{1}{2m} \displaystyle\sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2 $  
$\implies \frac{1}{2m} \displaystyle\sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^2 $

Objective  
$\textbf{min}_{w,b} J(w, b)$

### Gradient descent

$w_{next} = w - \alpha \frac{\partial}{\partial w} J(w, b)$  
$b_{next} = b - \alpha \frac{\partial}{\partial b} J(w, b)$

Where  
$\frac{\partial}{\partial w} J(w, b) = \frac{1}{m} \displaystyle\sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})x^{(i)} $  
$\frac{\partial}{\partial b} J(w, b) = \frac{1}{m} \displaystyle\sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)}) $

#### Simultaneous update

First, calculate both  
$w_{next} = w - \alpha \frac{\partial}{\partial w} J(w, b)$  
$b_{next} = b - \alpha \frac{\partial}{\partial b} J(w, b)$

And only then  
$w = w_{next}$  
$b = b_{next}$

#### Gradient descent algorithm

Repeat until convergence  
$w = w - \alpha \frac{1}{m} \displaystyle\sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})x^{(i)} $  
$b = b - \alpha \frac{1}{m} \displaystyle\sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)}) $  

## Coding Linear Regression

Given these

| Size (1000 sqm) | Price (1000's)           |
| ----------------| ------------------------ |
| 1               | 300                      |
| 2               | 500                      |

```python
x_train = np.array([1.0, 2.0])       #features
y_train = np.array([300.0, 500.0])   #target value
```

---
__Cost function__  
$J(w, b) =  \frac{1}{2m} \displaystyle\sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^2 $

```python
def compute_cost(x, y, w, b):
    m = x.shape[0] # Number of training examples
    sum_cost = 0

    for i in range(m):
        f_wb = w * x[i] + b
        sum_cost = sum_cost + (f_wb - y[i]) ** 2

    return 1 / (2 * m) * sum_cost
```

---
__Compute the gradient part of gradient descent, ie:__  
$\frac{1}{m} \displaystyle\sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})x^{(i)}$  
$\frac{1}{m} \displaystyle\sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})$  

```python
def compute_gradient(x, y, w, b): 
    m = x.shape[0] # Number of training examples
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b

        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]

        dj_db += dj_db_i
        dj_dw += dj_dw_i

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db
```

---

__Gradient descent__  
$w_{next} = w - \alpha \frac{\partial}{\partial w} J(w, b)$  
$b_{next} = b - \alpha \frac{\partial}{\partial b} J(w, b)$

```python
def gradient_descent(x, y, w_initial, b_initial, alpha, num_iters): 
    b = b_initial
    w = w_initial
    
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w , b) 

        b_next = b - alpha * dj_db
        w_next = w - alpha * dj_dw

        b = b_next
        w = w_next

        print(cost_function(x, y, w , b)) # Not required, just to track the cost on each iteration

    return w, b
```

So we can run:

```python
w_init = 0
b_init = 0
iterations = 10000
alpha = 1.0e-2 # learning rate
w_final, b_final = gradient_descent(x_train, y_train, w_init, b_init, alpha, iterations)
```
