# Homework1

**Name:** Youming Zhang

**NetID:** yz7399

## Question 1

(2 points) *Linear regression with non-standard losses*. In class we derived an analytical expression for the optimal linear regression model using the least squares loss. **If** ***X*** is the matrix of *n* training data points (stacked row-wise) and y is the vector of their corresponding labels, then:

**a.** Using matrix/vector notation, write down a loss function that measures the training error in
terms of the $l_1$-norm. Write down the sizes of all matrices/vectors.

**Solution:**

We can use Mean Absolute Error(MAE) to represent loss function using *$l_1$-norm*

given 

$y = (y_1,\cdots,y_n)^T$ is an $n\times 1$ vector

$X = (x_1^T,\cdots,x_n^T)$ is an $n\times d$ matrix

$w = (w_1,\cdots,w_d)$ is an $d \times 1$ vector

$L(w) = \vert y - Xw \vert$  

**b.** Can you simply write down the optimal linear model in closed form, as we did for standard
linear regression? If not, why not?

**solution:**

No, we may not get the optimal linear model in closed form,

because $\frac{d(L(w))}{dw} = -X^T$, on one hand, this formula has nothing to do with $w$, on the other hand we cannot set $-X^T = 0$.



## Question2

**(3 points)** _Expressivity of neural networks_. Recall that the functional form for a single neuron is
given by _y_ = _σ_ (〈 _w,x_ 〉+ _b,_ 0), where _x_ is the input and _y_ is the output. In this exercise, assume
that _x_ and _y_ are 1-dimensional (i.e., they are both just real-valued scalars) and _σ_ is the unit step
activation. We will use multiple layers of such neurons to approximate pretty much any function
_f_. There is no learning/training required for this problem; you should be able to guess/derive the
weights and biases of the networks by hand.

a.A _box_ function with height _h_ and width _δ_ is the function _f_ ( _x_ ) = _h_ for 0 _< x < δ_ and 0
      otherwise. Show that a simple neural network with 2 hidden neurons with step activations
      can realize this function. Draw this network and identify all the weights and biases. (Assume
      that the output neuron only sums up inputs and does not have a nonlinearity.)

**Solution:**



b.Now suppose that _f_ is _any arbitrary, smooth, bounded_ function defined over an interval
      [− _B,B_ ]. (You can ignore what happens to the function outside this interval, or just assume
      it is zero). Use part a to show that this function can be closely approximated by a neural
      network with a hidden layer of neurons. You don’t need a rigorous mathematical proof here;
      a handwavy argument or even a figure is okay here, as long as you convey the right intuition. 

c. Do you think the argument in part b can be extended to the case of _d_ -dimensional inputs?
(i.e., where the input _x_ is a vector – think of it as an image, or text query, etc). If yes,
comment on potential practical issues involved in defining such networks. If not, explain
why not.

 

## Question3

(3 points) Calculating gradients. Suppose that _z_ is a vector with _n_ elements. We would like to compute the gradient of _y_ =softmax( _z_ ). Show that the Jacobian of _y_ with respect to _z_ , _J_ , is given by the 

​													     $$J_{ij} = \frac{\partial y_i}{\partial z_j} =y_i(\delta_{ij} - y_j)$$

where $\delta_{ij}$ is the Dirac delta, i.e.., 1 if $i = j$ and 0 else. *Hint: Your algebra could be simplified if you try computing the log derivative,* $\frac{\partial \log y_i}{\partial z_j}$

$y_i = \frac{e^{z_i}}{\sum_{j=1}^{n}e^{z_j}}$

$Loss = -\sum_{j=1}^{n}t_ilogy_i$ here, $t_i$ is ground truth value, $y_i$ is value producted by softmax.

$log^{y_i}=log^{e^{z_i}}-log^{\sum_{j=1}^{n}e^{z_j}}=z_i - log^{\sum_{j=1}^{n}e^{z_i}}$

$loss_i = -log^{y_i}=log^{\sum_{j=1}^{n}e^{z_j}} - z_i$

we want to minimize $loss_i$, then we need to calculate the gradient

$\frac{\partial{loss_i}}{\partial{{z_i}}} = \frac{e^{z_i}}{\sum_{j=1}^{n}e^{z_j}}-1 = y_i - 1$

 















