# Homework3

**Name:** Youming(Remi) Zhang

**NetID:** yz7399

## Question1

**(3** **points)** *CNNs* *vs* *RNNs*. Until now we have seen examples of how to perform image classification using both feedback convolutional (CNN) architectures as well as recurrent (RNN) architectures.

a. Give two benefits of CNN models over RNN models for image classification.

b. Now, give two benefits of RNN models over CNN models.

**Solution(a):**

(1): RNN has Long-range dependencies which mean the model may not have good information about previous words. While CNN only depends on the current input.

(2): CNN is more stable than RNN because RNN is more vulnerable to gradient vanishing.

**Solution(b):**

(1): CNN must have a fixed input size, RNN can take arbitrary input length.

(2): RNN uses time-series information to identify patterns between input and output, so it performs better in a scenario like  audio recognition and NLP



## Question2

**(4 points)** *Recurrences* *using RNNs.* Consider the recurrent network architecture below in Figure 1. All inputs are integers, hidden states are scalars, all biases are zero, and all weights are indicated by the numbers on the edges. The output unit performs binary classification. Assume that the input sequence is of **even** length. What is computed by the output unit at the final time step? Be precise in your answer. It may help to write out the recurrence clearly.

**Solution:**

Given that:

$h_t = x_t - h_{t-1}$

$y_t = sigmoid(1000h_t)$

$x_t = [x_1,x_2,...,x_{2n}]$

When t = 1

$$h_1 = x_1$$

$$y_1 = sigmoid(1000x_1)$$

When t = 2

$h_2 = x_2 - h_1 = x_2 - x_1$

$y_2 = sigmoid(1000h_2) = sigmoid[1000 (x_2 - x_1)]$

When t = 3

$h_3 = x_3 - h_2 = x_3 - x_2 + x_1$

$y_3 = sigmoid(1000h_3) = sigmoid[1000(x_3 - x_2 + x_1)]$

When t = 2n

$h_{2n} = x_{2n} - h_{2n-1} = \sum_{i=1}^nx_{2i} - \sum_{i=1}^nx_{2i-1}$

$y_{2n} = sigmoid(1000.h_{2n})$



## Question 3

**(3** **points)** *Attention!* *My* *code* *takes* *too* *long.* In class, we showed that a computing a regular self-attention layer takes *O*($T^2$) running time for an input with *T* tokens. Propose two different ways to reduce this running time to *O*(*T* ), and comment on their possible pros vs cons.



