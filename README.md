# PARAMETERIZED-NEURAL-NETWORKS
Introduction into Parametrized Neural Networks (***PNNs***) for simple physics example in 1 and 2 dimensions.

## Underlying Physics Problem
When measuring a new particle one normally encounters the problem of testing hypothesis against each other.
The $H_0$ hypothesis is usually the background only model. The alternative hypothesis is $H_1$. 
The problem is now a classification problem with probabilities. <br>
The questions arise: With which probability can be reject the null hypothesis and accept the alternative hypothesis?<br>
For analytic equations for the signal and background one can solve this problem relatively elegent by using the pdfs (probability density functions) and applying the NP-lemma that states the likelihood ratio is the best test statistic. <br>
From there on, one can evaulate further on the test statistic and make a statement.<br>

This cannot be applied for non-analytic distribution which are often the case in particle physics environments. Nevertheless, the setup is predefined for using a classification model to determine if an event is accepted in the signal model or not. <br>

Now, another limiting factor would be if we had to train for every siganl model a new NN. This can be solved in a way such that the neural network is paramterized with the $H_0$ and $H_1$. 

## Parameterized Neural Networks 

Networks are parameterized with the hypotheses $H_0$ and $H_1$.<br>
For the 1 dimensional example the network input has three input nodes:

| 1 | 2 | 3 |
| --- |--- |--- |
| _measured_ x position | x position of $H_0$ | x position of $H_1$ |

For the 2 dimensional example the network input has six input nodes:

| 1 | 2 | 3 | 4 | 5 | 6 |
|--- |--- |--- |--- |--- |--- |
| _measured_ x position | _measured_ y position | x position of $H_0$ | y position of $H_0$ | x position of $H_1$ | y position of $H_1$ |

## Classification Task

The task for the network is basically classifiying an event into two categories ($H_0$ or $H_1$). <br>
For this task there is a sigmoid function at the end of the the NN. 
When cutting the sigmoid function away after training one obtains the log likelihod ratio between the two models.<br>
This is better known as the _likelihood ratio trick_.
The output approaches the log likelihood ratio wich is according to Neyman and Pearson the best test statistic.<br>
The NN is therefore proven to do the best possible job to a non-analytic solution.

## Results

### 1D
The GIF shows two gaussian curves with different $\mu$
 and the same $\sigma$. The ***orange*** curve displays the ***constant background***. The moving blue* curve is the ***physics signal***. The ***line in dark blue*** shows dependent on the position of the peaks (x-variable), the ***probability to discard the background hypothesis*** $H_0$ and accept the signal hypothesis $H_1$.<br>
For the very special case where the two gaussian are overlapping, the blue line should be excactly horizontal. This is equivalent to sayin that the data can not be seperated in this case. Of course this imperfrection arises from imperfection of training and data preparation.

![](PARAMETERIZED-NEURAL-NETWORKS/animations/1d_signal.gif)


### 2D

![](../animations/2d_signal_circular.gif)

![](../animations/2d_signal_grid.gif)





