# Introduction

Let's start by introducing the problem that we are dealing with, the task is to use the Califrona census data to build a model of housing prices in the state, which includes metrics such as:
* Population
* Median Income
* Median Housing Price
For each housing block group, which represent the smallest geographical unit for which the US census bureau publishes sample data. 

The task of the model is to learn from the data and predict the median housing price in any district, given all the other metrics.

# Frame the Problem

The first thing to establish is to get an idea for the objective that this model will be used for, while you may be a nerd like me and get the most enjoyment out of just training the model, people down here in the real world want something from your model, typically a business oriented thing, whatever that is, so the first step is to **ask what the goal is**, typically to your boss, and their response is:

	This output, the median price of a house in a given district, will be fed to another machine learning model "alongside other signals".

Signals here referring to any input that is fed to the model.

The purpose of this other model is to make a decision on whether something is worth investing it or not, as such it is critical that this is done right.

The second thing to do is to **ask about what the current solution looks like**. This way you can get a reference for what you are working with, and what to improve on.

With these information now at your disposal, we can finally start framing our problem, we do so by asking:

* What kind of learning method do we employ here?
	* Is it supervised, unsupervised, or semi-supervised?
* What kind of learning algorithm do we need to use?
	* Is it a classification task, a regression task, or something else?
* Should you use batch or online learning methods?

This is a supervised learning task, because the data is already organized for us, it is also a regression task, since we are trying to come up with a function that, given inputs, best predicts what the output will be, in fact, it is polynomial regression, since we have more than one input. It is also a univariate task, in the sense that we are only trying to predict one value, if it was such a case where we would have to predict multiple values, then we would have a multivariate task. This task will use batch learning since the data is small enough to be trained on at once, and there is no continuous flow of data.

# Select a Performance Measure

So now that we have decided on what needs to be done, it is a good idea now to select a performance measure for our purposes, let's begin with the most common one, the Root Mean Squared Error or RMSE, here's what the general formula looks like: $$RMSE(\textbf{X}, h)=\sqrt{\frac{1}{m} \sum_{i = 1}^{m} \ (h (\textbf{x}^{(i)})-y^{(i)})^2}$$
There seems to be a lot to unpack here in terms of notation, so let's go over them real quick:

* $m$ - represents the number of instances (or samples) of the validation set we have, essentially if our validation set has 2000 samples, then $m = 2000$
* $i$ - represents the $i^{th}$ sample from the validation set
* $h$ - is our hypothesis function, i.e. our prediction function, which represents the function that our model uses to estimate a value, in our case the price, based on the input supplied to it
* $\textbf{x}$ - represents an input vector, which consists of all the relevant features that are to be processed as inputs, in such a case when we feed an input $\textbf{x}$ to our hypothesis function $h(\textbf{x})$ the result is a predicted value $\hat{y}$, known as "y hat", which is the output that we predict.
* $y$ - represents the actual value, there is a reason why this is a performance measure, we run it against the actual value, and see how accurate our model is.

This was an overview of the notation used, let's explore another performance measure, the mean absolute error, and then conclude by going over the scenarios when these different measures are typically used: $$\text{MAE}(\textbf{X}, h)=\frac{1}{m} \sum_{i=1}^{m} \ |h(\textbf{x}^{(i)})-y^{(i)}|$$
So what exactly is the difference between these two? Well, typically the most common performance measure is the RMSE, but when you run into a situation when there are many outliers in your validation set, it would make more sense to use the MAE method, as it penalizes outliers less.

Now that we have looked at the bigger picture, it is always a good idea to double check the things you have gathered so far, like making sure that this is indeed a regression task you are working on, as it would suck to know a few months into your project that this in fact only had to be a classification task that classified the price into three distinct categories. 

With this out of the way, it is time to move on to working on the actual data, to this end, we will move to jupyter notebooks!