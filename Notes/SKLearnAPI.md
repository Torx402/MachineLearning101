# SKLearn API

This document will go over how the sklearn API works, and what its parts are.

## Consistency

All objects have a consistent and simple interface

#### Estimators

This is any object that can estimate some parameters of a given dataset, in the example we looked previously, `imputer` was one such estimator, and it performed estimation using the `fit()` method, where it estimated the the median of all attributes in the dataset. The dataset(s) is the only parameter that the estimator takes, with other parameters simply being *hyperparameters*, such as the strategy hyperparameter, which we set to "median" in our example, to find the median.

#### Transformers

No, we are not referring to the awesome mechatrons that can transform into cars. Transformers are estimators that can transform a dataset, the imputer is a transformer and it does so using the `transform()` method. It returns the transformed datatset as a numpy array.

#### Predictors

Some estimators are known as predictors, such as `LinearRegression`, it can predict based on a set of instances passed through its argument using the `predict()` method, and returns a set of predictions it has made. It also has a `score()` method, which measures the quality of predictions given a test set, and corresponding label set if it is a supervised machine learning task.

#### Inspection

All of the estimator's hyperparameters are accessible via public instance variables, such as `imputer.strategy` and its learned parameters are accessed with an underscore at the end, such as `imputer.statistics_`.

## Nonproliferation of Classes

All of the classes used to store and return data are either NumPy arrays or SciPy sparse matrices, with hyperparameters just being python strings or numbers.

## Composition

More complex objects are reusing existing blocks as much as possible, this means that it is easy to create a Pipeline estimator that consists of some number of transformers, ending with a final estimator.

## Sensible Defaults

SKLearn provides reasonable default values for most parameters, meaning that getting a model up and running is quick and easy.
