# Feature Scaling

This is possible one of the most important transformations in the data preparation phase. Feature Scaling involves transforming the range of values of the attributes such that they all have the same range. If we consider our data, an attribute like total rooms ranges from 6 to 39,320, while median income only ranges from 0 to 15, obviously there's a massive difference in the scale of the values that our attributes can take on. To remedy this, there are two main methods(note that such transformations are only applied to the input attributes, not the targets):
* Min-max scaling (Normalization)
* Standardization

## Min-Max Scaling

Min-max scaling involves squishing down all the values to fit into the range 0-1, this is done by the following: $$x_{t} = \frac{x - min}{max - min}$$

Where:

$x_{t}$ is the transformed sample

$x$ is the sample before transformation

SKLearn provides a `MinMaxScaler` transformer for this, with a `feature_range` hyperparameter in case you would like to change the default range of 0-1

## Standardization

The other method is known as standardization , this approaches the problem in a different manner, suppose $x \in X$, where $x$ is a data sample of some attribute $X$. Then: $$x_t = \frac{x - X.mean()}{\sigma^2}$$

This approach ensures that the resulting distribution has unit variance. SKLearn provides an implementation of standardization via `StandardScaler`.

## So which one do we pick?

The answer, as always, is it depends. Standardization does not squish the values into some specific range, which may not be useful for models that require an input of 0-1, such as neural networks. However, standardization is much less error prone to outliers. Suppose that, by error, an entry in our dataset had a median income value of 100. In the case of normalization, it would squish the values down to a range of 0-0.15, rather than 0-1, which throws the results off, while standardization would not be affected as much.