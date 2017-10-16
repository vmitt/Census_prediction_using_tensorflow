# Census Prediction using Tensorflow
In this project, I'm going to show how you can use the **Tensorflow** to train **Census Dataset**, a model that can predict the income class of US population.

# Census dataset
This data set contains weighted census data extracted from the 1994 and 1995 Current Population Surveys conducted by the U.S. Census Bureau. The data contains 15 demographic and employment related variables. 

The instance weight indicates the number of people in the population that each record represents due to stratified sampling.

There are 199523 instances in the data file and 99762 in the test file. 

The census data is divided into two parts:

Training data(adult.data.txt)
Testing data(adult.test.txt)

The data was split into train/test in approximately 2/3, 1/3 proportions using MineSet's MIndUtil mineset-to-mlc. 

# Problem:
Predict the income class of US population.

# Prerequisites
1. Tensorflow
2. Pandas
3. Urlib

# Description
TensorFlow’s API (tf.estimator.learn) is used to configure, train, and evaluate the models. In this, we’ll use tf.estimator.learn to construct a neural network classifier and train it on the Census dataset to predict the income class of US population.

In this firstly we'll load the Census data to Tensorflow, using the urllib method, In this we directly provide the url of the dataset we want to use.

Next, we'll built a neural network classifier using the tf.estimator

tf.estimator the variety of predifined models, which can be used for training and evaluating the model.

Now then after configuring our model, now we'll train the model.

In the end, the model thus predict the income class of US population. We can check our logs using the Tensorboard.

## Launching Tensorboard
To run TensorBoard, use the following command:

    tensorboard --logdir=path/to/log-directory
