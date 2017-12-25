Outline

Goals:
- use `tf.data` and `tf.estimator`
- finetune a pre-trained model in a clean way
- explain inner workings of `tf.estimator`?
  - already done in the TF blog post


## Introduction

- why finetuning is useful
- (why tf.data is useful?)
- why estimators are useful
- outline of the rest

## Estimators feel simple at first

- few lines of code
- a bit the same feeling as in keras: everything is easy when you want to do standard stuff (ex: train MNIST), but if you deviate from the standard it becomes much more difficult
  - because no access to tf.Session()
  - hard to debug because errors are deep into the `tf.estimator` code
  - structure is rigid, and there is a lack of documentation (except good 3 blog posts)

- everything needs to be standardized in tf.estimator -> tradeoff between convenience and structure / between ease of use and flexibility
  - same between tensorflow and keras

## Data input

- describe dataset
  - put images of each class
- we don't have a lot of data -> pre-trained weights

### A standard way to feed data

- everything in tf.estimator is about standards
- input_fn
  - no argument
  - returns `features, labels`

### Building a tf.data.Dataset

- explain the code


## Defining the model

- explain the tf.estimator model_fn
- Three steps
  - prediction
  - evaluation
  - training

## Loading pre-trained weights

- explain why it is tricky
  - how would we do with a session
  - but we don't have one
- explain tf.train.Scaffold
  - when is it called, how it works
- explain when checkpoints are saved
- explain how using direct initialization saves everything as tf.constant in the graph and makes memory blow up

- show results

## Conclusion
?
