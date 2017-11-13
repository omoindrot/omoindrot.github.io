---
layout: post
title: Finetuning a model using tf.estimator
description: ""
excerpt: ""
date:   2017-11-09
permalink: /estimator
mathjax: true
use_math: true
comments: true
---

#TODO: first talk about finetuning a model
When I was a Teaching Assistant in [CS231n][cs231n], most of the student projects tried to finetune a model pre-trained on ImageNet.
This is because they didn't have enough data to build their own model, and is often the case.

- wrote a github gist

- tensorflow v1.4 brings stable `tf.data` module and function `tf.estimator.train_and_evaluate`
- good time to see what `tf.estimator` can do


The main interest in using `tf.estimator` is that all the training procedures are already implemented, and you don't need to worry about the details of training, evaluating or sending summaries to tensorboard.

In a few lines of code, we can do the whole logic of training and evaluation.

```python
estimator = tf.estimator.Estimator(model_fn, params=args, model_dir=LOG_DIR)

#TODO: change num_steps?
num_steps = 10000

train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=num_steps)
eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)

tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
```
#TODO: How to finetune a model with estimators? They have their own logic, with saving and training but getting into the model is pretty difficult.

We now have to specify `train_input_fn` and `eval_input_fn` that will feed the data into the model.  
We also have to define the model in `model_fn`.

---
## Data input
Let's suppose that we have a dataset containing 8 classes of animals: `["bear", "bird", "cat", "dog", "giraffe", "horse", "sheep", "zebra"]`. For each class, we have 100 training images and 25 validation images. The images have size `(224, 224, 3)`.  
Building a classifier on these images is pretty difficult, but we don't have enough data to train a very deep model.

The solution here is to use a model pre-trained on ImageNet. We'll see in the next parts how to define the model and load the weights from a pre-trained model.


### What is `train_input_fn`?
#TODO: change title

Using `tf.estimator` gives a standard way to think about data input. The interface of `train_input_fn` is the following:
- it takes no argument
- it returns:
  - `features`: `Tensor` or dictionary of tensors
  - `labels`: `Tensor` or dictionary of tensors


In our case the only features are the image itself, and the only label is the category of the image (8 categories in total).

The `features` and `labels` will be used directly in the model function during training and evaluation. The easiest way to have these images and labels tensors ready to be consumed by the model is to use a `tf.data.Dataset`.


### Building a tf.data.Dataset 

We begin with a list of image filenames and their labels. For instance:
```python
filenames = ['img1.jpg', 'img2.jpg', 'img3.jpg']
labels = [4, 2, 7]
```

We can build a dataset from these two lists by iterating through both files, using `tf.data.Dataset.from_tensor_slices`:

```python
dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
```

```python
if is_training: dataset = dataset.shuffle(buffer_size=100 * batch_size)
dataset = dataset.repeat(num_epochs)

dataset = dataset.map(_parse_function, num_parallel_calls=num_threads)
if is_training:
    dataset = dataset.map(training_preprocess, num_parallel_calls=num_threads)
else:
    dataset = dataset.map(val_preprocess, num_parallel_calls=num_threads)

dataset = dataset.prefetch(batch_size)

batched_dataset = dataset.batch(batch_size)
```

```python
iterator = batched_dataset.make_one_shot_iterator()
images, labels = iterator.get_next()
```




---
## Defining the model


---
## Loading pre-trained weights





Check mathjax: \\(f(x)\\)
Check mathjax: \\[f(x)\\]

Second check: $f(x)$

Third check:

$$
f(x)
$$

Fourth check: $$ f(x) $$

Link: [TensorFlow home page][link]

Link: [Openface blog post][openface-blog]


[cs231n]: https://cs231n.stanford.edu
[link]: https://tensorflow.org
[openface-blog]: http://bamos.github.io/2016/01/19/openface-0.2.0/
