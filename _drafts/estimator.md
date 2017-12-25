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

TODOs:
- decide to use "I" or "we" (we when reader can be included?)
  - check in the post at the end
- try to justify the text?


In this post, we'll see a concrete TensorFlow example using [`tf.data`][tf-data] and [`tf.estimator`][tf-estimator]: finetune a pre-trained model (VGG on ImageNet) on a new task. Some inspiration comes from a [github gist][my-gist] that I wrote to help students kickstart their project in [CS231n][cs231n] (where I was a TA).

Getting the details of `tf.estimator` right can be difficult, but the rewards are huge. With `tf.estimator`, we get a lot of things for free: saving, evaluation, model exporting, distributed training...



## Introduction

Most of the time in deep learning, models are not trained from scratch. This is because deep learning models require a lot of data to be trained, and we often don't have a big enough dataset.  
The solution is to initialize the model with weights pre-trained on a bigger dataset, like ImageNet.

Estimators have been added to the "main" tensorflow in version `1.4` under `tf.estimator`. They still feel a bit difficult to work with, and there is a lack of simple tutorials for using them.

The main interest in using `tf.estimator` is that all the training procedures are already implemented, and you don't need to worry about the details of training, evaluating or sending summaries to tensorboard.
Estimators have multiple advantages, summed up in the [official guide][tf-guide].


The following sections will explain:
- [how](#estimators-feel-simple-at-first) estimators feel simple but are not
- [how](#data-input) to feed data into the estimator
- [how](#defining-the-model) to build the model
- [how](#loading-pre-trained-weights) to load pre-trained weights


## Estimators feel simple at first


In a few lines of code, we can do the whole logic of training and evaluation.

{% highlight python %}
estimator = tf.estimator.Estimator(model_fn)

train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, num_steps=10000)
eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)

tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
{% endhighlight %}


#TODO: How to finetune a model with estimators? They have their own logic, with saving and training but getting into the model is pretty difficult.

We now have to specify `train_input_fn` and `eval_input_fn` that will feed the data into the model.  
We also have to define the model in `model_fn`.

For a great example of this in action, go check out [the official blog post][estimator-blog] detailing how to build a customized estimator.


However, the issue is that it's difficult to slightly modify estimators because the whole structure is very rigid.
We'll see in [the last section](#loading-pre-trained-weights) how to use `tf.train.Scaffold` to initialize the weights of the model before training.

---
## Data input
We have a dataset containing 8 classes of animals: `["bear", "bird", "cat", "dog", "giraffe", "horse", "sheep", "zebra"]`. For each class, we have 100 training images and 25 validation images. The images have size `(224, 224, 3)`.  

TODO: put example images of each class

Building a deep learning classifier on these images is pretty difficult because we don't have enough data to train it.

The solution here is to use a model pre-trained on ImageNet. We'll see in the last part how to define the model and load the weights from a pre-trained model.


### A standard way to feed data

Using `tf.estimator` gives a standard way to think about data input. The interface of `train_input_fn` is the following:
- it takes no argument
- it returns:
  - `features`: `Tensor` or dictionary of tensors
  - `labels`: `Tensor` or dictionary of tensors

Example:
{% highlight python %}
def input_fn():

{% endhighlight %}


In our case the only features are the image itself, and the only label is the category of the image (8 categories in total).

The `features` and `labels` will be used directly in the model function during training and evaluation. The easiest way to have these images and labels tensors ready to be consumed by the model is to use a `tf.data.Dataset`.


### Building a tf.data.Dataset 

We begin with a list of image filenames and their labels. For instance:
{% highlight python %}
filenames = ['img1.jpg', 'img2.jpg', 'img3.jpg']
labels = [4, 2, 7]
{% endhighlight %}

We can build a dataset from these two lists by iterating through both files, using `tf.data.Dataset.from_tensor_slices`:

{% highlight python %}
dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
{% endhighlight %}

Here is how we transform the dataset for training:
{% highlight python %}
dataset = dataset.shuffle(buffer_size=100 * batch_size)
dataset = dataset.repeat(num_epochs)

# Use `num_parallel_calls` to have multiple threads process the input in parallel
dataset = dataset.map(_parse_function, num_parallel_calls=num_threads)
dataset = dataset.map(training_preprocess, num_parallel_calls=num_threads)

dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(1)  # make sure that we always have 1 batch ready
{% endhighlight %}

Finally, we create an iterator from the dataset and get the next element:
{% highlight python %}
# One shot iterator means it cannot be created again (once it runs out of data)
iterator = dataset.make_one_shot_iterator()
images, labels = iterator.get_next()
{% endhighlight %}


The resulting pair `(images, labels)` is a pair of tensors that can plugged into the graph as inputs.
One great advantage of using a `tf.data.Dataset` is that we don't need to feed anything into the graph.

Instead of using the standard method of feeding data into the graph with a `feed_dict` like this:
{% highlight python %}
# With tf.placeholder as input
sess.run(train_op, feed_dict={images: batch_img, labels: batch_labels})
{% endhighlight %}

We can directly call operations in the graph without feeding anything. The queues in `tf.data` will automatically fetch images and labels and batch them together.

{% highlight python %}
# With tf.data as input
sess.run(train_op)
{% endhighlight %}


---
## Defining the model

- define the model without finetune in mind
- easy steps:
  - define loss
  - define training op

{% highlight python %}
def model_fn(features, labels, mode, params):
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    vgg = tf.contrib.slim.nets.vgg
    logits, _ = vgg.vgg_16(features, num_classes=8, is_training=is_training)

    predictions = {
        'classes': tf.to_int32(tf.argmax(logits, axis=1)),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    # 1. Prediction mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # ---------------------------------------------------------------------
    # Using tf.losses, any loss is added to the tf.GraphKeys.LOSSES collection
    # We can then call the total loss easily
    softmax_loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    loss = tf.losses.get_total_loss()
{% endhighlight %}



Evaluation:
{% highlight python %}
    # Evaluation metrics
    accuracy = tf.metrics.accuracy(labels, predictions['classes'])
    eval_metric_ops = {'accuracy': accuracy}

    # 2. Evaluation mode
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)
{% endhighlight %}

Training:
{% highlight python %}
    # 3. Training mode
    # First we want to train only the reinitialized last layer fc8 for a few epochs.
    # We run minimize the loss only with respect to the fc8 variables (weight and bias).
    optimizer = tf.train.GradientDescentOptimizer(params.learning_rate)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

{% endhighlight %}


---
## Loading pre-trained weights

Everything works well if we follow the standard way to build a `tf.estimator`.  
It gets tricky when we try to deviate from this standard, because we don't have access to the tensorflow session here.

The estimator will take care of creating the session (a `tf.train.MonitoredSession`), save the variables and do all the work. It simplifies our code, but at the same time it makes it more complex to tinker with the model.


For example here, we want to initialize the model with pre-trained weights.


- show how we do it if we have the session
- introduce tf.train.Scaffold
- pass it into the estimator
- demo that it works


{% highlight python %}
    # Specify where the model checkpoint is (pretrained weights).
    model_path = params.model_path
    assert os.path.isfile(model_path), "Model file couldn't be found at %s" % model_path

    # Restore only the layers up to fc7 (included)
    # Calling function `init_fn(sess)` will load all the pretrained weights.
    variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['vgg_16/fc8'])
    init = tf.contrib.framework.assign_from_checkpoint_fn(model_path, variables_to_restore)
    init_fn = lambda scaffold, session: init(session)

    # We need to return the initialization operation within a scaffold for tf.estimator
    scaffold = tf.train.Scaffold(init_fn=init_fn)
{% endhighlight %}



[my-gist]: https://gist.github.com/omoindrot/dedc857cdc0e680dfb1be99762990c9c
[cs231n]: https://cs231n.stanford.edu
[link]: https://tensorflow.org
[openface-blog]: http://bamos.github.io/2016/01/19/openface-0.2.0/
[tf-guide]: https://www.tensorflow.org/programmers_guide/estimators#advantages_of_estimators
[tf-data]: https://www.tensorflow.org/api_docs/python/tf/data
[tf-estimator]: https://www.tensorflow.org/api_docs/python/tf/estimator
[estimator-blog]: https://developers.googleblog.com/2017/12/creating-custom-estimators-in-tensorflow.html

