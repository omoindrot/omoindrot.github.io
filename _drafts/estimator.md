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

The code used in this post can be found on [github][github]. It is commented and very readable.



## Introduction

Most of the time in deep learning, models are not trained from scratch. This is because deep learning models require a lot of data to be trained, and we often don't have a big enough dataset.  
The solution is to initialize the model with weights pre-trained on a bigger dataset, like ImageNet.

Estimators have been added to the "main" TensorFlow in version `1.4` under `tf.estimator`. They still feel a bit difficult to work with, and there is a lack of simple tutorials for using them.

The main interest in using `tf.estimator` is that all the training procedures are already implemented, and you don't need to worry about the details of training, evaluating or sending summaries to TensorBoard.
Estimators have multiple advantages, summed up in the [official guide][tf-guide].


Below is a diagram explaining how Estimators fit in TensorFlow ecosystem. Estimators are on the same level as Keras as they have the same purpose: make it easy to create models. Both can use `tf.data` and `tf.layers`.

|![estimator-image]  |
|:--:|
| *TensorFlow ecosystem* |  

\_

  


The following sections will explain:
- how estimators feel simple but are not
- how to feed data into the estimator
- how to build the model
- how to load pre-trained weights


## Estimators feel simple at first


In a few lines of code, we can do the whole logic of training and evaluation.

{% highlight python %}
estimator = tf.estimator.Estimator(model_fn)

train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, num_steps=10000)
eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)

tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
{% endhighlight %}


We now have to specify `train_input_fn` and `eval_input_fn` that will feed the data into the model, and the `model_fn` that defines the model.  
For a great example of this in action, go check out [the official blog post][estimator-blog-3] detailing how to build a custom estimator.


However, the issue is that it's difficult to slightly modify estimators because the whole structure is very rigid. For instance we can't easily access the TensorFlow Session inside `model_fn`, which makes it hard to initialize weights from a pre-trained model.  
We'll see in [the last section](#loading-pre-trained-weights) how to use `tf.train.init_from_checkpoint` to work around that.

---
## Data input
We have a Dataset containing 8 classes of animals: `["bear", "bird", "cat", "dog", "giraffe", "horse", "sheep", "zebra"]`. For each class, we have 100 training images and 25 validation images.
We resize the images to have size `(224, 224, 3)`.  

![bear-image] | ![dog-image]

<p align="center">
<em>Left: dog, Right: bear</em>
</p>

Building a deep learning classifier on these images is pretty difficult because we don't have enough data to train it (only 800 training images).  


### A standard way to feed data

Using `tf.estimator` gives a standard way to think about data input. The interface of `train_input_fn` is the following:
- it takes no argument
- it returns:
  - `features`: `Tensor` or dictionary of tensors
  - `labels`: `Tensor` or dictionary of tensors


In our case the only features are the image itself, and the only label is the category of the image (8 categories in total).

The `features` and `labels` will be used directly in the model function during training and evaluation. The easiest way to have these images and labels tensors ready to be consumed by the model is to use a `tf.data.Dataset`.


### Building a tf.data.Dataset 

We begin with a list of image filenames and their labels. For instance:
{% highlight python %}
filenames = ['img1.jpg', 'img2.jpg', 'img3.jpg']
labels = [4, 2, 7]
{% endhighlight %}

We can build a Dataset from these two lists by iterating through both files, using `tf.data.Dataset.from_tensor_slices`:

{% highlight python %}
# Make sure that filenames and labels are Tensors
dataset = tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
{% endhighlight %}

Here is how we transform the Dataset for training:
{% highlight python %}
dataset = dataset.shuffle(buffer_size=len(filenames))
dataset = dataset.repeat(num_epochs)

# Use `num_parallel_calls` to have multiple threads process the input in parallel
dataset = dataset.map(_parse_function, num_parallel_calls=num_threads)
dataset = dataset.map(training_preprocess, num_parallel_calls=num_threads)

dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(1)  # make sure that we always have 1 batch ready
{% endhighlight %}

Finally, we create an iterator from the Dataset and get the next element:
{% highlight python %}
# One shot iterator means it cannot be created again (once it runs out of data)
iterator = dataset.make_one_shot_iterator()
images, labels = iterator.get_next()
{% endhighlight %}


The resulting pair `(images, labels)` is a pair of tensors that can plugged into the graph as inputs.
One great advantage of using a `tf.data.Dataset` is that we don't need to feed anything into the graph.

Instead of using the standard method of feeding data into the graph with a `feed_dict` like this:
{% highlight python %}
# With tf.placeholder as input, we need to input batches of data at each step
sess.run(train_op, feed_dict={images: batch_img, labels: batch_labels})
{% endhighlight %}

We can directly call operations in the graph without feeding anything. The queues in `tf.data` will automatically fetch images and labels and batch them together.

{% highlight python %}
# With tf.data as input, no need to input anything
sess.run(train_op)
{% endhighlight %}


---
## Defining the model

The main graph used for training, evaluation and prediction is coded in the `model_fn` of the Estimator.

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

    softmax_loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    loss = tf.losses.get_total_loss()

    # Evaluation metrics
    accuracy = tf.metrics.accuracy(labels, predictions['classes'])
    eval_metric_ops = {'accuracy': accuracy}

    # 2. Evaluation mode
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)

    # 3. Training mode
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'vgg_16/fc8')

    optimizer = tf.train.GradientDescentOptimizer(params.learning_rate)
    train_op = optimizer.minimize(loss, var_list=var_list, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
{% endhighlight %}


---
## Loading pre-trained weights

Everything works well if we follow the standard way to build a `tf.estimator`.  
It gets tricky when we try to deviate from this standard, because we don't have access to the TensorFlow Session here.

The Estimator will take care of creating the session (a `tf.train.MonitoredSession`), save the variables and do all the work. It simplifies our code, but at the same time it makes it more complex to tinker with the model.


For example here, we want to initialize the model with pre-trained weights. If we didn't use Estimators, we could follow the [guidelines][tf-saver] of TensorFlow and use a Saver and the current Session to restore weights:
{% highlight python %}
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, params.model_path)
{% endhighlight %}

It is possible to do this using the [`scaffold`][tf-scaffold] object, but here is a simpler way to initialize weights. We use `tf.train.init_from_checkpoint`, providing an `assignment_map` of variables to restore. If the string finishes with a `/`, it means we load every variable from that scope.  
The way this works is that the variables initializers get overwritten by a constant initializer containing the pre-trained weights.

>Be careful to only load these weights in training mode, not in eval or predict.

TODO: if we restart training, what happens??
      does it load from checkpoint or load from log_dir ?


{% highlight python %}
# Specify where the model checkpoint is (pretrained weights).
assert os.path.isfile(params.model_path),\
       "Model file couldn't be found at {}".format(params.model_path)

# Be careful to only run `init_from_checkpoint` in training mode
assignment_map = {
	'vgg_16/conv1/': 'vgg_16/conv1/',
	'vgg_16/conv2/': 'vgg_16/conv2/',
	'vgg_16/conv3/': 'vgg_16/conv3/',
	'vgg_16/conv4/': 'vgg_16/conv4/',
	'vgg_16/conv5/': 'vgg_16/conv5/',
	'vgg_16/fc6/': 'vgg_16/fc6/',
	'vgg_16/fc7/': 'vgg_16/fc7/',
}
tf.train.init_from_checkpoint(params.model_path, assignment_map)
{% endhighlight %}

Having to specify the whole `assignment_map` can feel cumbersome, but it's a good way to avoid making mistakes: all the pre-trained variables can be found in one place.


## Conclusion


### Resources

- [full code][github] (clean and commented)
- Official programming guide for custom [`tf.estimator`][tf-guide]
- Official blog posts about Estimators:
  - [part 1][estimator-blog-1]: introduction to Estimators and focus on pre-made Estimators
  - [part 2][estimator-blog-2]: usage of feature columns for data input
  - [part 3][estimator-blog-3]: create a custom Estimator
- original [gist][my-gist]



[bear-image]: images/bear.jpg
[dog-image]: images/dog.jpg
[estimator-image]: https://3.bp.blogspot.com/-l2UT45WGdyw/Wbe7au1nfwI/AAAAAAAAD1I/GeQcQUUWezIiaFFRCiMILlX2EYdG49C0wCLcBGAs/s1600/image6.png
[my-gist]: https://gist.github.com/omoindrot/dedc857cdc0e680dfb1be99762990c9c
[github]: https://github.com/omoindrot/tensorflow_finetune
[cs231n]: https://cs231n.stanford.edu
[tf-guide]: https://www.tensorflow.org/programmers_guide/estimators#advantages_of_estimators
[tf-data]: https://www.tensorflow.org/api_docs/python/tf/data
[tf-estimator]: https://www.tensorflow.org/api_docs/python/tf/estimator
[tf-scaffold]: https://www.tensorflow.org/api_docs/python/tf/train/Scaffold
[tf-saver]: https://www.tensorflow.org/programmers_guide/saved_model#restoring_variables
[estimator-blog-1]: https://developers.googleblog.com/2017/09/introducing-tensorflow-datasets.html
[estimator-blog-2]: https://developers.googleblog.com/2017/11/introducing-tensorflow-feature-columns.html
[estimator-blog-3]: https://developers.googleblog.com/2017/12/creating-custom-estimators-in-tensorflow.html

