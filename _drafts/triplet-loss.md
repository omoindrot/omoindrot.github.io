---
layout: post
title: Triplet Loss and Online Triplet Mining in TensorFlow
description: ""
excerpt: "Triplet loss is difficult to get right, especially inside a TensorFlow graph. We'll see a naive approach and follow with a much more efficient sampling method for triplet loss."
author: "Olivier Moindrot"
date:   2018-03-17
permalink: /triplet-loss
mathjax: true
comments: true
---

About two years ago, I was doing face recognition during my internship at [Reminiz][reminiz] and I answered a [question][stackoverflow] on stackoverflow about implementing triplet loss in TensorFlow. I concluded by saying:

>Clearly, implementing triplet loss in Tensorflow is hard, and there are ways to make it more efficient than sampling in python but explaining them would require a whole blog post !


Triplet loss is known to be difficult to implement, especially if you add the constraints of building a computational graph in TensorFlow.

In this post, I will define the triplet loss and the different strategies to sample triplets.
I will then explain how to correctly implement triplet loss with online triplet mining in TensorFlow.

All the code can be found on this [github repository][github].


<br>



**Table of contents**

* TOC
{:toc}

---


## Triplet loss and triplet mining

### Why not just use softmax?

The triplet loss for face recognition has been introduced by the paper [*FaceNet: A Unified Embedding for Face Recognition and Clustering*][facenet] from Google.
They describe a new approach to build face embeddings using online triplet mining, which will be discussed in the [next section](#offline-and-online-triplet-mining).

Usually in supervised learning we have a fixed number of classes and train the network using the softmax cross entropy loss.
However in some cases we need to be able to have a variable number of classes.
In face recognition for instance, we need to be able to compare two unknown faces and say whether they are from the same person or not.

Triplet loss in this case is a way to learn good embeddings for each face. In the embedding space, faces from the same person should be close together and form well separated clusters.


### Definition of the loss

|![triplet-loss-img] |
|:--:|
| *Triplet loss on two positive faces (Obama) and one negative face (Macron)* |

<br>

The goal of the triplet loss is to make sure that:
- Two examples with the same label have their embeddings close together in the embedding space
- Two examples with different labels have their embeddings far away. 

However, we don't want to push the train embeddings of each label to collapse into very small clusters.
The only requirement is that given two positive examples of the same class and one negative example, the negative should be farther away than the positive by some margin.

<br>
To formalise this requirement, the loss will be defined over **triplets** of embeddings:
- an **anchor**
- a **positive** of the same class as the anchor
- a **negative** of a different class

For some distance on the embedding space $d$,  the loss of a triplet $(a, p, n)$ is:

$$
\mathcal{L} = max(d(a, p) - d(a, n) + margin, 0)
$$

We minimize this loss, which pushed $d(a, p)$ to $0$ and $d(a, n)$ to be greater than $d(a, p) + margin$. As soon as $n$ becomes an "easy negative", the loss becomes zero.


### Triplet mining

Based on the definition of the loss, there are three categories of triplets:
- **easy triplets**: triplets which have a loss of $0$, because $d(a, p) + margin < d(a,n)$
- **hard triplets**: triplets where the negative is closer to the anchor than the positive, i.e. $d(a,n) < d(a,p)$
- **semi-hard triplets**: triplets where the negative is not closer to the anchor than the positive, but still has positive loss: $d(a, p) < d(a, n) < d(a, p) + margin$

Each of these definitions depend on where the negative is, relatively to the anchor and positive. We can therefore extend these three categories to the negatives: hard negatives, semi-hard negatives or easy negatives.

The figure below shows the three corresponding regions of the embedding space for the negative.

|![triplet-types-img] |
|:--:|
| *The three types of negatives, given an anchor and a positive* |


<br>
Choosing what kind of triplets we want to train on will greatly impact our metrics.
In the original Facenet [paper][facenet], they pick a random semi-hard negative for every pair of anchor and positive, and train on these triplets.

### Offline and online triplet mining

We have defined a loss on triplets of embeddings, and have seen that some triplets are more useful than others. The question now is how to sample, or "mine" these triplets.

**Offline triplet mining**

The first way to produce triplets is to compute them offline, at the beginning of each epoch for instance.
We compute all the embeddings on the training set, and then only select hard or semi-hard triplets.
We can then train one epoch on these triplets.

Overall this technique is not very efficient since we need to do a full pass on the training set to generate triplets.
It also requires to update the offline mined triplets regularly.

Moreover, we need to compute $3B$ embeddings to only get $B$ triplets.

<br>
**Online triplet mining**

Online triplet mining was introduced in *Facenet* and has been well described by Brandom Amos in his blog post [*OpenFace 0.2.0: Higher accuracy and halved execution time*][openface-blog].

The idea here is to compute useful triplets on the fly, for each batch of inputs.
Given a batch of $B$ examples, we can find a maximum of $B^3$ triplets.
Of course, most of these triplets are not **valid** (i.e. they don't have 2 positives and 1 negative).
Among valid triplets, most will be *easy triplets* with loss $0$.

This technique gives you more triplets for a single batch of inputs, and doesn't require any offline mining. It is therefore much more efficient. We will see an implementation of this in the last [part](#a-better-implementation-with-online-triplet-mining).


### Strategies in online mining

Detailed explanation in the paper [*In Defense of the Triplet Loss for Person Re-Identification*][in-defense]

- batch all
- batch hard

---

## A naive implementation of triplet loss

In the [stackoverflow answer][stackoverflow], I gave a simple implementation of triplet loss for offline triplet mining:

```python
anchor_output = ...    # shape [None, 128]
positive_output = ...  # shape [None, 128]
negative_output = ...  # shape [None, 128]

d_pos = tf.reduce_sum(tf.square(anchor_output - positive_output), 1)
d_neg = tf.reduce_sum(tf.square(anchor_output - negative_output), 1)

loss = tf.maximum(0.0, margin + d_pos - d_neg)
loss = tf.reduce_mean(loss)
```

The network is replicated three times (with shared weights) to produce the embeddings of $B$ anchors, $B$ positives and $B$ negatives.
We then simply compute the triplet loss on these embeddings.

This is an easy implementation, but also a very inefficient one because it uses offline triplet mining.

---

## A better implementation with online triplet mining

All the relevant code is available on github in file [`model/triplet_loss.py`][triplet-loss-file].


### Implementation in TensorFlow

*There is an existing implementation of triplet loss with semi-hard online mining in TensorFlow: [`tf.contrib.losses.metric_learning.triplet_semihard_loss`][tf-triplet-loss].
Here we will not follow this implementation and start from scratch.*



**Compute the distance matrix**

Since we have triplets, it will be easier to work with 3D tensors of shape $(B, B, B)$ where $B$ is the batch size. 
This is useful because an element at index `(i, j, k)` will refer to the triplet with anchor $i$, positive $j$ and negative $k$.

We will first compute the distance matrix in an efficient way:


- distance matrix

- reshape to $(B, B, 1)$ for $d(a, p)$
- reshape to $(B, 1, B)$ for $d(a, n)$

In our case we work
$$
d(a, p) - d(a, n) + margin = ||a - p||^2 - ||
$$

**Batch all strategy**
- we can combine and get the triplet loss

```python
triplet_loss = positive_norm - 2.0 * anchor_positive_dot_product - \
               negative_norm + 2.0 * anchor_negative_dot_product + margin
```

- now we need to remove the invalid triplets: ...
- we want to take the average
  - but only on triplets with positive loss
  - otherwise as the number of hard or semi-hard triplets decreases, the loss will also decrease and the learning rate will be artificially lowered

- create a 3D mask which will be `True` only if $(a, p, n)$ is valid
- count the number of positive triplets
- sum and divide by this number

**Batch hard strategy**
...


### Testing our implementation

If you don't trust that the implementation above works as expected, then you're right!
The only way to make sure that there is no bug in the implementation is to write tests for every function in [`model/triplet_loss.py`][triplet-loss-file]

This is especially important for tricky functions like this that are difficult to implement in TensorFlow but much easier to write using three nested for loops in python for instance.
The tests are written in [`tests/test_triplet_loss.py`][triplet-loss-test], and compare the result of our TensorFlow implementation with the results of a simple numpy implementation.

To check yourself that the tests pass, run:
```bash
python -m tests.test_triplet_loss
```

## Conclusion

TensorFlow doesn't make it easy to implement triplet loss, but with a bit of effort we can build a good-looking version of triplet loss with online mining.

The tricky part is mostly how to compute efficiently the distances between embeddings, and how to mask out the invalid / easy triplets.


## Resources

- [github repo][github] for this blog post
- [Facenet paper][facenet]
- Detailed explanation of online triplet mining in [*In Defense of the Triplet Loss for Person Re-Identification*][in-defense]
- blog post by Brandom Amos on online triplet mining: [*OpenFace 0.2.0: Higher accuracy and halved execution time*][openface-blog].
- source code for the built-in TensorFlow function for semi hard online mining triplet loss: [`tf.contrib.losses.metric_learning.triplet_semihard_loss`][tf-triplet-loss].



[github]: https://github.com/omoindrot/tensorflow-triplet-loss
[triplet-types-img]: images/triplets.png
[triplet-loss-img]: images/triplet_loss.png
[openface-blog]: http://bamos.github.io/2016/01/19/openface-0.2.0/
[stackoverflow]: https://stackoverflow.com/a/38270293/5098368
[facenet]: https://arxiv.org/abs/1503.03832
[in-defense]: https://arxiv.org/abs/1703.07737
[reminiz]: https://reminiz.com
[triplet-loss-file]: https://github.com/omoindrot/tensorflow-triplet-loss/blob/master/model/triplet_loss.py
[triplet-loss-test]: https://github.com/omoindrot/tensorflow-triplet-loss/blob/master/tests/test_triplet_loss.py
[tf-triplet-loss]: https://www.tensorflow.org/api_docs/python/tf/contrib/losses/metric_learning/triplet_semihard_loss
