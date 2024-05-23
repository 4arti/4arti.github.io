---

layout: post
title: "Exploring the Landscape of Convolutional Neural Networks"
date: 2024-05-23
description: >
  A review of CNN architectures

tags: CNN
categories: [Deep Learning, Research]
related_posts: false
featured: false
published: true
author: Aarti
mermaid:
  enabled: true
---

> **Objective** :
> - Fundamentals of CNN
> - Understanding the Evolution of CNN Architectures
> - Practical Applications

---

# 1. Fundamentals of Convolutional Neural Networks(CNN)

At their core, CNNs leverage a series of convolutional layers, pooling layers, and fully connected layers to process input data. The convolutional layers apply learnable filters to input images, extracting features such as edges, textures, and patterns. Pooling layers then downsample the feature maps, reducing their spatial dimensions while preserving important features. Finally, fully connected layers combine these features to make predictions about the input data.

> A key difference between *Dense* layer and a *Convolutional* layer is that, dense layer learns global patterns whereas convolutional layer learns local patterns. This key distinction allows convolutional layers to capture spatial hierarchies and learn translational invariance.

**Translational Invariance:**

A CNN can learn to detect patterns regardless of their position within the input image. For example, a CNN trained to recognize cats should be able to identify a cat regardless of whether it appears in the center or the corner of the image.
Translational invariance is a desirable property in tasks such as object recognition and classification, as it allows models to generalize better to unseen data and variations in object position or orientation

**Spatial Hierarchy:**

 At the lowest layers of a CNN, simple features such as edges, corners, and textures are learned. These features have a small spatial scope and represent basic visual elements present in the input data. As we move deeper into the network, features become more complex and encompass larger spatial regions. For example, a deeper layer might learn features like shapes, object parts, or entire objects.


The basic components of a CNN architecture are:
- Convolutional Layer
- Pooling Layer
- Activation Function
- Batch Normalization
- Dropout
- Fully Connected Layer

## Convolutional Layer

A convolutional layer in a Convolutional Neural Network (CNN) performs feature extraction by applying a set of learnable filters (kernels) to the input data. Each filter scans across the input data, computing the dot product between its weights and the values in its receptive field, resulting in a feature map that highlights the presence of certain patterns or features in the input.


<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/convolution_operation.png" title="Convolution Operation" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Convolution operation 
    <a href="https://www.researchgate.net/figure/Convolution-operation_fig2_355656417">Source</a>
</div>

<br>
Key aspects of a convolutional layer:

<br> 

**Feature Detection**: The convolutional filters detect various features such as edges, textures, and shapes present in the input data.

**Spatial Hierarchies**: Through the hierarchy of layers, convolutional layers capture increasingly complex and abstract spatial structures, building upon features learned in previous layers.

**Parameter Sharing**: By sharing weights across different spatial locations, convolutional layers learn translational invariance, enabling them to detect patterns regardless of their position within the input.









