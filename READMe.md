# Udacity RoboND - Project 4: Deep Learning - Follow Me
---

### Hardware Setup: local machine
Intel CPU
2 Nvidia 980Ti GTX
Samsung 950 M.2 (OS)
1 Samsung 2TB SSD (Data)

### Software Setup
Windows 10, CUDA 8.0, Cudnn 7.12
Anaconda with Python 3.5.2   
used environment.yml from repo but replaced tensorflow with tensorflow-gpu

### What are we trying to do in this project
---
We are trying to locate a target in a picture and determine where in the picture the target is located.  For this we need to use a fully convolutional network (FCN) which retains spatial information, rather than a fully connected network which does not.

An FCN can extract features with different levels of complexity and segment them into separate categories. In this project we are interested in segmenting into: 1) the target, 2) other people, and 3) the background.

### Network
---
An FCN consists of three sections: 

    1) an encoder section: extracts features from the image.
    2) a 1x1 convolution layer: helps to reduce the dimensionality of a layer without losing information about pixel locations
    3) a decoder section: upscales the output from the encoder back to the same size as origional image.

I tried various combinations fo FCNs and hyperparameters to achieve the required final score > 0.40.  My final chosen FCN consisted of:

    output layer
    d1: decoder layer1, 32 filters, skip connection to c1
    d2: decoder layer2, 64 filters, skip connection to e1
    d3: decoder layer3, 128 filters, skip connection to e2
    c2: 1x1 convolutional layer, 256 filters
    e3: encoder layer3, 128 filters
    e2: encoder layer2, 64 filters
    e1: encoder layer1, 32 filters
    c1: 1x1 convolutional layer, 32 filters
    Input

A table summarizing my results with varisous FCN and hyperparameters is shown results below.

### Hyperparameters
---
learning_rate = 0.005   # 0.001
batch_size = 32         # 64
num_epochs = 20         # 15
steps_per_epoch = 200   # 4131//batch_size+1    # 1000
validation_steps = 50   # 1184//batch_size+1   # 50
workers = 8             # 4

### Results
---
There are three different predictions available from the helper code provided:  

    patrol_with_targ: Test how well the network can detect the hero from a distance.
    patrol_non_targ: Test how often the network makes a mistake and identifies the wrong person as the target.
    following_images: Test how well the network can identify the target while following them.
    
    number true positives: 539, number false positives: 0, number false negatives: 0
    number true positives: 0, number false positives: 38, number false negatives: 0
    number true positives: 131, number false positives: 3, number false negatives: 170
    
    weight | 0.7604994324631101
    final_iou | 0.5452292991525213
    final_score | 0.4146465725677517
    
<strong>Modified DH Parameter Table</strong>

i | epochs | lr | parameters | time | weight | iou | score
--- | --- | --- | --- | ---
1 | 20 | 0.005 | 100000 | 15min | 0.70 | 0.40 | 0.18
2 | 20 | 0.005 | 100000 | 15min | 0.70 | 0.40 | 0.18
3 | 20 | 0.005 | 100000 | 15min | 0.70 | 0.40 | 0.18

Run | Epochs | LR | parameters | time | weight | iou | score
--- | --- | --- | --- | --- | --- | --- | ---
1 | 20 | 0.005 | 100000 | 15min | 0.70 | 0.40 | 0.18





*Still* | `renders` | **nicely**
1 | 2 | 3

### Future Enhancements
---
    1 Collect more data
    2 Pooling layers, dropout
    3 Deeper layers (more memory)
