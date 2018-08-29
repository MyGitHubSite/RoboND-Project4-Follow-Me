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

**Hyperparameters: LR=0.005, Batch=32, Epochs=20, Epoch Steps=138**

    Run 1: 1x1Conv(16), Enc(32), 1x1Conv(64), Decoder
    Run 2: 1x1Conv(16), Enc(32), Enc(64), 1x1Conv(128), Dec(64), Dec(32)
    Run 3: 1x1Conv(16), Enc(32), Enc(64), Enc(128), 1x1Conv(256), Dec(128), Dec(64), Dec(32)
    Run 4: 1x1Conv(16), Enc(32), Enc(64), Enc(128), Enc(256), 1x1 Conv(512), Dec(256), Dec(128), Dec(64), Dec(32)

**Hyperparameters: Same as runs 1-4 but changed Epoch Steps to 200**

    Run 5: 1x1Conv(16), Enc(32), Enc(64), Enc(128), 1x1Conv(256), Dec(128), Dec(64), Dec(32)
    Run 6: 1x1Conv(16), Enc(32), Enc(64), Enc(128), Enc(256), 1x1 Conv(512), Dec(256), Dec(128), Dec(64), Dec(32)

**Hyperparameters: Same as Run 5 and 6 but changed the number of filters on the first 1x1conv layer to 32**

    Run 7: 1x1Conv(32), Enc(32), Enc(64), Enc(128), 1x1Conv(256), Dec(128), Dec(64), Dec(32)
    Run 8: 1x1Conv(32), Enc(32), Enc(64), Enc(128), Enc(256), 1x1 Conv(512), Dec(256), Dec(128), Dec(64), Dec(32)

**Hyperparameters: Same as Run 7 but changed Epochs to 50**

    Run 9: 1x1Conv(32), Enc(32), Enc(64), Enc(128), 1x1Conv(256), Dec(128), Dec(64), Dec(32)

**Model Results**

Run | Epochs |  LR   | Batch | Steps/Epoch | Runtime | Weight | IOU   | Score | HTML
--- | :----: | :---: | :---: | :---------: | :-----: | :----: | :---: | :---: | ----
1   | 20     | 0.005 | 32    | 138         | 12m15s  | 0.70   | 0.27  | 0.19 | [html](../blob/master/LICENSE)
2   | 20     | 0.005 | 32    | 138         | 15m20s  | 0.72   | 0.52  | 0.37
3   | 20     | 0.005 | 32    | 138         | 16m45s  | 0.75   | 0.56  | 0.41
4   | 20     | 0.005 | 32    | 138         | 17m41s  | 0.68   | 0.48  | 0.33
5   | 20     | 0.005 | 32    | 200         | 25m18s  | 0.76   | 0.56  | 0.42
6   | 20     | 0.005 | 32    | 200         | 28m40s  | 0.67   | 0.53  | 0.36
7   | 20     | 0.005 | 32    | 200         | 27m37s  | 0.77   | 0.59  | 0.46
8   | 20     | 0.005 | 32    | 200         | 28m50s  | 0.76   | 0.52  | 0.397
9   | 50     | 0.005 | 32    | 200         | ms  | 0.   | 0. | 0.

### Future Enhancements
---
    1 Collect more data
    2 Pooling layers, dropout
    3 Deeper layers (more memory)
