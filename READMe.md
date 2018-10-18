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
We are trying to locate a target ("what") in a picture and determine "where"  in the picture the target is located.  
A typical classification model only needs to understand what is in an image and does not retain pixel spatial information.  However, in order to understand where an object class resides in an image we need keep the spatial information for each pixel and assign the pixels to each class.

For this we need to use a fully convolutional network (FCN) which retains spatial information, rather than a fully connected network which does not.

An FCN can extract features with different levels of complexity and segment them into separate categories. In this project we are interested in segmenting into: 1) the target, 2) other people, and 3) the background.

We are try to predict 3 classes: 1) the target, 2) other people, and 3) the background

### Network
---
A Fully Convolutional Network (FCN) consists of three sections: 

    1) Encoders: a downsampling path which captures contextual inforamation, but loses spatial information.  
    2) 1x1 Convolution Layer: helps to reduce the dimensionality of a layer without losing information about pixel locations.  
    3) Decoders: an upsampling path which recovers lost spatial inforamtion and restores the image to it's original size.  
        - Skip connections from the downsampling path helps to combine the contextual information with spatial information.  
          upsampling doesn’t recover all the spatial information  

Encoders:
    SeparableConv2DKeras(filters=filters, kernel_size=3, strides=strides, padding='same', activation='relu')(input_layer)  
    BatchNormalization allows the network to learn fast. In addition, it limit big changes in the activation functions inside the network, i.e., there is a more smooth and solid learning in the hidden layers. 

Decoders:  
    BilinearUpSampling2D((2, 2)  
    Bilinear upsampling is a resampling technique that utilizes the weighted average of four nearest known pixels, located diagonally to a given pixel, to estimate a new pixel intensity value. The weighted average is usually distance dependent.  

**Separable Convolutional 2D with Batch Normalizxation**

    def separable_conv2d_batchnorm(input_layer, filters, strides=1):
        output_layer = SeparableConv2DKeras(filters=filters, kernel_size=3, strides=strides,
                                            padding='same', activation='relu')(input_layer)
        output_layer = layers.BatchNormalization()(output_layer)
        # output_layer = MaxPool2D(pool_size=(2, 2))(output_layer)
        return output_layer  

**Convolutional 2D with Batch Normalization**

    def conv2d_batchnorm(input_layer, filters, kernel_size=3, strides=1):
        output_layer = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                                     padding='same', activation='relu')(input_layer)
        output_layer = layers.BatchNormalization()(output_layer)
        # output_layer = layers.Dropout(0.4)(output_layer)
        return output_layer  

**Bilinear Upsampling**

    def bilinear_upsample(input_layer):
        output_layer = BilinearUpSampling2D((2, 2))(input_layer)
        return output_layer  

**Encoder Block**

    def encoder_block(input_layer, filters, strides):
        output_layer = separable_conv2d_batchnorm(input_layer, filters=filters, strides=strides)
        return output_layer  

**Decoder Block**

    def decoder_block(small_ip_layer, large_ip_layer, filters):
        upsample = bilinear_upsample(small_ip_layer)
        concat = layers.concatenate([upsample, large_ip_layer])
        c1 = separable_conv2d_batchnorm(concat, filters=filters, strides=1)
        output_layer = c1
        return output_layer  

I tried various combinations fo FCNs and hyperparameters to achieve the required final score > 0.40.  My final chosen FCN consisted of:

**Model Results**

Run   | Epochs |  LR   | Batch | Steps/Epoch | Score | PDF
:---: | :----: | :---: | :---: | :---------: | :---: | ----
1     | 20     | 0.005 | 32    | 129         | 0.19  | [Run1](/pdfs/Run1.pdf)
2     | 20     | 0.005 | 32    | 129         | 0.37  | [Run2](/pdfs/Run2.pdf)
3     | 20     | 0.005 | 32    | 129         | 0.41  | [Run3](/pdfs/Run3.pdf)
4     | 20     | 0.005 | 32    | 129         | 0.33  | [Run4](/pdfs/Run4.pdf)
:---: | :----: | :---: | :---: | :---------: | :---: | ----
5     | 20     | 0.005 | 32    | 158         | 0.42  | [Run5](/pdfs/Run5.pdf)
6     | 20     | 0.005 | 32    | 158         | 0.36  | [Run6](/pdfs/Run6.pdf)
7     | 20     | 0.005 | 32    | 158         | 0.46  | [Run7](/pdfs/Run7.pdf)
8     | 20     | 0.005 | 32    | 158         | 0.397 | [Run8](/pdfs/Run8.pdf)
:---: | :----: | :---: | :---: | :---------: | :---: | ----


    Inputs (160x16x3 Images)
    Encoder Layer 1, 32 Filters
    Encoder Layer 2, 64 Filters
    Encoder Layer 3, 128 Filters
    1x1 convolutional Layer, 256 Filters
    Decoder Layer 3, 128 Filters, Skip Connection from Encoder Layer 2
    Decoder Layer 2, 64 Filters, Skip Connection from Encoder Layer 1    
    Decoder Layer 1, 32 Filters, Skip Connection from Inputs
    Output Layer

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


### Future Enhancements
---
    1 Collect more data
    2 Pooling layers, dropout
    3 Deeper layers (more memory)
