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

I tried various combinations of FCNs with increasingly deeper layers to achieve the required final score > 0.40.  
**
      Model 1        Model 2        Model 3        Model 4         Model 5  
    ------------   ------------   ------------   -------------   -------------
    Inputs         Inputs         Inputs         Inputs          Inputs
    Encoder(32)    Encoder(32)    Encoder(32)    Encoder(32)     Encoder(32)
    1x1Conv(64)    Encoder(64)    Encoder(64)    Encoder(64)     Encoder(64)
    Decoder(32)    1x1Conv(128)   Encoder(128)   Encoder(128)    Encoder(128)
    Outputs        Decoder(64)    1x1Conv(256)   Encoder(256)    Encoder(256)
                   Decoder(32)    Decoder(128)   1x1Conv(512)    Encoder(512)
                   Outputs        Decoder(64)    Decoder(256)    1x1Conv(1024)
                                  Decoder(32)    Decoder(128)    Decoder(512)
                                  Outputs        Decoder(64)     Decoder(256)
                                                 Decoder(32)     Decoder(128)
                                                 Outputs         Decoder(64)
                                                                 Decoder(32)
                                                                 Outputs
    ------------   ------------   ------------   -------------   -------------                                                               Note: Number of filters in ()

For my model runs I used the original training and validation data.  The hyperparameters and model results for each run were:

### Hyperparameters
---
learning_rate = 0.005   
batch_size = 32         
num_epochs = 20         
steps_per_epoch = 129   # 4131 images // batch_size = 129
validation_steps = 42   # 1184 images // batch_size = 42
workers = 2             

**Model Results**                    
---
**Using just original Training and Validation Images**

Model | Epochs |  LR   | Batch | Steps/Epoch | Score  | PDF
:---: | :----: | :---: | :---: | :---------: | :---:  | ----
1     | 20     | 0.005 | 32    | 129         | 0.202  | [Run1](/pdfs/Run1.pdf)
2     | 20     | 0.005 | 32    | 129         | 0.360  | [Run2](/pdfs/Run2.pdf)
3     | 20     | 0.005 | 32    | 129         | 0.399  | [Run3](/pdfs/Run3.pdf)
4     | 20     | 0.005 | 32    | 129         | 0.381  | [Run4](/pdfs/Run4.pdf)
5     | 20     | 0.005 | 32    | 129         | 0.393  | [Run5](/pdfs/Run4.pdf)

I did not get to the 0.40 required score with any of these runs but model3 was close.  For my next set of runs I chose to augment the data by flipping each image.  This doubled the number of training and validation images and helped to balance out some biases in the image poses.

I kept the hyperparameters the same except I increased the steps_per_epoch and validation steps to account for twice as many images.

### Hyperparameters
---
learning_rate = 0.005   
batch_size = 32         
num_epochs = 20         
steps_per_epoch = 259   # 8262 images // batch_size = 259
validation_steps = 84   # 2368 images // batch_size = 84
workers = 2             

**Using Original + Flipped Training and Validation Images**

:---: | :----: | :---: | :---: | :---------: | :---:  | ----
1     | 20     | 0.005 | 32    | 258         | 0.226  | [Run6](/pdfs/Run5.pdf)
2     | 20     | 0.005 | 32    | 258         | 0.366  | [Run7](/pdfs/Run6.pdf)
3     | 20     | 0.005 | 32    | 258         | 0.421  | [Run8](/pdfs/Run7.pdf)
4     | 20     | 0.005 | 32    | 258         | 0.356  | [Run9](/pdfs/Run8.pdf)
5     | 20     | 0.005 | 32    | 258         | 0.417  | [Run10](/pdfs/Run8.pdf)
:---: | :----: | :---: | :---: | :---------: | :---:  | ----

Model 3 again was the best performer and achieved a score of 0.421.

My final chosen FCN consisted of:

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


### Results
---

**Hyperparameters: LR=0.005, Batch=32, Epochs=20, Epoch Steps=138**



### Future Enhancements
---
    1 Collect more data
    2 Pooling layers, dropout
    3 Deeper layers (more memory)
