# Udacity RoboND - Project 4: Deep Learning - Follow Me

In this project We are trying to locate a target ("what") in an image and determine "where" in the image the target is located.  A typical classification model only needs to understand what is in an image and does not retain pixel spatial information.  However, in order to understand where an object class resides in an image we need keep the spatial information for each pixel and assign the pixels to each class.  For this we need to use a fully convolutional network (FCN) which retains spatial information, rather than a fully connected network which does not.

An FCN can extract features with different levels of complexity and segment them into separate categories. In this project we are interested in segmenting the pixels into three classses: 

  1) the "target" person  
  2) other people  
  3) the background  

### A: Fully Convolutional Network Defined   
---
A Fully Convolutional Network (FCN) consists of three sections: 

    Encoders: 
    
        a downsampling path which captures contextual information, but loses spatial information.  
        
    1x1 Convolution Layer: 
    
        helps to reduce the dimensionality of a layer without losing information about pixel locations.  
        
    Decoders: 
    
        an upsampling path which recovers lost spatial inforamtion and restores the image to it's original size.  

The single encoder block (layer) consists of a separable convolutional 2D layer along with batch normalization.

    def encoder_block(input_layer, filters, strides):
        output_layer = separable_conv2d_batchnorm(input_layer, filters=filters, strides=strides)
        return output_layer

Each encoder layer allows the model to gain a better understanding of the shapes in the image at the expense of losing spatial information.

**Separable Convolutional 2D with Batch Normalizxation**

Separable convolution layers are a convolution technique for increasing model performance by reducing the number of parameters in each convolution. This is achieved by performing a spatial convolution while keeping the channels separate, followed with a depthwise convolution. Instead of traversing the each input channel by each output channel and kernel, separable convolutions traverse the input channels with only the kernel, then traverse each of those feature maps with a 1x1 convolution for each output layer, before adding the two together. This technique allows for the efficient use of parameters.

    def separable_conv2d_batchnorm(input_layer, filters, strides=1):
        output_layer = SeparableConv2DKeras(filters=filters,kernel_size=3, strides=strides,
                                            padding='same', activation='relu')(input_layer)
        output_layer = layers.BatchNormalization()(output_layer) 
        return 

Batch normalization allows the network to learn more quickly by limiting large changes in the activation functions inside the network.

**Convolutional 2D with Batch Normalization**

With a 1x1 convolution layer, the data is flattened while still retaining spatial information from the encoder. An additional benefit of 1x1 convolutions is that they are an efficient approach for adding extra depth to the model.

    def conv2d_batchnorm(input_layer, filters, kernel_size=3, strides=1):
        output_layer = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                                     padding='same', activation='relu')(input_layer)
        output_layer = layers.BatchNormalization()(output_layer)
        return output_layer  

The decoder block is comprised of three parts:  

1) A bilinear upsampling layer.  
2) A layer to concatenate the upsampled small_ip_layer and the large_ip_layer.   
3) Some (one or two) additional separable convolution layers to extract some more spatial information from prior layers.   

The decoder block calculates the separable convolution layer of the concatenated bilinear upsample of the smaller input layer with the larger input layer. This structure mimics the use of skip connections by having the larger decoder block input layer act as the skip connection.  Skip connections allow the network to retain spatial information from prior layers that were lost in subsequent convolution layers. Skip layers use the output of one layer as the input to another layer. By using information from multiple image sizes, the model is able to make more precise segmentation decisions

    def decoder_block(small_ip_layer, large_ip_layer, filters):
        upsample = bilinear_upsample(small_ip_layer)
        concat = layers.concatenate([upsample, large_ip_layer])
        output_layer = separable_conv2d_batchnorm(concat, filters=filters, strides=1)
        return output_layer  

Each decoder layer is able to reconstruct a little bit more spatial resolution from the layer before it. The final decoder layer will output a layer the same size as the original model input image, which will be used for guiding the quadcopter.

**Bilinear Upsampling**

Bilinear upsampling uses the weighted average of the four nearest known pixels from the given pixel, estimating the new pixel intensity value. Although bilinear upsampling loses some finer details when compared to transposed convolutions, it has much better performance, which is important for training large models quickly.

    def bilinear_upsample(input_layer):
        output_layer = BilinearUpSampling2D((2, 2))(input_layer)
        return output_layer  


### B: Model Evaluation
---
**Scoring:**  

To evaluate how well the model has performed the metric Intersection over Union (IoU) is calculated.  IoU measures how much (in precent of pixels) the ground truth image overlaps with the segmented image from the FCN model.  

    Intersection over Union (IoU) = Area of Overlap / Area of Union    
    

### c: Model Output  
---
I tried various combinations of FCNs with increasingly deeper layers to achieve the required final score > 0.40.  

      Model 1        Model 2        Model 3        Model 4         Model 5  
      ------------   ------------   ------------   -------------   -------------  
      Inputs         Inputs         Inputs         Inputs          Inputs  
      Encoder(16)    Encoder(16)    Encoder(16)    Encoder(16)     Encoder(16)  
                     Encoder(32)    Encoder(32)    Encoder(32)     Encoder(32)  
                                    Encoder(64)    Encoder(64)     Encoder(64)  
                                                   Encoder(128)    Encoder(128)  
                                                                   Encoder(256)  
      1x1Conv(32)    1x1Conv(64)    1x1Conv(128)   1x1Conv(256)    1x1Conv(512)  
                                                                   Decoder(256)  
                                                   Decoder(128)    Decoder(128)  
                                    Decoder(64)    Decoder(64)     Decoder(64)  
                     Decoder(32)    Decoder(32)    Decoder(32)     Decoder(32)  
      Decoder(16)    Decoder(16)    Decoder(16)    Decoder(16)     Decoder(16)  
      Outputs        Outputs        Outputs        Outputs         Outputs  
      ------------   ------------   ------------   -------------   -------------  
      Note: Number of filters in ()  

For my model runs I used the original training and validation data provided.  

The hyperparameters and model results for each run were:  

#### Hyperparameters  
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

Model | Epochs |  LR   | Batch | Steps/Epoch | Score  |  
:---: | :----: | :---: | :---: | :---------: | :---:  |  
1     | 20     | 0.005 | 32    | 129         | 0.202  |  
2     | 20     | 0.005 | 32    | 129         | 0.360  |  
3     | 20     | 0.005 | 32    | 129         | 0.399  |  
4     | 20     | 0.005 | 32    | 129         | 0.381  |  
5     | 20     | 0.005 | 32    | 129         | 0.393  |  

I did not get to the 0.40 required score with any of these runs but model3 was close.  For my next set of runs I chose to augment the data by flipping each image.  This doubled the number of training and validation images and helped to balance out some biases in the image poses.

I kept the hyperparameters the same except I increased the steps_per_epoch and validation steps to account for twice as many images.

#### Hyperparameters  
---
learning_rate = 0.005   
batch_size = 32         
num_epochs = 20         
steps_per_epoch = 259   # 8262 images // batch_size = 259  
validation_steps = 84   # 2368 images // batch_size = 84  
workers = 2             

**Using Original + Flipped Training and Validation Images**  

Model | Epochs |  LR   | Batch | Steps/Epoch | Score  |  
:---: | :----: | :---: | :---: | :---------: | :---:  |  
1     | 20     | 0.005 | 32    | 258         | 0.226  |  
2     | 20     | 0.005 | 32    | 258         | 0.366  |  
3     | 20     | 0.005 | 32    | 258         | 0.421  |  
4     | 20     | 0.005 | 32    | 258         | 0.356  |  
5     | 20     | 0.005 | 32    | 258         | 0.417  |  
:---: | :----: | :---: | :---: | :---------: | :---:  |  

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

### D: Future Enhancements  
---
    1 Collect more data
    2 Pooling layers, dropout
    3 Deeper layers (more memory)
