# Lane-Detection-using-Deep-Learning
This repository contains codes and models to detect Lanes on the roads

Download Model: https://drive.google.com/file/d/1tvgvQE1Twa-OO-XTjruifImnQTt16WiX/view?usp=sharing 

Earlier I used to think, Lane detection is an easy go; You got the dataset, you train an image segmentation network such as FCN32/16/8, and you got the predictions. But this problem had me, it consumed me for 2 days! After solving it, I experienced how challenging this was. Now let’s see what I have got.

Pardon me for a long document, I am an author by nature, But seriously, I have enjoyed a lot while solving the problem, and I can not resist sharing my experience with you. 

# Approach

Approach was simple. Train an Image segmentation network on the given data set. And then comes the crucial part. Which network one should choose? Well I have some experience in image segmentation before, So my first choice was a vgg16-FCN. I downloaded the data set, resized images to 384X640 and created training and testing splits and put it all together on my G-drive. Next I have Created a runtime on google Colab and trained a network. I tested it on training data first and the result was really good. With full confidence I started testing on test Images, the first Image has run and BOOM! Blank screen with tiny white dots in front of my eyes. Failed. My classifier got a curse of overfitting! 

When the trained classifier failed, seriously, I lost hope in humanity. Then the real game started, I thought the classifier is not capable enough to solve this problem. I replaced the vgg16-FCN with a custom vgg16-unet. But again failed on test data. I was broke. 

# Challenges

After two failed attempts I sat with the most friendly guy of a data scientist; Data exploration! I started digging the data set of those 500 images for the first time. And you know what I got.

Lots of out of focus images. Blur.
Lots of images with very small labelled areas, that too is not correct in many places.
Incomplete labels. 
Lots of viewing angle variations (Shear, slight rotations, etc)
Illumination changes due to day and night time.

These are some of the challenges I have noticed at the first glance. Next second I have a thought of giving up. But my inner self started taunting me for this early give-up.

In the next few minutes, I was having the corner with an evil nightmare of a data scientist; Data cleaning!

# Data Cleaning

So I need to remove lots of images from the given data set. 

- I have removed images which were out of focus. 
- Some images with incomplete label information.
- There were some images which did not have lanes or having very small regions, which creates a lot of class imbalance. So I removed that too.   
- In some images there are a lot of portions from the dashboard that are visible more as compared to road, so I have removed that too.

So we are good now. I have again trained a classifier; wait two classifiers. And both of them failed me again. This is it!. I quit.

In deep thoughts of frustration, I was hovering my mouse pointer over the screen and suddenly stopped when it was on the folder of cleaned images. It shows 393 items inside. Well 393 for training a deep neural network! Are you joking?

# Data Augmentation

So the count was 393, that too without train and test split, If I will select only 10% for the testing, I will be getting only 353 images for training a Mammoth neural network. Here my best bet is to go for data augmentation.

I have chosen ImageDataGenerator from keras to do this task for me. I have created two data generators one for augmenting the images and another for augmenting the binary labels masks. I did initialize both the data generators with the same seed value so generated batches will have the same image-mask pair. Following are the transformations I chose for augmentation.

- Image re-scaling between 0 and 1, by dividing pixel values with 255.
- Image shearing with 30%.
- Set zoom range of 30%.
- 90 degree rotation range (who knows if somebody wants to drive the car on two wheels!).
- Width shift range and height shift range of 30%.
- And the horizontal flip.

These augmentations can easily take care of viewing angles and other affine transformations,  For various perspectives. These 5 affine transformations will increase our data set significantly.

# Networks

So whenever you try to solve a deep learning based problem, your obvious choice is always amongst the famous architecture like VGG-Nets,  ResNets, and Ineptions for image classification; similarly for image segmentation people always chooses something from, VGG-Net-FCNs, U-Nets, SegNet, E-Net etc. I have chosen something else!

Well I am a big fan of VGG net, It was the first network I have ever tried to implement, what a beauty written by Karen Simonyan & Andrew Zisserman (Respect!). Well It is a very simple yet one of the most powerful  deep neural networks you will ever see. No fancy layers, no branches, straight forward sequential network. One of the best networks to extract image features. So I have chosen VGG as the backend of my architecture. While VGG is a very good network for image classification tasks, Its siblings VGG-FCN are not that good for precise segmentation tasks. You will always get dilated kind of segmentation maps at the output. So how can we extract rich features and do a precise segmentation?

 I am a guy who has a rich background in medical image processing, mostly histological color images. And I used to work on various kinds of image segmentation tasks, and when it's medical image segmentation the best choice is always a U-Net.

# Architecture

For Image segmentation, an Encoder-Decoder network is always preferred, I chose VGG-16 as the Encoder part of the architecture and while decoding I have used skip connections from encoder layers similar to the U-Net architecture. These skip connections help our network to learn precise segmentation maps.  

So there are 13 convolution layers with relu activations along with 5 max pooling layers present in the encoder end, and in decoder end, 5 bilinear up sampling layers are concatenated with skip connections running from convolution layers from encoder end. After each concatenation, a pair of convolution layers present. Note that one can use  transposed convolution instead of bilinear convolutions. At the final layer we have a convolution layer with sigmoid activation. I have chosen sigmoid to use binary cross-entropy as a loss function.

# Performance

This time when I ran my classifier on the test data set, I crossed my fingers and hit the key. Well hard work always pays off! It ran! Successfully segmenting lanes in the test data set. Hurray! I deserve the cheer!
 
To measure the performance of segmentation algorithms, pixel based accuracy is the worst choice one can make. I have used the mean-IOU and Dice coefficient to measure the performance of the network.  I have got a dice score of 0.45 (on test data) which was 0.20 in case of simple VGG-Net and It was 0.05 without data augmentation. I know 0.45 is still to be improved. The dice score calculated after threshold the sigmoid output with 0.2. 

To improve the  network performance further we can use some Batch normalization layer in the architecture and can train the network for the longer duration. But for the time being I think I have got some results for you.

Images | Ground Truth | Predition
------------ | ------------- | ------------- 
![image_1](/images/1.png) | ![image_1](/images/1_gt.png)| ![image_1](/images/1_out.png)
Content in the first column | Content in the second column |













 


