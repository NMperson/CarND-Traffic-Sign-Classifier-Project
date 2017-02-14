#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output/hist.png "Visualization"
[image2]: ./output/1.jpg "Traffic Sign 1 - Processed"
[image4]: ./internet/1.jpg "Traffic Sign 1"
[image5]: ./internet/9.png "Traffic Sign 2"
[image6]: ./internet/20.png "Traffic Sign 3"
[image7]: ./internet/30.jpg "Traffic Sign 4"
[image8]: ./internet/40.jpg "Traffic Sign 5"
[image9]: ./output/conf.png "Confidence"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/NMperson/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the code cells under Step 1: dataset summary & exploration.

I used the shape attribute of the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the code cells inder "Include an exploratory visualization of the dataset"

First, I just played around with displaying images and cropping the images based on the coords and size matrices provided with the dataset. I realize the instructions don't necessarily say to do this, but if you're going to give me the data, I'm going to use it.

Here is an exploratory visualization of the data set. It is a histogram, showing the number of each kind of sign. Some signs have as few as 200 examples, while some have as many as 2000.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

Under step two, I first define several methods that do various preprocesses on the image. The next cell defines the method that iterates through each image in the dataset and runs the preprocessing methods. Some images throw errors when this step is run (trimming to size 0, I think) and those images are simply discareded. 

The cell after that calls the prprocessing methods on the training, validation, and test sets. 

My preprocessing method consists of four steps. First, the images are cropped to just the portion which shows the traffic sign. Second, the image is converted to greyscale. Third, the image is resized back to 32x32. Fourth, the image is normalized so that the minimum value and maximum value are -1 and 1, respectively. Finally, the image is reshaped from 2D to 3D with a singleton third dimension. This is a requirement for the later steps.

Here is an image before processing:

![alt text][image4]

Here is an image after processing:

![alt text][image2]

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The train, validation, and test sets are preprocessed, then split into their respecite images (X) and labels (y).

I do not split of a portion of the training set into a validation set, I use the validation set provided. This increases the risk of overfitting in my model.

My final training set had 34,799 images, my final test set had 12,627 images (because 3 don't complete my preprocessing successfully), and my validation set had 4,410 images.


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the "Model Architecture" portion of the ipython notebook. 

My final model consisted of the following layers:
This is almost identical to the lenet model.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 greyscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU 					|												|
| Max Pooling 			| 2x2 stride, outputs 5x5x16					|
| Flatten				| now array of 400								|
| Fully connected		| output is array of 120        				|
| RELU 					|												|
| Fully connected		| output is array of 84							|
| RELU 					|												|
| Fully connected		| output is 43, the number of signs to classify	|
| Softmax				|         										|
|						|												|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Under the section, "Train, Validate, and Test the Model", the first cell contains the code for training the model is located in the eigth cell of the ipython notebook. 


To train the model, I used an pass the input data to the leNet function to get the logits, then compare those values to the truth to get the entropy. We average the entropy across the trianing images, then minimize the loss function using the adam optimizer, which minimizes the training loss. The hyperparameters all worked fine for my in the lenet lab, so I continued to use them in this lab. Batch size is still 128, epochs is still 10, learning rate is still 0.001. 


####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in cunder under the "Analyze Performance" section of the notebook. Since I had experience with the lenet architecture, and it was developed for 32x32 greyscale images, and that was the format of this dataset as well, I thought it was a good fit for this application.

My final model results were:
* training set accuracy of 99.2%
* validation set accuracy of 94.0%
* test set accuracy of 0.907

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The main problem with all of these images is the simple fact that they do not come from the same dataset means that they are going to be altogether different than the training set, which will cause a reduction in performance. They are much less blurry than the training set (I elect not to perform any smoothing on the trianing set due to this reason), and that I beleive makes them much harder to identify. 

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| image 					|     Prediction	        	| 
|:-------------------------:|:-----------------------------:| 
| Speed limit (30km/h)      | Speed limit (30km/h)  		| 
| No Passing     			| No Passing					|
| Dangerous Curve to Right	| Dangerous Curve to Right		|
| Beware of Ice / Snow 		| Slippery Road					|
| Roundabout Mandatory		| Children Crossing      		|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This compares unfavorably to the accuracy on the test set, which acheived 90% accuracy.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the last code cell of the Ipython notebook.

For everya image except the last one, the mode is almost 100% confident that it has correctly identified the image. This is food for the first three signs, but bad for the fourth sign, which is incorrectly identifies but with high confidence.

It is 'good' that the fifth's sign's probability is lower, since it is in fact in correct, but the probability is still quite high.

| image 					|     Probability	        	| 
|:-------------------------:|:-----------------------------:| 
| Speed limit (30km/h)      | 0.999369  					| 
| No Passing     			| 1.000000						|
| Dangerous Curve to Right	| 0.998723						|
| Beware of Ice / Snow 		| 0.990996						|
| Roundabout Mandatory		| 0.8717427						|

Here is a plot of the values of the probabilities for each image:

![alt text][image9]
