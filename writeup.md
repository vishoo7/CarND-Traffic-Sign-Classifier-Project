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

[image1]: ./hist.png "Visualization"
[image2]: ./test_images/a.png "Traffic Sign 1"
[image3]: ./test_images/b.png "Traffic Sign 2"
[image4]: ./test_images/c.png "Traffic Sign 3"
[image5]: ./test_images/d.png "Traffic Sign 4"
[image6]: ./test_images/e.png "Traffic Sign 5"
[image7]: ./test_images/f.png "Traffic Sign 6"
[image8]: ./test_images/g.png "Traffic Sign 7"
[image9]: ./test_images/h.png "Traffic Sign 8"
[image10]: ./test_images/i.png "Traffic Sign 9"
[image11]: ./architecture.png "Architecture"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/vishoo7/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the "In [3]" code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. First, is a histogram showing the count for each class. Second, I display 10 random images

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What techiques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the "In [4]" code cell of the IPython notebook. I normalized the input data using Contrast-limited Adaptive Histogram Equalization (CLAHE). This can help if there's a large variation in contrast among the images. In general, normalization is more necessary when we are dealing with data with vastly different scales. Here all of our pixel values are between 0-255.

I skipped grayscaling because I don't think it would add much value (and it was mentioned as unnecessary by a udacity instructor) 

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

In cell "In [1]" I loaded the training, validation, and testing data. I did not have to split it as the zip file contained 3 separate .p files. The size of the training set and test set are noted in question 1, and the validation set is 4410.


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6 w/ RELU activation

Pooling. Input = 28x28x6. Output = 14x14x6

Layer 2: Convolutional. Output = 10x10x16 w/ RELU activation

Pooling. Input = 10x10x16. Output = 5x5x16

Flatten. Input = 5x5x16. Output = 400

Layer 3: Fully Connected. Input = 400. Output = 120 w/ RELU activation and dropout

Layer 4: Fully Connected. Input = 120. Output = 84 w/ RELU activation

Layer 5: Fully Connected. Input = 84. Output = 43

![alt text][image11]

####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Cells In[8]-In[10] contain the training model and relevant parameters. 

To train the model, I used the LeNet architecture from the CNN lesson. I added more epics (to 30) and changed the batch size (to 256). I also played around with the learning rate, but in the end reducing it did not seem to change much (or anything). In cell In[6], where I defined the LeNet architecture, the drop out rate seemed to change the accurate rate significantly. In the end I went with the suggestion that between .25 and .5 is the best place to set it.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the "In[10] cell of the Ipython notebook. The test accuracy is in the "In[11]" cell.

My final model results were:
* validation set accuracy vacilating around 0.96
* test set accuracy of 0.95

I discussed some of the tweaks and their effects in the previous answer, and continue below:

* What was the first architecture that was tried and why was it chosen?

I used the LeNet (convolutional neural network) architecture since it was the suggested architecture for this type of classification problem. It lends itself well to the problem since the sign can be positioned in different parts and sizes of the image.

* What were some problems with the initial architecture?

The validation accuracy did not reach very high. The amount of time spent minimizing the cost function was very brief.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

I went through the deep neural network lessons and recalled that drop outs are a very good solution. This helps solve the problem of overfitting. Since the validation set accuracy and the test set accuracy were high, it appears I achieved a good balance between overfitting and underfitting

* Which parameters were tuned? How were they adjusted and why?

Learning rate, drop out rate, epochs, and batch size. I was a little unsure about the ideal epoch and batch sizes. I tried to limit the epochs to a point where the learning rate wasn't moving significantly anymore. I also noticed that the batch size effected how long each epoch was take. I eventually did my processing on AWS so speed was less of a concern. I hope that more experience gives me a better intuition on these hyperparameter values. I am very curious as to what's ideal here.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

The convolution layer helps because the nature of the problem is that the details lie in smaller sections of the bigger picture. So I took the original LeNet architecture and added the drop out later, which helped for overfitting (the accuracy of the validation set increased). I hope to learn about more ways to improve the solution with changes to the architecture and parameters. Additionally, adding more image training data or appropriately mangingling the images in the training set would also improve accuracy.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web (https://s-media-cache-ak0.pinimg.com/originals/ce/55/f8/ce55f8319078dab5dbc37c51a77a837f.jpg):

![alt text][image2] ![alt text][image3] ![alt text][image4] ![alt text][image5] ![alt text][image6] ![alt text][image7] ![alt text][image8] ![alt text][image9] ![alt text][image10]

The 6th image might be difficult to classify because it is somewhat blurry, and looks rather generic (in other words, has a similar shape to many other signs). The predictor classified it as "Go straight or right" (as named in the signnames.csv file).

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the "In[29]" cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Turn right ahead      		| Turn right ahead   									| 
| Stop     			| Stop 										|
| Wild animals crossing					| Wild animals crossing											|
| Speed limit (60km/h)	      		| Speed limit (60km/h)					 				|
| Children crossing			| Children crossing      							|
| End of No passing			| End of No passing      							|
| Yield			| Yield      							|
| Priority road			| Priority road      							|
| Slippery road			| Slippery Road      							|


The model was able to correctly guess 9 of the 9 traffic signs, which gives an accuracy of 100%. This compares similarly to the accuracy on the test set of 95%

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the "In [17]" cell of the Ipython notebook.

The top probabilities for each prediction are shown below:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 30.1%      		| Turn right ahead   									| 
| 65.9%     			| Stop 										|
| ~100%					| Wild animals crossing											|
| ~100%	      		| Speed limit (60km/h)					 				|
| ~100%			| Children crossing      							|
| 60.7%			| Go straight or right      							|
| ~100%			| Yield      							|
| 95.7%			| Priority road      							|
| ~45%			| Slippery Road      							|

The remainder of the percentages can be viewed in cell "In [31]". 
