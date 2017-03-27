# **Traffic Sign Recognition** 


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done.

The code for this step is contained in the second, code cell of the IPython notebook.  

After an initial exploration I noticed that the training set was highly unbalanced. Some classes were much more represented than others. Moreover, some tests showed that images from the least represented classes were less likely to be predicted correctly. To fix these issues, I augmented the dataset with variations of the existing images, by translating, blurring, mirroring or skewing the images. I also made sure that each class has roughly the same number of images.

After this data augmentation step, 

* The size of training set is 172,000 images
* The size of test set is 43,000 images
* The shape of a traffic sign image is 32x32x3 (RGB)
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

Code cell 3 displays the first image of each class, and cell 4 displays the final distribution after the data augmentation step. It shows that the data is better balanced than initially.

### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. 

The code for this step is contained in the fifth code cell of the IPython notebook.

As a first step, I decided to convert the images from BGR (the output of the data augmentation step) to RGB to match the colors of images that will be fed in to the network during the test phase. 

As a second step, I normalized the images by subtracting 128 from each color component and dividing the resulting image by 128 so that the values are contained in a narrower interval, which clearly helps the network to reach better accuracy levels.

#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? 

The code for splitting the data into training and validation sets is contained at the end of the the first code cell

To cross validate my model, I randomly split the training data into a training set and validation set. I did this by using the test_train_split method from scikit-learn.

My final training set had 172,000 number of images. My validation set had 43,000 images

The code for augmenting the data set is contained in the augment_data.py file. I decided to generate additional data because the original data set was unbalanced and did not contain enough images to train the network to accurately detect certain classes. To add more data to the the data set, I generated new images from the original ones by blurring, translating, skewing and mirroring using OpenCV.


#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.)

The code for my final model is located in cells 8, 9 and 10 of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|---------------------|---------------------------------------------| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x18 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x18 				|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 10x10x36 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x36				|
| Flatten | Outputs 900 |
| Fully connected		| Outputs 450        									|
| RELU | |
| Dropout | Probability : .65 |
| Fully connected		| Outputs 86        									|
| RELU | |
| Dropout | Probability : .65 |
| Fully connected		| Outputs 43        									|
| RELU | |
| Dropout | Probability : 1 |

 


#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in cells 11, 12 and 13 of the ipython notebook. 

To train the model, I used an Adam Optimizer with a batch size of 90 and 100 epochs. 

The best learning rate is very low at .00004. Setting the learning rate higher leads to low and decreasing accuracy to a value close to zero. Setting it lower makes the network very slow to learn.

#### 5. Describe the approach taken for finding a solution. 

The initial approach was to start with the LeNet architecture as is. The network initially did not learn anything (low and decreasing accuracy), until I found a good learning rate. 

I also decreased the batch size to 90. It slightly improved the accuracy.

I then changed the depth of the convolutional and the size of the fully connected layers, which led to a better accuracy. I also tweaked the dropout probabilities, and found that .65 worked well, mostly using a "trial and error" approach.

The final accuracy was 96.2% 

### Test a Model on New Images

#### 1. Choose five traffic signs, discuss what quality or qualities might be difficult to classify.

The 5 traffic signs are displayed in the IPython Notebook. These images were taken from home made video taken on Belgian roads. To test the model, I specifically chose Belgian signs which were similar to German ones, and which could be found in the training data set.

The first  image should be fairly easy to classify, because it is very sharp and clearly distinguishable from other types of traffic signs.

The second and fifth images, are good quality but with a higher level of detail, which may make them a little bit more difficult to classify.

The fourth image is clear and sharp, but there are many other similar traffic signs (other speed limit signs) in the training set, which may be confusing. 

The third image was taken in the dark and is somewhat blurry, it should be the hardest to classify.

#### 2. Discuss the model's predictions on these new traffic signs 
The code for making predictions on my final model is located in the Ipython notebook under the "Predict the sign type for each image" title

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This is slightly lower than the 96.2% achieved on the validation data set, but with only 5 images.

On the 5 images, only the darkest and blurriest one was not predicted accurately, and the classifier ranked the correct class as 4th. Even the class with the highest probability only has a 16% probability. 
For all other signs, the correct class was detected accurately with higher levels of confidence. 

For the first image, the model is relatively sure that it has the correct sign type (69.31%). The next probabilities are much lower (33.73% and less).

For the second image, the network still has a stronger preference for its first prediction (39.46% for the first vs. 24.45% for the second) but is less certain than for the first image.

The third and hardest to predict, was misclassified, but with a low confidence level (16.18%). The model fails at classifiying it correctly, but it does not have much confidence in its prediction either. The correct sign type is ranked fourth in the top 5 with only 8.02%.

The fourth image was correctly classified but with a low confidence level. Probably because there are other similar speed limit signs in the training set.
The second and third sign types ranked in the top 5 are other speed limit signs.

The last image was ranked with the most confidence of all (81.38%) with a very clear preference for the first choice. The second sign type of the top 5 only has a 40.61% confidence.

All the top 5 are displayed as bar charts in the IPython Notebook.
