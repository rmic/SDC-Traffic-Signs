# Load pickled data
import pickle
import numpy as np

import cv2
import random
import matplotlib

from sklearn.utils import shuffle

import matplotlib.pyplot as plt
from math import *
import sys
training_file = "./datasets/train.p"
x_training_file = "./datasets/x_train_augmented_test.p"
y_training_file = "./datasets/y_train_augmented_test.p"
validation_file = "./datasets/valid.p"
testing_file = "./datasets/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
#with open(x_training_file, mode='rb') as f:
#    X_train = pickle.load(f)

#with open(y_training_file, mode='rb') as f:
#    y_train = pickle.load(f)


X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']



### Replace each question mark with the appropriate value.
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = X_train.shape[0]
n_valid = X_valid.shape[0]
# TODO: Number of testing examples.
n_test = X_test.shape[0]

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
unique_labels = np.unique(y_train)
n_classes = len(unique_labels)

print("Number of training examples =", n_train)
print("Number of validation examples =", n_valid)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)




### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.


# Visualizations will be shown in the notebook.
# %matplotlib inline
# First, let's check what the images in the data set look like ...
i = 1
for label in unique_labels:
        # Pick the first image for each label.
        index = np.where(y_train == label)[0][0]
        image = X_train[index]
        #plt.subplot(7, 7, i)
        #plt.axis('off')
        i += 1
        #_ = plt.imshow(image)
#plt.show()


# Let's check the distribution of training data in terms of classes

x= np.arange(0,n_classes)
plt.hist(y_train, bins=n_classes,rwidth=0.85)
plt.show()




# Mirrors (flips) an image, vertically and/or horizontally
def mirror(img, vertical=True):
    out = img.copy()
    if vertical:
        out = cv2.flip(out, 1)

    return out


# Test
def test_transform(imgId, transformation, *trans_args):
    img = X_train[imgId]
    plt.subplot(1, 2, 1)
    plt.axis('off')
    plt.title("Original")
    _ = plt.imshow(img)
    newImg = transformation(img, trans_args)
    plt.subplot(1, 2, 2)
    plt.axis('off')
    plt.title("Transformed")
    _ = plt.imshow(newImg)

    plt.show()


# test_transform(6789, mirror)


# distorts the image by a random (but bounded) number of pixels
# in multiple directions
def skew(img, args):
    rows, cols, ch = img.shape
    x1 = random.randint(1, 10)
    y1 = random.randint(5, 10)
    x2 = random.randint(25, 30)
    y2 = y1
    x3 = y2
    y3 = x2
    x4 = random.randint(1, 5)
    y4 = random.randint(1, 5)
    x5 = random.randint(20, 30)
    y5 = random.randint(1, 10)

    pts1 = np.float32([[y1, x1], [y2, x2], [y3, x3]])
    pts2 = np.float32([[x4, y4], [x3, y3], [x5, y5]])
    M = cv2.getAffineTransform(pts1, pts2)
    return cv2.warpAffine(img, M, (cols, rows))


#test_transform(6789, skew)


# Rotates image by a random but bounded number of degrees
# around a random point close to the center. (max 5px away in each direction)
def rotate(img, args):
    center = (np.array(image.shape[:2]) / 2)
    center[0] += random.randint(-5, 5)
    center[1] += random.randint(-5, 5)
    center = tuple(center)
    angle = random.randint(-25, 25)
    rot = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    return cv2.warpAffine(img, rot, image.shape[:2], flags=cv2.INTER_LINEAR)


#test_transform(6789, rotate)


# Returns an image on which a gaussian blur filter has been applied
def blur(img, args):
    r = 3
    return cv2.GaussianBlur(img,ksize=(r,r),sigmaX=0)

# test_transform(6789, blur)


# Moves the image
def translation(img, args):
    dx = random.randint(-5,5)
    dy = random.randint(-5,5)
    num_rows, num_cols = img.shape[:2]
    trans_matrix = np.float32([ [1,0,dx], [0,1,dy] ])
    return cv2.warpAffine(img, trans_matrix, (num_cols, num_rows))

# test_transform(6789, translation)


mirrorable = [11, 12, 13, 15, 17, 18, 22, 26, 27, 35]
transforms = [translation, blur, rotate, skew, mirror]
images_in_class = 5000


### This has been executed once and the data is stored in the x_train_augmented and y_train_augmented files

# How many images do we want in each class and how many are missing ?

#classes, counts = np.unique(y_train, return_counts=True)
#for clazz, count in zip(classes, counts):
#    print(str(clazz) + " has " + str(count) + " images, " + str(images_in_class - count) + " to be generated.")
#    y_indices = np.where(y_train == clazz)
#    z = 1 + len(np.where(mirrorable == clazz)[0])
    #for i in range((images_in_class - count)):
        #j = random.randint(0, len(y_indices[0]) - 1)
        #k = y_indices[0][j]
        #img = X_train[k]
        #func = transforms[random.randint(0, len(transforms) - z)]
        #newImg = cv2.cvtColor(func(img, None), cv2.COLOR_BGR2RGB)
        #cv2.imwrite("datasets/augmentation/" + str(clazz) + "/" + str(i) + ".png", newImg)
        #img = cv2.imread("datasets/augmentation/" + str(clazz) + "/" + str(i) + ".png")
        #X_train = np.append(X_train, img)
        #y_train = np.append(y_train, clazz)
        #if(i%50 == 0):
        #    print(i)

    #pickle.dump(X_train, open("x_train_augmented_"+str(clazz)+".p", "wb"))



# PRE PROCESSING

def pre_process(img):
    return ((img - [128.0, 128.0, 128.0]) / 128.0)

X_train = np.array(list(map(pre_process, X_train)))
X_valid = np.array(list(map(pre_process, X_valid)))

# MODEL ARCHITECTURE

import tensorflow as tf


EPOCHS = 100
BATCH_SIZE = 250

from tensorflow.contrib.layers import flatten
import tfhelper

strides = {  "l1": 1, "p1": 2, 'l2': 1, 'p2': 2, 'l3': 1, 'l4':1, "out": 1}

dropout = { "l3": .75,  "l4": .75, 'out': 0.75 }


shapes = {
    'input': [32,32,3],
    'l1': [28,28,6],
    'p1': [14,14,6],
    'l2': [10,10,16],
    'p2': [5,5,16],
    'flat': [400],
    'l3': [86],
    'l4': [43],
    'out': [43]
}

pipeline = ['input', 'l1','p1',  'l2', 'p2', 'flat', 'l3', 'l4', 'out']


tfh = tfhelper.TFHelper(strides, dropout, shapes, pipeline)



def LeNet(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    # Layer 1: Convolutional.
    l1 = tfh.convLayer(x, 'l1')
    print(l1)
    # Pooling.
    p1 = tfh.poolLayer(l1, 'p1')
    print(p1)
    # Layer 2: Convolutional.
    l2 = tfh.convLayer(p1, 'l2')
    print(l2)
    # Pooling.
    p2 = tfh.poolLayer(l2, 'p2')
    print(p2)
    # Flatten.
    flat = flatten(p2)
    print(flat)

    # Layer 3: Fully Connected.

    l3 = tfh.fullLayer(flat, 'l3')
    print(l3)
    # Layer 4: Fully Connected.

    l4 = tfh.fullLayer(l3, 'l4')
    print(l4)
    # Layer 5: Fully Connected.
    logits = tfh.fullLayer(l4, 'out')
    print(logits)
    return logits


# Feature And Labels

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)

# Training pipeline
rate = 0.005

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

def update_progress(progress):
    print('\r %.2f %%' % (progress*100))

# Model evaluation

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

def update_progress(progress):
    sys.stdout.write('\r[{0}{1}] {2}%'.format('#'*ceil(progress/10), ' '*(10-ceil(progress/10)), progress))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(EPOCHS):
        #print("EPOCH "+str(i))
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            update_progress(end * 100 / num_examples)


        print("EPOCH {} ...".format(i + 1))
        validation_accuracy = evaluate(X_valid, y_valid)
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()