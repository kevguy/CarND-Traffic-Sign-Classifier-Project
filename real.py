# Step 0: Load the Data

# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = 'train.p'
testing_file = 'test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']
print(X_train)

# Step 1: Dataset Summary & Exploration
### Replace each question mark with the appropriate value.

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(set(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

### Data exploration visualization goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
import cv2
import random
import numpy as np
from skimage import exposure
from skimage.morphology import disk
from skimage.filters import rank

'''
# Visualizations will be shown in the notebook.
#   %matplotlib inline
for i in range(1):
    index = random.randint(0, len(X_train))
    image = X_train[index].squeeze()
    #plt.figure()
    plt.imshow(image)
    #plt.show()


### Draw graph showing number of images for different classes
# y axis
y_axis = np.zeros(n_classes)
for label in y_train:
    y_axis[label]+=1

x_axis = np.arange(n_classes)

plt.title('Graph showing number of images for different classes')
plt.xlabel('Label')
plt.ylabel('Count')
plt.bar(x_axis, y_axis)
#plt.show()
'''

# Step 2: Design and Test a Model Architecture
# Pre-processing:
# 1. All images are down-sampled or upsampled to 32x32
#    (dataset samples sizes vary from 15x15 to 250x250)
# 2. Images are converted to YUV space
# 3. The Y channel is then preprocessed with global and 
#    local contrast normalization while U and V channels 
#    are left unchanged.
# Global normalization first centers each image around its mean value,
# whereas local normalization (see II-B) emphasizes edges.

# The first step is not necessary
# All the images we got are in 32x32x3

def _ycc(r, g, b): # in (0,255) range
    y = .299*r + .587*g + .114*b
    cb = 128 -.168736*r -.331364*g + .5*b
    cr = 128 +.5*r - .418688*g - .081312*b
    return y, cb, cr

def _rgb(y, cb, cr):
    r = y + 1.402 * (cr-128)
    g = y - .34414 * (cb-128) -  .71414 * (cr-128)
    b = y + 1.772 * (cb-128)
    return r, g, b

def RGB2YUV2(img_in):
    img_out = img_in
    img_out[:,:,0] = .299*img_in[:,:,0] + .587*img_in[:,:,1] + .114*img_in[:,:,2]
    img_out[:,:,1] = 128 -.168736*img_in[:,:,0] -.331364*img_in[:,:,1] + .5*img_in[:,:,2]
    img_out[:,:,2] = 128 +.5*img_in[:,:,0] - .418688*img_in[:,:,1] - .081312*img_in[:,:,2]
    return img_out

def YUV2RGB2(img_in):
    img_out = img_in
    img_out[:,:,0] = img_in[:,:,0] + 1.402 * (img_in[:,:,2]-128)
    img_out[:,:,1] = img_in[:,:,0] - .34414 * (img_in[:,:,1]-128) -  .71414 * (img_in[:,:,2]-128)
    img_out[:,:,2] = img_in[:,:,0] + 1.772 * (img_in[:,:,2]-128)
    return img_out

def RGB2YUV(img_in):
    return cv2.cvtColor(img_in, cv2.COLOR_BGR2YUV, 3)

def YUV2RGB(img_in):
    return cv2.cvtColor(img_in, cv2.COLOR_YUV2RGB, 3)

def grayscaleImage(img_in):
    gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    return gray

def LocalConstrastNormalization(img):
    #clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
    #img_clahe = clahe.apply(img)
    #return img_clahe
    selem = disk(5)
    img_eq = rank.equalize(img, selem=selem)
    return img_eq

def GlobalConstrastNormalize(img_in):
    return exposure.equalize_hist(img_in)

def preprocess_image(img_in):
    #yuv_img = RGB2YUV2(img_in)
    # global_constrast_img = yuv_img
    # global_constrast_img[:,:,0] = GlobalConstrastNormalize(global_constrast_img)[:,:,0]
    # local_constrast_img = yuv_img
    # local_constrast_img[:,:,0] = GlobalConstrastNormalize(local_constrast_img)[:,:,0]
    grayscale_img = grayscaleImage(img_in)
    zeros = np.zeros((32,32,3))
    norm_image = cv2.normalize(img_in, zeros, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return norm_image

def preprocess_image_test(train_input):
    n_rows = 2
    n_cols = 6
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 6))
    fig.subplots_adjust(hspace=.1, wspace=.001)
    axs = axs.ravel()
    for i in range(n_cols):
        print(i)
        index = random.randint(0, len(train_input))
        image = train_input[index].squeeze()
        axs[i].axis('off')
        axs[i].imshow(image)
        new_img = preprocess_image(image)
        axs[i+n_cols].axis('off')
        axs[i+n_cols].imshow(new_img, cmap="Greys_r")
    plt.show()
#preprocess_image_test(X_train)

def preprocess_images(train_input):
    new_train_input = []
    for index in range(len(train_input)):
        new_train_input.append(preprocess_image(train_input[index]))
        # plt.imshow(new_train_input[index])
        # plt.show()
        if (index%20 == 0):
            print(index)
    return new_train_input

def preprocess_images_test(train_input):
    train_input = preprocess_images(train_input)
    for i in range(10):
        index = random.randint(0, len(train_input))
        image = train_input[index].squeeze()
        plt.figure(figsize=(1,1))
        plt.imshow(image, cmap="Greys_r")
        plt.show()

#preprocess_images_test(X_train)

X_train_normalized = preprocess_images(X_train)
X_test_normalized = preprocess_images(X_test)





# Shuffle the data
from sklearn.utils import shuffle
X_train_normalized, y_train = shuffle(X_train_normalized, y_train)

### Generate additional data (OPTIONAL!)
### and split the data into training/validation/testing sets here.
### Feel free to use as many code cells as needed.
from sklearn.model_selection import train_test_split

X_train_normalized,X_validation,y_train,y_validation = train_test_split(X_train_normalized,y_train,test_size=0.4,random_state=0)


# The model
from tensorflow.contrib.layers import flatten

def LeNet(x):    
    # Hyperparameters
    mu = 0
    sigma = 0.1
    
    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mu, stddev = sigma)) # (height, width, input_depth, output_depth)
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b # (batch, height, width, depth)

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)
    
    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # SOLUTION: Activation.
    fc2    = tf.nn.relu(fc2)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits

'''
def LeNet(x):    
    # Hyperparameters
    mu = 0
    sigma = 0.1
    
    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)
    
    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # SOLUTION: Activation and dropout.
    fc1    = tf.nn.relu(fc1)
    fc1  = tf.nn.dropout(fc1, keep_prob)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # SOLUTION: Activation and dropout
    fc2    = tf.nn.relu(fc2)
    fc2    = tf.nn.dropout(fc2, keep_prob)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits
'''
# Train
import tensorflow as tf
### Train your model here.
### Feel free to use as many code cells as needed.
#x is a placeholder for a batch of input images. y is a placeholder for a batch of output labels.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)
one_hot_y = tf.one_hot(y, 43)



#Create a training pipeline that uses the model to classify data.
rate = 0.001
EPOCHS=20
BATCH_SIZE=128
save_file = './model.ckpt'

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)
test_prediction = tf.argmax(logits, 1)
test_softmax = tf.nn.softmax(logits)


#Evaluate how well the loss and accuracy of the model for a given dataset.
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

'''
#Run the training data through the training pipeline to train the model.
#Before each epoch, shuffle the training set.
#After each epoch, measure the loss and accuracy of the validation set.
#Save the model after training.
with tf.Session() as sess:
    print('Session starts')
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train_normalized)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train_normalized, y_train = shuffle(X_train_normalized, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train_normalized[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
            
        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, save_file)
    print("Model saved")
'''
# Do evaluation
with tf.Session() as sess:
    #saver = tf.train.import_meta_graph('model.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test_normalized, y_test)

    prediction=test_accuracy * len(X_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))


# Step 3: Test a Model on New Images
# Reinitialize and re-import if starting a new kernel here
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # mpimg 
from PIL import Image
import os
# %matplotlib inline

import tensorflow as tf
import numpy as np
import cv2


### Load the images and plot them here.
### Feel free to use as many code cells as needed.
fig, axs = plt.subplots(1,5, figsize=(4, 2))
fig.subplots_adjust(hspace = .2, wspace=.001)
axs = axs.ravel()
new_images = []

n_rows = 2
n_cols = 10
fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 6))
fig.subplots_adjust(hspace=.1, wspace=.001)
axs = axs.ravel()






for i in range(n_cols):
    file_name = os.path.join(os.path.abspath('.'), 'test_images', str(i+1) + '.jpg')
    #image = np.asarray(Image.open(file_name).convert('RGB'))
    image = cv2.imread(file_name)

    #image.setflags(write=1)
    axs[i].axis('off')
    axs[i].imshow(image)
    # new_img = preprocess_image(image)
    # axs[i+n_cols].axis('off')
    # axs[i+n_cols].imshow(new_img, cmap="Greys_r")
    # new_images.append(new_img)
    new_images.append(preprocess_image(image))
#plt.show()



### Run the predictions here.
### Feel free to use as many code cells as needed.

new_labels = [13, 25, 32, 0, 7, 38, 12, 3, 17, 18]


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph('model.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    new_accuracy = evaluate(new_images, new_labels)
    print("Test Set Accuracy = {:.3f}".format(new_accuracy))



### Visualize the softmax probabilities here.
### Feel free to use as many code cells as needed.
prediction = tf.nn.softmax(logits)
top_k = tf.nn.top_k(prediction, k=3)
new_images_pred = np.asarray(new_images)
with tf.Session() as sess:
    saver = tf.train.import_meta_graph('model.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    new_prediction= sess.run(prediction, feed_dict={x: new_images_pred})
    new_top_k = sess.run(top_k, feed_dict={x: new_images_pred})

print(new_prediction)
print(new_top_k)

fig, axs = plt.subplots(10,2, figsize=(9, 19))
axs = axs.ravel()

for i in range(len(new_prediction)*2):
    if i%2 == 0:
        axs[i].axis('off')
        axs[i].imshow(cv2.cvtColor(new_images[i//2], cv2.COLOR_BGR2RGB))
    else:
        axs[i].bar(np.arange(n_classes), new_prediction[(i-1)//2]) 
        axs[i].set_ylabel('Softmax probability')

fig.show()
