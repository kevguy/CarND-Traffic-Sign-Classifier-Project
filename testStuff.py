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

import matplotlib.pyplot as plt
import cv2
import random
import numpy as np
from skimage import exposure
from skimage.morphology import disk
from skimage.filters import rank

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
    yuv_img = RGB2YUV2(img_in)
    # global_constrast_img = yuv_img
    # global_constrast_img[:,:,0] = GlobalConstrastNormalize(global_constrast_img)[:,:,0]
    # local_constrast_img = yuv_img
    # local_constrast_img[:,:,0] = GlobalConstrastNormalize(local_constrast_img)[:,:,0]
    grayscale_img = grayscaleImage(img_in)
    return grayscale_img

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


### Load the images and plot them here.
### Feel free to use as many code cells as needed.
fig, axs = plt.subplots(1,5, figsize=(4, 2))
fig.subplots_adjust(hspace = .2, wspace=.001)
axs = axs.ravel()
new_images = []

n_rows = 2
n_cols = 5
fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 6))
fig.subplots_adjust(hspace=.1, wspace=.001)
axs = axs.ravel()






for i in range(n_cols):
    file_name = os.path.join(os.path.abspath('.'), 'test_images', str(i+1) + '.png')
    image = np.asarray(Image.open(file_name).convert('RGB'))
    #image = cv2.imread(file_name)

    image.setflags(write=1)
    axs[i].axis('off')
    axs[i].imshow(image)
    #new_img = preprocess_image(image)
    #axs[i+n_cols].axis('off')
    #axs[i+n_cols].imshow(new_img, cmap="Greys_r")
    #new_images.append(new_img)
    new_images.append(image)
#plt.show()

new_images = preprocess_images(new_images)

print(new_images[0].shape)