import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from matplotlib import pyplot as plt

INVALID_LABEL = "Invalid / No hand"

def get_top_left_and_bottom_right_coordinates(frame_width, frame_height, size, thickness):
   assert size < frame_width and size < frame_height
   top_left_y = int((frame_height-size)/2-thickness)
   top_left_x = int((frame_width-size)/2-thickness)
   bottom_right_y = int(top_left_y + size + thickness)
   bottom_right_x = int(top_left_x + size + thickness)
   return (top_left_x, top_left_y), (bottom_right_x, bottom_right_y)

'''
    Hand segmentation. Assumes hand appears in the center of the frame
    Adapted from: https://github.com/mbeyeler/opencv-python-blueprints/blob/master/chapter2/gestures.py
'''
def segment_hand(frame):
    # cv2.imshow('original frame', frame)
    height, width = frame.shape
    # find center (21x21 pixel) region of the frame
    # half-width of 21 is 21/2-1
    center_half = 10
    center = frame[height//2 - center_half:height//2 + center_half, width//2 - center_half:width//2 + center_half]

    # get median depth value of center region
    med = np.median(center)

    # set pixel color to gray if diff to median is within a certain value, otherwise set to 0
    frame = np.where(abs(frame-med) <= 10, 128, 0).astype(np.uint8)

    # morphological closing to close small holes in hand region
    kernel = np.ones((3, 3), np.uint8)
    frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)

    # find regions connected to the hand (center of the frame)
    small_kernel = 3
    frame[height//2 - small_kernel:height//2 + small_kernel, width//2 - small_kernel:width//2 + small_kernel] = 128

    mask = np.zeros((height+2, width+2), np.uint8)
    flood = frame.copy()
    floodflags = 4
    floodflags |= (255 << 8)
    cv2.floodFill(flood, mask, (width//2, height//2), 255, flags=floodflags)

    # thresholding to remove non-hand pixels
    _, flooded = cv2.threshold(flood, 129, 255, cv2.THRESH_BINARY)
    #cv2.imshow('after segmentation', flooded)
    return flooded

def capture_image(depth_img, img_count, path):
    img = cv2.resize(depth_img, (224, 224))
    img = cv2.bilateralFilter(img, 5, 50, 100)
    filename = "{}.jpg".format(img_count)
    cv2.imwrite(path + filename, img)
    print("Captured image #{}".format(img_count))

def plot_two_images(img1, img2):
    plt.subplot(211)
    plt.imshow(img1)
    plt.title('Original Frame')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(212)
    plt.imshow(img2)
    plt.title('Region of Interest (ROI)')
    plt.xticks([])
    plt.yticks([])
    plt.suptitle("Image Acquisition")
    plt.show()

def resize_and_smooth(img):
    img = cv2.resize(img, (224, 224))
    img = cv2.bilateralFilter(img, 5, 50, 100)
    return img

'''
    Convert an image into a format that can be used by the model
'''
def convert_image(img):
    img_array_expanded_dims = np.expand_dims(img, axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

'''
    For getting the label mapping with the previously saved model. 
    https://stackoverflow.com/questions/38971293/get-class-labels-from-keras-functional-model
'''
def get_class_mapping():
    return {
        0: 1,
        1: 10,
        2: INVALID_LABEL,
        3: 2,
        4: 3,
        5: 4,
        6: 5,
        7: 6,
        8: 7,
        9: 8,
        10: 9
    }