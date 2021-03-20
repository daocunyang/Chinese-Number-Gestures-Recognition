import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk

import util

# initialize the trained model for performing recognition
model = tf.keras.models.load_model("../models/train1_MobileNets_V1_base_original_data")
#print(model.summary())
mapping = util.get_class_mapping()

#cam_video.set(cv2.CAP_PROP_FRAME_WIDTH, 320);
#cam_video.set(cv2.CAP_PROP_FRAME_HEIGHT, 180);

frame_count = 0
image_number = 22  # used for saving the training images

SCALE_FACTOR = 2
BORDER_THICKNESS = 1

# capture image of following size
TARGET_WIDTH = (224*SCALE_FACTOR)+2*BORDER_THICKNESS
TARGET_HEIGHT = (224*SCALE_FACTOR)+2*BORDER_THICKNESS

# True = Capture training images, False = use it to recognize gestures
CAPTURE_MODE = False

last_prediction = util.INVALID_LABEL

# Initialize recognition output text widget
output_text = tk.Tk()
output_text.title('Recognition Output')
output_text.geometry("350x50")

label = tk.Label(output_text, text=last_prediction)
label.pack()
label.config(font=("Times", 44))
label.config(fg="#0000FF")
label.config(bg="yellow")

# start capturing video
cam_video = cv2.VideoCapture(0)

if not cam_video.isOpened():
    print("------ PANIC!!! Cannot open camera ------")
    exit()

fps = int(cam_video.get(cv2.CAP_PROP_FPS))
# default width and height on MacBook pro: 1280 x 720 (16:9)
width = cam_video.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cam_video.get(cv2.CAP_PROP_FRAME_HEIGHT)
print("fps: {}, original width: {}, original height:, {}".format(fps, width, height))

cv2.startWindowThread()

while True:
    # Capture frame-by-frame
    ret, frame = cam_video.read()
    output_text.update()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    original_frame = frame
    frame = cv2.flip(frame, 1)  # flip the frame horizontally
    # frame shape = (720, 1280, 3)
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    # point (x, y) - (horizontal, vertical)
    top_left = (frame_width - TARGET_WIDTH, 0)
    bottom_right = (frame_width, TARGET_HEIGHT)
    cv2.rectangle(frame, top_left, bottom_right, color=(0, 255, 0), thickness=BORDER_THICKNESS)

    img = util.remove_background(frame)

    # Clip the region of interest (ROI)
    ROI = img[1:TARGET_HEIGHT-1, frame_width - TARGET_WIDTH+1:frame_width-1]
    cv2.imshow('img', img)
    cv2.waitKey(1)
    #util.plot_two_images(frame, img)

    if CAPTURE_MODE:
        # Capture images for training
        # MAKE SURE MOUSE FOCUS IS ON THE IMAGE WINDOW!!!
        key = cv2.waitKey(0)
        #print("Key: {}".format(str(key)))
        if key == ord('q'):
            print("Program exiting...")
            break
        elif key == ord('c'):
            filename = "{}.jpg".format(image_number)
            resized_img = cv2.resize(ROI, (224, 224))
            smoothed_img = cv2.bilateralFilter(resized_img, 5, 50, 100)  # smoothing filter
            cv2.imwrite(filename, smoothed_img)
            print("Captured image #{}".format(image_number))
            image_number += 1
        else:
            # when other keys are pressed, just continue to next frame
            pass
    else:
        # run every 0.5 second
        #print("frame_count: {}".format(str(frame_count)))
        if frame_count % fps == 0 or frame_count % fps == 0.5*fps:
            # perform hand gesture recognition
            img = util.resize_and_smooth(ROI)
            img = util.convert_image(img)
            prediction = model.predict(img)
            arg = np.argmax(prediction)
            number = mapping[arg]
            if number != last_prediction:
                last_prediction = number
                label.config(text=str(number))
                print("Predicted number: {}".format(str(number)))
    frame_count += 1

# When everything done, release the capture
cam_video.release()
cv2.destroyAllWindows()