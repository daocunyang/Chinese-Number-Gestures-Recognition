import numpy as np
import cv2
import sys
import tensorflow as tf
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Frame
import tkinter as tk

import util

# initialize the trained model for performing recognition
model = tf.keras.models.load_model("../models/train6_MobileNets_V2_base_depth_data")
#print(model.summary())
mapping = util.get_class_mapping()

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

try:
    from pylibfreenect2 import OpenGLPacketPipeline
    pipeline = OpenGLPacketPipeline()
except:
    try:
        from pylibfreenect2 import OpenCLPacketPipeline
        pipeline = OpenCLPacketPipeline()
    except:
        from pylibfreenect2 import CpuPacketPipeline
        pipeline = CpuPacketPipeline()
print("Packet pipeline:", type(pipeline).__name__)

enable_rgb = True
enable_depth = True

fn = Freenect2()
num_devices = fn.enumerateDevices()
if num_devices == 0:
    print("No device found.")
    sys.exit(1)

serial = fn.getDeviceSerialNumber(0)
device = fn.openDevice(serial, pipeline=pipeline)

types = 0
if enable_rgb:
    types |= FrameType.Color
if enable_depth:
    types |= (FrameType.Ir | FrameType.Depth)
listener = SyncMultiFrameListener(types)

# Register listeners
device.setColorFrameListener(listener)
device.setIrAndDepthFrameListener(listener)

if enable_rgb and enable_depth:
    device.start()
else:
    device.startStreams(rgb=enable_rgb, depth=enable_depth)

SCALE_FACTOR = 1.5
size = 224*SCALE_FACTOR
THICKNESS = 2
FPS = 30

FRAME_WIDTH = 512
FRAME_HEIGHT = 424
TARGET_WIDTH = size+2*THICKNESS
TARGET_HEIGHT = size+2*THICKNESS

CAPTURE_MODE = False
IMG_SAVE_PATH = "/Users/Daocun/Desktop/CS231A/project/data/"

frame_count = 0
img_count = 1  # used when naming the training samples

undistorted = Frame(FRAME_WIDTH, FRAME_HEIGHT, 4)
registered = Frame(FRAME_WIDTH, FRAME_HEIGHT, 4)

while True:
    frames = listener.waitForNewFrame()
    output_text.update()

    if enable_rgb:
        color = frames["color"]
    if enable_depth:
        ir = frames["ir"]
        depth = frames["depth"]

    if enable_depth:
        depth = depth.asarray()
        depth = util.segment_hand(depth)
        top_left, bottom_right = util.get_top_left_and_bottom_right_coordinates(512, 424, size, THICKNESS)
        cv2.rectangle(depth, top_left, bottom_right, color=(255, 0, 0), thickness=THICKNESS)
        if CAPTURE_MODE:
            # capture an image every second, and save as a training sample
            if frame_count % FPS == 0:
               ROI = depth[top_left[1]+2:bottom_right[1], top_left[0]+2:bottom_right[0]]
               #cv2.imshow('img', ROI)
               util.capture_image(ROI, img_count, IMG_SAVE_PATH)
               img_count += 1
        else:
            # capture every 0.5 second
            if frame_count % FPS == 0 or frame_count % FPS == 0.5 * FPS:
                ROI = depth[top_left[1] + 2:bottom_right[1], top_left[0] + 2:bottom_right[0]]
                img = util.resize_and_smooth(ROI)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                img = util.convert_image(img)
                prediction = model.predict(img)
                arg = np.argmax(prediction)
                number = mapping[arg]
                if number != last_prediction:
                    last_prediction = number
                    label.config(text=str(number))
                    print("Predicted number: {}".format(str(number)))
        cv2.imshow("depth image", depth)
    if enable_rgb:
        cv2.imshow("color", cv2.resize(color.asarray(), (int(1920/3), int(1080/3))))
    listener.release(frames)

    key = cv2.waitKey(delay=1)
    if key == ord('q'):
        break

    frame_count += 1

device.stop()
device.close()