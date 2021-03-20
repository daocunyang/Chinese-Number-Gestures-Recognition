import os
import cv2
import shutil
import random

'''
os.chdir('/Users/Daocun/Desktop/CS231A/project/data/depth')

for i in range(1, 12):
    os.mkdir(f'valid/{i}')
    os.mkdir(f'test/{i}')

    valid_samples = random.sample(os.listdir(f'train/{i}'), 20)
    for j in valid_samples:
        shutil.move(f'train/{i}/{j}', f'valid/{i}')

    test_samples = random.sample(os.listdir(f'train/{i}'), 5)
    for k in test_samples:
        shutil.move(f'train/{i}/{k}', f'test/{i}')
'''

#os.chdir('/Users/Daocun/Desktop/CS231A/project/data/depth/train')
rootdir = '/Users/Daocun/Desktop/CS231A/project/data/depth/valid/'
targetdir = '/Users/Daocun/Desktop/CS231A/project/data/depth_600/valid/'

for subdir, dirs, files in os.walk(rootdir):
    number = subdir.rsplit('/', 1)[-1]
    for file in files:
        if file.endswith(".jpg"):
            img = cv2.imread(os.path.join(subdir, file), cv2.IMREAD_UNCHANGED)
            scaled_img = cv2.resize(img, (600, 600))
            #print(os.path.join(subdir, file))
            filename = targetdir + number + "/" + file
            folder = targetdir + number
            if not os.path.isdir(folder):
                os.mkdir(folder)
                print("created directory")
            cv2.imwrite(filename, scaled_img)