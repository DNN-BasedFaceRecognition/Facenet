from os import listdir
from os.path import isdir
from PIL import Image
from numpy import asarray
from numpy import savez_compressed
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
from glob import glob
import os.path
#import rawpy, imageio
import cv2
import math
import random

#changes made to RGB color space didnt do anything. Accuracy didnt
# change at all even for a (120,150) chnage.
# HSV colorspace only reduced the accuracy by about 5%...

randx, randy = 120, 150

foldername = '/home/rohan/SeniorDesign/FaceDataset/train/Rohan Kapoor/' 
foldername2 = '/home/rohan/SeniorDesign/Facenet/test_images/' 
pic_name = 'IMG_0776.jpg'

def open_face(filename):
    img_org = cv2.imread(filename)
    cv2.imshow("Original", img_org)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    img_org = cv2.cvtColor(img_org, cv2.COLOR_RGB2HSV)
    cv2.imshow("HSV", img_org)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    rows,columns,channels = img_org.shape
    print("shape:", rows, columns, channels)

    for x in range(0, rows):
        for y in range(0, columns):
            r, g, b = img_org[x,y]
            i = random.randint(1,6)
            if i == 1:
                r = r+random.randint(randx,randy)
                if r > 255:
                    r = 255
            if i == 2:
                r = r-random.randint(randx,randy)
                if r < 0:
                    r = 0
            if i == 3:
                g = g+random.randint(randx,randy)
                if g > 255:
                    g = 255
            if i == 4:
                g = g-random.randint(randx,randy)
                if g < 0:
                    g = 0
            if i == 5:
                b = b+random.randint(randx,randy)
                if b > 255:
                    b = 255
            if i == 6:
                b = b-random.randint(randx,randy)
                if b < 0:
                    b = 0
            img_org[x,y] = [r,g,b]

    img_org = cv2.cvtColor(img_org, cv2.COLOR_HSV2RGB)
                
    cv2.imshow("New", img_org)
    cv2.waitKey(0)
    cv2.destroyAllWindows()     

    cv2.imwrite(foldername2 + 'copy-' + pic_name, img_org)
    


open_face(foldername + pic_name)

