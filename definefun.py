from tokenize import tabsize
from cv2 import IMREAD_COLOR
from matplotlib import image
import pandas as pd 
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from shutil import copyfile
import cv2
from skimage import io
import glob
from PIL import Image
import PIL
from skimage.transform import resize
import pandas as pd
import mahotas
from skimage.color import rgb2gray
from skimage.color import rgb2hsv
from skimage.io import imread, imshow
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
fixed_size       = tuple((500, 500))
bins             = 8

def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick
def simpleImage(path):
     load_img_rz = np.array(Image.open(path).resize((200,200)))
     return load_img_rz


def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
 

    # return the histogram
    return hist.flatten() 

def grayimage(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature    

def Calculate_testX():
    path = glob.glob("/home/younes/IdeaProjects/iaf/Data/testmer/*")
    images = []
    for file in path:
        img = cv2.imread(file)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # compute the color histogram
        hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
        cv2.normalize(hist, hist)
        images.append(hist.flatten())
    #second path
    path = glob.glob("/home/younes/IdeaProjects/iaf/Data/test/*")
    for file in path:
        img = cv2.imread(file)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # compute the color histogram
        hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
        cv2.normalize(hist, hist)
        images.append(hist.flatten())    
    images = np.array(images, dtype=object)
    return images


def Calculate_trainX():
    path = glob.glob("/home/younes/IdeaProjects/iaf/Data/trainmer/*")
    images = []
    for file in path:
        img = cv2.imread(file)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # compute the color histogram
        hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
        cv2.normalize(hist, hist)
        images.append(hist.flatten())
    #second path
    path = glob.glob("/home/younes/IdeaProjects/iaf/Data/trainailleurs/*")
    for file in path:
        img = cv2.imread(file)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # compute the color histogram
        hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
        cv2.normalize(hist, hist)
        images.append(hist.flatten())    
    images = np.array(images, dtype=object)
    return images

def npArrayRep (image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    dim = (80,80)
    #resize image
    image = cv2.resize(image, dim)
    image = np.array(image)
    return image.flatten()

def segmentationimage(img):
    originalimage=img
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    light_blue = (90, 70, 50)
    dark_blue = (128, 255, 255)
# You can use the following values for green
   # light_green = (40, 40, 40)
   # dark_greek = (70, 255, 255)
    mask = cv2.inRange(hsv_img,light_blue,dark_blue)
    result = cv2.bitwise_and(img, img, mask=mask)
    img = Image.fromarray(result, 'RGB')
    
    #img.show()
    img=np.array(img)
    return img
def ImageInParts(img):
    (h,w,c) = img.shape
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    parts = []
    step_x = 3
    step_y = 3
    eqs = []
    eq_img = np.zeros_like(gray_img)

    for x in range(step_x):
        for y in range(step_y):
            xratio1 = x/step_x
            xratio2 = (x+1)/step_x
            yratio1 = y/step_y
            yratio2 = (y+1)/step_y
            part = gray_img[int(yratio1*h):int(yratio2*h),int(xratio1*w):int(xratio2*w)].copy()
            parts.append(part)

            

            eq = cv2.equalizeHist(part)
            eqs.append(eq)    
  
    return eqs        
def DefineX():
     path = glob.glob("/home/younes/IdeaProjects/iaf/Data/Mer/*")
     images = []
     cli=0
     for file in path:
        
        img = cv2.imread(file)
        
        #images.append(ImageInParts(img))
        images.append(fd_histogram(img))
        #images.append(segmentationBlueImg(img))
        #images.append(npArrayRep (img))
        #images.append(fd_hu_moments(img))
        
        cli=cli+1
       
    
        
    #second path
     path = glob.glob("/home/younes/IdeaProjects/iaf/Data/Ailleurs/*")
     for file in path:
        
        img = cv2.imread(file)
        
        #images.append(ImageInParts(img))
        images.append(fd_histogram(img))
        #images.append(segmentationBlueImg(img))
       # images.append(npArrayRep (img))
        #images.append(fd_hu_moments(img))
        
       
     images = np.array(images)
    # nsamples, nx, ny = images.shape
    # images = images.reshape((nsamples,nx*ny))
     
     return (cli, images)

def defineY(nbic1, nbtotali):
    y=[]
    for i in range(0,nbic1):
        y.append(1)
    for i in range(nbic1,nbtotali) :
        y.append(0)
    return np.array(y)       

def contruire_dataset():
    nb, X=DefineX()
    Y=defineY(nb,len(X))
    return (X,Y)





# X_test=Calculate_testX()
#X_test = Calculate_testX()
#X_train = Calculate_trainX()
"""""
Y_train = []
Y_test = []

for i in range(0, 88):
    Y_train.append(1)
for i in range(88, len(X_train)):
    Y_train.append(0)

for i in range(0, 117):
    Y_test.append(1)
for i in range(117, len(X_test)):
    Y_test.append(0)
#Y_train = np.array(Y_train)
#Y_test = np.array(Y_test)

"""
"""
path = glob.glob("/home/younes/IdeaProjects/iaf/Data/Mer/jjjjj.jpeg")
     
for file in path:
   
    
    img = cv2.imread(file)
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    topLeft = img[0:cY, 0:cX]
    topRight = img[0:cY, cX:w]
    bottomLeft = img[cY:h, 0:cX]
    bottomRight = img[cY:h, cX:w]
    print(fd_histogram(topLeft))
    
"""
"""
path = glob.glob("/home/younes/IdeaProjects/iaf/Data/Ailleurs/99ijC3.jpeg")
for file in path:
        
    img = cv2.imread(file)
    originalimage=img
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    light_blue = (90, 70, 50)
    dark_blue = (128, 255, 255)
# You can use the following values for green
    light_green = (40, 40, 40)
    dark_greek = (70, 255, 255)
    mask = cv2.inRange(hsv_img,light_blue,dark_blue)
    result = cv2.bitwise_and(img, img, mask=mask)
    img = Image.fromarray(result, 'RGB')
    
    
    img.show()
    
    """



