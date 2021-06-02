#%% Imports
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def feature_binary(path='0',img=None,krnl = 30): #img is the ROI image processed by get_ROI
    kern = cv.getGaborKernel((krnl,krnl),4,0,10,0.5,0,ktype=cv.CV_64F)
    kern2 = cv.getGaborKernel((krnl,krnl),4,np.pi/2,10,0.5,0,ktype=cv.CV_64F)
    if path!='0':
        img=cv.imread(path,0)
    ROI=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    fimg = cv.filter2D(ROI,cv.CV_8UC3,kern)
    fimg2=cv.filter2D(ROI,cv.CV_8UC3,kern2)
    ret,fimg_b=cv.threshold(fimg,230,255,cv.THRESH_BINARY_INV)
    ret,fimg2_b=cv.threshold(fimg2,245,255,cv.THRESH_BINARY_INV) #respectively binarized
    orimg=cv.bitwise_or(fimg_b,fimg2_b)
    return orimg

def cut_image_based_on_Gabor(img):
    orimg= feature_binary(img = img)
    cols_sum= np.sum(orimg,axis=0)
    rows_sum= np.sum(orimg,axis=1)
    i_start = np.argmax(cols_sum>0)
    i_end = np.nonzero(cols_sum>0)[-1][-1]
    j_start = np.argmax(rows_sum>0)
    j_end = np.nonzero(rows_sum>0)[-1][-1]    
    cut_img = img[j_start:j_end,i_start:i_end,:]
    return cut_img

def Normalize_NY_image(img):
    img_out = np.zeros_like(img,dtype = float)
    d0 = np.max(img[0])-np.min(img[0])
    d1 = np.max(img[1])-np.min(img[1])
    d2 = np.max(img[2])-np.min(img[2])
    if d0 == 0:
        d0 =1
    if d1 == 0:
        d1 =1
    if d2 == 0:
        d2 =1                    
    img_out[0]=(img[0]-np.min(img[0]))/d0
    img_out[1]=(img[1]-np.min(img[1]))/d1
    img_out[2]=(img[2]-np.min(img[2]))/d2
    return img_out    


def normalize_batch(batch):
    normalized_batch = batch
    normalized_batch_max = normalized_batch.max(axis=1).max(axis=1)
    normalized_batch_min = normalized_batch.min(axis=1).min(axis=1)
    span_= normalized_batch_max - normalized_batch_min 
    span_ = np.expand_dims(span_, axis=1)
    span_ = np.expand_dims(span_, axis=1)
    normalized_batch = normalized_batch/span_
    return normalized_batch
# %%
if __name__ == "__main__":
    img = cv.imread(r'C:\Users\vgliner\OneDrive - JNJ\Desktop\frame7901.jpg')
    cv.imshow('Original image',img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    for idx in range(20):
        orimg= feature_binary(img = img)
        cv.imshow('orimg',orimg)
        cv.waitKey(0)
        cv.destroyAllWindows()
    cut_img = cut_image_based_on_Gabor(img,orimg)
    cv.imshow('cut_img',cut_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
# %%
