#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

from DataPathclass import *
DataPathobj = DataPath(dataSource,VideoIndex)
from parameterClass import *
Parameterobj = parameter(dataSource,VideoIndex)
import scipy.ndimage as ndimg
from scipy.sparse import csr_matrix
import pickle


# -- open the video
# cap    = cv2.VideoCapture("../data/sb-camera4-0750am-0805am.avi")
cap = cv2.VideoCapture(DataPathobj.video)

st, fr = cap.read()

# -- initialize the images and background
img0 = fr.mean(-1)
img  = 1.0*np.zeros_like(fr)
bkg  = 1.0*np.zeros_like(fr)

# -- utilities
alpha = 0.01

# -- initialize visualization
fig, ax = plt.subplots(1,2,figsize=[15,8])
fig.subplots_adjust(0.05,0.05,0.95,0.95)
[i.axis("off") for i in ax]
fi = ax[0].imshow(fr[:,:,::--1])
im = ax[1].imshow(img0,"gist_gray",interpolation="nearest",clim=[0,1])

# -- loop through frames
print("burning in background...")
buringLen = 400
for ii in range(buringLen):
    st, fr[:,:,:] = cap.read()
    img[:,:,:] = fr
    bkg[:,:,:] = (1.0-alpha)*bkg + alpha*img


mask_tensor = []
videolength = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

frame_idx  = 0
cap = cv2.VideoCapture(DataPathobj.video) #reinitialize to start in the beginning

# for frame_idx in range(buringLen+1,videolength,1):
while frame_idx<videolength:
    print 'frame_idx',frame_idx
    st, fr[:,:,:] = cap.read()
    img[:,:,:] = fr
    bkg[:,:,:] = (1.0-alpha)*bkg + alpha*img

    mask = np.abs(img-bkg).max(2)>30

    """get the foreground mask"""
    # mask_open = ndimg.morphology.binary_opening(mask)
    # masknohole = ndimg.binary_fill_holes(mask_open).astype(int)

    masknohole = ndimg.binary_fill_holes(mask).astype(int)
    mask_close = ndimg.morphology.binary_closing(masknohole).astype(int)

    fi.set_data(fr[:,:,::-1])
    # im.set_data(mask)
    im.set_data(mask_close)
    fig.canvas.draw()
    plt.pause(1e-3)

    mask_close = csr_matrix(mask_close)
    mask_tensor.append(mask_close)

    frame_idx+=1
    """saving..."""
    # if (frame_idx>0 and np.mod(frame_idx,Parameterobj.trunclen)==0) or frame_idx==videolength:
    #     print "Save the mask tensor into a pickle file:"
    #     index = frame_idx/Parameterobj.trunclen
    #     print 'index',index
    #     savename = os.path.join(DataPathobj.blobPath,'running_bgsub_mask_tensor'+str(index).zfill(3)+'.p')
    #     pickle.dump(mask_tensor, open( savename, "wb" ))
    #     mask_tensor = []




