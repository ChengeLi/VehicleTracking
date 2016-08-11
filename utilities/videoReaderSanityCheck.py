#opencv videoreader test

import cv2
import os
import sys
import pdb
import glob as glob

import matplotlib.pyplot as plt
from DataPathclass import *
DataPathobj = DataPath(dataSource,VideoIndex)
from parameterClass import *
Parameterobj = parameter(dataSource,VideoIndex)


def readVideo(cap,subSampRate):
    """when read video in a loop, every subSampRate frames"""
    status, frame = cap.read()  
    for ii in range(subSampRate-1):
        status, frameskip = cap.read()
    return frame, status


def readBuffer(startOffset, cap):
	for ii in range(startOffset):
		ret, frame = cap.read()
	return cap



cap = cv2.VideoCapture(DataPathobj.video)
print 'fps=', np.int(DataPathobj.cap.get(cv2.cv.CV_CAP_PROP_FPS))  ## this is not reliable
print 'whole frame count=', np.int(DataPathobj.cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))  ## this is not reliable, either
startOffset = 0
cap = readBuffer(startOffset, cap)
frameInd = 0
subSampRate = 1
status =  True
while status:
	frame,status = readVideo(cap, subSampRate)
	# cv2.imshow('vis', frame)
	# cv2.waitKey(5)    
	# print 'current frame loc=', DataPathobj.cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
	frameInd+=1
	print startOffset+(frameInd-1)*subSampRate


print 'is the last ', startOffset+(frameInd-1)*subSampRate, '= 54552?'


















