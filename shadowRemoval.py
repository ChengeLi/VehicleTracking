# shadow removal 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pdb
import glob as glob

# method 1: decrease illuminance
imageList = sorted(glob.glob('/Users/Chenge/Documents/github/AIG/DoT/CanalSt@BaxterSt-96.106/imgs/*.jpg'))
testImg = cv2.imread(imageList[0])

testImgRGBSum = np.zeros(testImg.shape[0:2])
testImgRGBSum[:,:] = np.int32(testImg[:,:,0])+np.int32(testImg[:,:,1])+np.int32(testImg[:,:,2])

height, width = testImg.shape[0:2]
"""if img has white borders"""
# borderIndicator_hori = np.where(testImgRGBSum[height/2,:]==255*3)[0]
# borderIndicator_vert = np.where(testImgRGBSum[:,width/2]==255*3)[0]
# borderIndicatorDiff_hori = np.diff(borderIndicator_hori)
# borderIndicatorDiff_vert = np.diff(borderIndicator_vert)

# boarder_hori = np.where(borderIndicatorDiff_hori>=0.5*width)[0][0]+1
# boarder_vert = np.where(borderIndicatorDiff_vert>=0.5*height)[0][0]+1

# newImg = testImg[boarder_vert:(height-boarder_vert),boarder_hori:(width-boarder_hori+50),:] 
# # newImg = testImg[borderIndicator_vert[boarder_vert-1]:borderIndicator_vert[boarder_vert],\
# # borderIndicator_hori[boarder_hori-1]:borderIndicator_hori[boarder_hori],:]

newImg = testImg
NumImg = len(imageList)
NormalizedImgList = np.zeros((NumImg,newImg.shape[0],newImg.shape[1],newImg.shape[2]))

for kk in range(len(imageList)):
	newImg = cv2.imread(imageList[kk])
	# newImg = testImg[boarder_vert:(height-boarder_vert),boarder_hori:(width-boarder_hori+50),:] 
	NormalizedImg = np.zeros(newImg.shape)
	NormalizedImg[:,:,0] = newImg[:,:,0]/np.maximum(newImg[:,:,1],newImg[:,:,2])
	NormalizedImg[:,:,1] = newImg[:,:,1]/np.maximum(newImg[:,:,0],newImg[:,:,2])
	NormalizedImg[:,:,2] = newImg[:,:,2]/np.maximum(newImg[:,:,0],newImg[:,:,1])

	NormalizedImgList[kk,:,:,:] = NormalizedImg
	# plt.imshow(NormalizedImg[:,:,::-1])
	# plt.pause(0.001)

# method 2: 


















