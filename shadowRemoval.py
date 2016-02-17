# shadow removal 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pdb
import glob as glob


def NormalizeImg(oriImg):
	NormalizedImg2 = np.zeros(oriImg.shape)
	# NormalizedImg[:,:,0] = oriImg[:,:,0]/np.maximum(oriImg[:,:,1],oriImg[:,:,2])
	# NormalizedImg[:,:,1] = oriImg[:,:,1]/np.maximum(oriImg[:,:,0],oriImg[:,:,2])
	# NormalizedImg[:,:,2] = oriImg[:,:,2]/np.maximum(oriImg[:,:,0],oriImg[:,:,1])
	oriImg2 = np.double(oriImg)
	epsilon = pow(10,-6)
	# epsilon = 0
	NormalizedImg2[:,:,0] = oriImg2[:,:,0]/(oriImg2[:,:,0]+oriImg2[:,:,1]+oriImg2[:,:,2]+epsilon)
	NormalizedImg2[:,:,1] = oriImg2[:,:,1]/(oriImg2[:,:,0]+oriImg2[:,:,1]+oriImg2[:,:,2]+epsilon)
	NormalizedImg2[:,:,2] = oriImg2[:,:,2]/(oriImg2[:,:,0]+oriImg2[:,:,1]+oriImg2[:,:,2]+epsilon)


	NormalizedImg = np.zeros(oriImg.shape)
	# epsilon = 0
	NormalizedImg[:,:,0] = np.double(oriImg[:,:,0])/(oriImg[:,:,0]+oriImg[:,:,1]+oriImg[:,:,2]+epsilon)
	NormalizedImg[:,:,1] = np.double(oriImg[:,:,1])/(oriImg[:,:,0]+oriImg[:,:,1]+oriImg[:,:,2]+epsilon)
	NormalizedImg[:,:,2] = np.double(oriImg[:,:,2])/(oriImg[:,:,0]+oriImg[:,:,1]+oriImg[:,:,2]+epsilon)
	

	pdb.set_trace()

	# NormalizedImg[:,:,:][np.isnan(NormalizedImg[:,:,:])] = 0
	# NormalizedImg[:,:,:][NormalizedImg[:,:,:]==inf] = 0
	pdb.set_trace()
	NormalizedImg = np.uint8(NormalizedImg*255)
	plt.figure()
	plt.imshow(NormalizedImg[:,:,::-1])
	plt.figure()
	plt.imshow(oriImg[:,:,::-1])
	pdb.set_trace()

	NormalizedImg[:,:,0] = np.uint8(NormalizedImg[:,:,0]/(np.nanmax(NormalizedImg[:,:,0][:])-np.min(NormalizedImg[:,:,0][:]))*255)
	NormalizedImg[:,:,1] = np.uint8(NormalizedImg[:,:,1]/(np.nanmax(NormalizedImg[:,:,1][:])-np.min(NormalizedImg[:,:,1][:]))*255)
	NormalizedImg[:,:,2] = np.uint8(NormalizedImg[:,:,2]/(np.nanmax(NormalizedImg[:,:,2][:])-np.min(NormalizedImg[:,:,2][:]))*255)

	return NormalizedImg

def cutOutBorder(Img):
	"""if img has white borders"""
	height,width = Img.shape[:2]
	ImgRGBSum = np.zeros(Img.shape[0:2])
	ImgRGBSum[:,:] = np.int32(Img[:,:,0])+np.int32(Img[:,:,1])+np.int32(Img[:,:,2])

	borderIndicator_hori = np.where(ImgRGBSum[height/2,:]==255*3)[0]
	borderIndicator_vert = np.where(ImgRGBSum[:,width/2]==255*3)[0]
	borderIndicatorDiff_hori = np.diff(borderIndicator_hori)
	borderIndicatorDiff_vert = np.diff(borderIndicator_vert)

	boarder_hori = np.where(borderIndicatorDiff_hori>=0.5*width)[0][0]+1
	boarder_vert = np.where(borderIndicatorDiff_vert>=0.5*height)[0][0]+1

	# oriImg = Img[boarder_vert:(height-boarder_vert),boarder_hori:(width-boarder_hori+50),:] 
	newImg2 = Img[borderIndicator_vert[boarder_vert-1]:borderIndicator_vert[boarder_vert],\
	borderIndicator_hori[boarder_hori-1]:borderIndicator_hori[boarder_hori],:]

	return newImg2


# method 1: decrease illuminance
imageList = sorted(glob.glob('../DoT/CanalSt@BaxterSt-96.106/imgs/*.jpg'))
oriImg = cv2.imread(imageList[0])
height, width = oriImg.shape[0:2]
NumImg = len(imageList)
NormalizedImgList = np.zeros((NumImg,oriImg.shape[0],oriImg.shape[1],oriImg.shape[2]))

for kk in range(len(imageList)):
	oriImg = cv2.imread(imageList[kk])
	NormalizedImg = NormalizeImg(oriImg)
	NormalizedImgList[kk,:,:,:] = NormalizedImg
	# plt.imshow(NormalizedImg[:,:,::-1])
	# plt.pause(0.001)
pdb.set_trace()



# method 2: 

oriImg = cv2.imread(imageList[20])

aveImg = cv2.imread('../DoT/CanalSt@BaxterSt-96.106/ave3.png')
aveImg = cutOutBorder(aveImg)
aveImg = cv2.resize(aveImg, oriImg.shape[:2][::-1]) # the resize function is dimension reversed...==


diffImg = np.abs(oriImg - aveImg)
diffImg = cv2.cvtColor(diffImg, cv2.COLOR_BGR2GRAY)
th,diffImg = cv2.threshold(diffImg,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


# NormalizedImg = np.zeros(oriImg.shape)
# NormalizedImg = np.uint16(oriImg/(aveImg+np.exp(-8)))
NormalizeImg = NormalizeImg(oriImg)

A = 1
gamma = 1.01
NormalizedImg = A*NormalizedImg**gamma


NormalizedImg = np.array(NormalizedImg, dtype=np.uint16)
plt.figure()
plt.imshow(NormalizedImg)

fr_shadow = cv2.cvtColor(NormalizedImg, cv2.COLOR_BGR2GRAY)
fr_shadow[:,:] = np.uint8(fr_shadow[:,:]/(np.nanmax(fr_shadow[:,:])-np.min(fr_shadow[:,:]))*255)


# fr_shadow = np.array(fr_shadow, dtype=np.uint8)

thresh,fr_shadow_bin = cv2.threshold(fr_shadow,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
plt.figure()
plt.imshow(fr_shadow_bin)

finalImg = np.zeros(oriImg.shape[:2])
for hh in range(height):
	for ww in range(width):
		if (diffImg[hh,ww]>=0) and (fr_shadow[hh,ww]>thresh):
			finalImg[hh,ww] = fr_shadow[hh,ww]
		else:
			finalImg[hh,ww] = 0

plt.figure()
plt.imshow(np.uint8(finalImg))


















