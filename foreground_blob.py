# find blobs in the incPCP foreground mask and code different blobs using different colors
import os
import cv2
import pdb
import pickle
import numpy as np
import glob as glob
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import h5py
import scipy.ndimage as ndimg 
from scipy.io import loadmat

from DataPathclass import *
DataPathobj = DataPath(dataSource,VideoIndex)
from parameterClass import *
Parameterobj = parameter(dataSource,VideoIndex)


def blobImg2blobmatrix(maskgray):
	# maskgray = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
	# ret,thresholded = cv2.threshold(maskgray,127,255,0)
	(blobLabelMatrix, numFgPixel) = ndimg.measurements.label(maskgray)
	# BlobCenters = np.array(ndimg.measurements.center_of_mass(thresholded,blobLabelMatrix,range(1,numFgPixel+1)))
	BlobCenters = np.array(ndimg.measurements.center_of_mass(maskgray,blobLabelMatrix,range(1,numFgPixel+1)))
	# blobCenter_X_Matrix = np.zeros_like(blobLabelMatrix)
	# blobCenter_Y_Matrix = np.zeros_like(blobLabelMatrix)
	# for ii in range(numFgPixel):
	# 	blobCenter_X_Matrix[blobCenterMatrix==ii]=BlobCenters[ii][0];
	# 	blobCenter_Y_Matrix[blobCenterMatrix==ii]=BlobCenters[ii][1];
	# pdb.set_trace()
	return blobLabelMatrix, BlobCenters


def readData():
	matfilepath = DataPathobj.blobPath
	matfiles = sorted(glob.glob(matfilepath + '*.mat'))
	# matfiles = sorted(glob.glob('/Users/Chenge/Desktop/testMask/incPCPmask/' + '*.mat'))
	'==============================================================================='
	'change the offset!!'
	'==============================================================================='
	offset = 0
	matfiles = matfiles[offset:]
	return matfiles,offset



def readVideo(cap,subSampRate):
	"""bug in cap.set, not accurate"""
	# cap.set ( cv2.cv.CV_CAP_PROP_POS_FRAMES , max(0,position))
	# status, frame = cap.read()
	status, frame = cap.read()	
	for ii in range(subSampRate-1):
		status, frameskip = cap.read()

	return frame



if __name__ == '__main__':
	
	"""for video reading"""
	video_src = DataPathobj.video
	cap       = cv2.VideoCapture(video_src)
	"""this frame count is not the same with what Matlab detected! bug in opencv"""
	# nframe = np.int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
	fps    = int(np.round(cap.get(cv2.cv.CV_CAP_PROP_FPS)))


	matfiles,offset = readData()
	for matidx, matfile in enumerate(matfiles):
	# for	matidx in range(1,len(matfiles)):
		# matfile = matfiles[matidx]
		try:  #for matfile <-v7.3
			mask_tensor = loadmat(matfile)
		except:  #for matfile -v7.3
			h5pymatfile = h5py.File(matfile,'r').items()[0]
			# variableName = h5pymatfile[0]
			variableData = h5pymatfile[1]
			mask_tensor = variableData.value
			
		trunclen  = Parameterobj.trunclen
		subSampRate = int(np.round(DataPathobj.cap.get(cv2.cv.CV_CAP_PROP_FPS)/Parameterobj.targetFPS))
		blobLabelMtxList = []
		blobCenterList   = []
		Nrows  = DataPathobj.cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
		Ncols  = DataPathobj.cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
		"""in the incPCP code, mask file is saved as a mtx with width=fps/5*600"""
		MaskMatfileShape=int(np.round(DataPathobj.cap.get(cv2.cv.CV_CAP_PROP_FPS)/5*trunclen))

			

		frame_idx = 0
		while frame_idx*subSampRate <mask_tensor.shape[0]:
			print "frame_idx: ", int(frame_idx*subSampRate +subSampRate*trunclen*matidx)
			# ori_img = cv2.imread(ori_list[(fkltrame_idx*subSampRate)/choice_Interval])
			# mask    = cv2.imread(mask_list[(frame_idx*subSampRate)/choice_Interval])
			ImgSlice = (mask_tensor[frame_idx*subSampRate,:].reshape((Ncols,Nrows))).transpose() #Careful!! Warning! It's transposed!
			maskgray = ImgSlice

			"""see the foreground blob image"""

			# plt.imshow(np.uint8(ImgSlice),cmap = 'gray')
			# plt.draw()
			# plt.pause(0.0001)

			"""visualization"""
			# frame = readVideo(cap,subSampRate)
			# frame2 = frame
			# # frame2[:,:,1] = frame[:,:,1]+(maskgray*100)
			# # foreground = (np.array(frame2[:,:,1])*[np.array(maskgray==1)])[0,:,:]

			# maskedFrame = (frame[:,:,1]*maskgray)
			# maskedFrame[maskedFrame!=0]=255
			# maskedFrame_inv =frame[:,:,1]*(1-maskgray)
			# frame2[:,:,1] = maskedFrame+maskedFrame_inv
			# cv2.imwrite(DataPathobj.blobPath +str(frame_idx*subSampRate+subSampRate*trunclen*matidx).zfill(7)+'.jpg',frame2)
			# cv2.imshow('frame2', frame2)
			# cv2.waitKey(0)


			# plt.imshow(frame2[:,:,::-1])
			# plt.draw()
			# plt.pause(0.001)


			"""use ndimage.measurements"""
			blobLabelMatrix, BlobCenters = blobImg2blobmatrix(maskgray)
			sparse_slice = csr_matrix(blobLabelMatrix)
			blobLabelMtxList.append(sparse_slice)
			blobCenterList.append(BlobCenters)
			frame_idx = frame_idx+1
			#end of while loop
			if ((frame_idx>0) and (np.mod(frame_idx,trunclen)==0)) or (frame_idx*subSampRate==mask_tensor.shape[0]):
				print "Save the blob index tensor into a pickle file:"
				# savename = os.path.join(DataPathobj.blobPath,'blobLabelList'+str(matidx+1+offset).zfill(3)+'.p')
				# index = ((offset+matidx)*MaskMatfileShape+frame_idx*subSampRate)/trunclen
				index = matidx+1
				print 'index',index
				savename = os.path.join(DataPathobj.blobPath,'blobLabelList'+str(index).zfill(3)+'.p')

				pickle.dump(blobLabelMtxList, open( savename, "wb" ))
				blobLabelMtxList = []

				print "Save the blob centers..."
				# savename = os.path.join(DataPathobj.blobPath,'blobCenterList'+str(matidx+1+offset).zfill(3)+'.p')
				savename = os.path.join(DataPathobj.blobPath,'blobCenterList'+str(index).zfill(3)+'.p')
				pickle.dump(blobCenterList, open( savename, "wb" ))
				blobCenterList = []



		"""directly dumping to pickle not allowed, memory error! """




































