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


def readData(userPCA):
	if userPCA:
		maskfiles = sorted(glob.glob(DataPathobj.blobPath + '*.mat'))
		# matfiles = sorted(glob.glob('/Users/Chenge/Desktop/testMask/incPCPmask/' + '*.mat'))
	else:
		maskfiles = sorted(glob.glob(DataPathobj.blobPath + '*running_bgsub_mask_tensor*.p'))

	"""==============================================================================="""
	"""change the offset!!"""
	"""==============================================================================="""
	offset = 0
	maskfiles = maskfiles[offset:]
	return maskfiles,offset


def readVideo(cap,subSampRate):
	"""bug in cap.set, not accurate"""
	# cap.set ( cv2.cv.CV_CAP_PROP_POS_FRAMES , max(0,position))
	# status, frame = cap.read()
	status, frame = cap.read()	
	for ii in range(subSampRate-1):
		status, frameskip = cap.read()

	return frame



if __name__ == '__main__':
	userPCA = True
	maskfiles, offset = readData(userPCA)
	"""this frame count is not the same with what Matlab detected! bug in opencv"""
	# nframe = np.int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
	fps = int(np.round(DataPathobj.cap.get(cv2.cv.CV_CAP_PROP_FPS)))
	# assert fps==30, "fps=%d" %fps

	# fps = 30
	for matidx, matfile in enumerate(maskfiles):
	# for	matidx in range(len(maskfiles)-1,len(maskfiles),1):
	# for	matidx in range(0,5,1):
		matfile = maskfiles[matidx]
		if userPCA:
			# try:  #for matfile <-v7.3
			# 	mask_tensor = loadmat(matfile)
			# except:  #for matfile -v7.3
			h5pymatfile = h5py.File(matfile,'r').items()[0]
			# variableName = h5pymatfile[0]
			variableData = h5pymatfile[1]
			mask_tensor = variableData.value
		else:
			mask_tensor_sparse = pickle.load(open(matfile,'rb'))
			mask_tensor = []
			for ii in range(len(mask_tensor_sparse)):
				mask_tensor.append(csr_matrix(mask_tensor_sparse[ii]).toarray())
			mask_tensor = np.array(mask_tensor)


			
		trunclen  = Parameterobj.trunclen
		subSampRate = int(fps/Parameterobj.targetFPS)
		subSampRate_matlab = int(30/Parameterobj.targetFPS)  ##IT'S JUST 6

		blobLabelMtxList = []
		blobCenterList   = []
		Nrows  = DataPathobj.cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
		Ncols  = DataPathobj.cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
		"""in the incPCP code, mask file is saved as a mtx with width=fps/5*600"""
		# MaskMatfileShape=int((fps/5)*trunclen)
		MaskMatfileShape=int((30/5)*trunclen)

		
		frame_idx = 0
		global_frame_idx = int(mask_tensor.shape[0]*(offset+matidx)+frame_idx*subSampRate_matlab)
		while frame_idx*subSampRate_matlab <mask_tensor.shape[0]:
			print "frame_idx: ", frame_idx
			print "global_frame_idx: ", global_frame_idx

			# ori_img = cv2.imread(ori_list[(fkltrame_idx*subSampRate)/choice_Interval])
			# mask    = cv2.imread(mask_list[(frame_idx*subSampRate)/choice_Interval])
			if userPCA:
				ImgSlice = (mask_tensor[frame_idx*subSampRate_matlab,:].reshape((Ncols,Nrows))).transpose() #Careful!! Warning! It's transposed!
			else:
				ImgSlice = mask_tensor[frame_idx*subSampRate_matlab,:].reshape((Nrows,Ncols))

			maskgray = ImgSlice

			# """see the foreground blob image"""

			# plt.imshow(np.uint8(ImgSlice),cmap = 'gray')
			# plt.draw()
			# plt.pause(0.0001)

			"""visualization"""
			# frame = readVideo(DataPathobj.cap,subSampRate)

			# frame2 = frame
			# # # frame2[:,:,1] = frame[:,:,1]+(maskgray*100)
			# # # foreground = (np.array(frame2[:,:,1])*[np.array(maskgray==1)])[0,:,:]

			# maskedFrame = (frame[:,:,1]*maskgray)
			# maskedFrame[maskedFrame!=0]=255
			# maskedFrame_inv = frame[:,:,1]*(1-maskgray)
			# frame2[:,:,1] = maskedFrame+maskedFrame_inv
			# # cv2.imwrite(DataPathobj.blobPath +str(frame_idx*subSampRate+subSampRate*trunclen*matidx).zfill(7)+'.jpg',frame2)
			# # cv2.imshow('frame2', frame2)
			# # cv2.waitKey(0)

			# plt.imshow(frame2[:,:,::-1])
			# plt.draw()
			# plt.pause(0.001)


			"""use ndimage.measurements"""

			blobLabelMatrix, BlobCenters = blobImg2blobmatrix(maskgray)
			sparse_slice = csr_matrix(blobLabelMatrix)
			blobLabelMtxList.append(sparse_slice)
			blobCenterList.append(BlobCenters)
			frame_idx = frame_idx+1
			global_frame_idx = int(mask_tensor.shape[0]*(offset+matidx)+frame_idx*subSampRate)
			#end of while loop

			# if global_frame_idx<1800:
			# 	continue
			if ((frame_idx>0) and (np.mod(frame_idx,trunclen)==0)) or (frame_idx*subSampRate_matlab>=mask_tensor.shape[0]):
				print "Save the blob index tensor into a pickle file:"
				# savename = os.path.join(DataPathobj.blobPath,'blobLabelList'+str(matidx+1+offset).zfill(3)+'.p')
				# index = global_frame_idx/(trunclen*subSampRate)
				index = offset+matidx
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




































