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


if __name__ == '__main__':
	matfiles,offset = readData()
	frame_idx = 0 
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
		# frame_idx = 0 
		Nrows  = DataPathobj.cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
		Ncols  = DataPathobj.cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
		'in the incPCP code, mask file is saved as a mtx with width=fps/5*600'
		MaskMatfileShape=int(np.round(DataPathobj.cap.get(cv2.cv.CV_CAP_PROP_FPS)/5*trunclen))
		while frame_idx*subSampRate <mask_tensor.shape[0]:
			print "frame_idx: ", int(frame_idx*subSampRate +subSampRate*trunclen*matidx)
			# ori_img = cv2.imread(ori_list[(frame_idx*subSampRate)/choice_Interval])
			# mask    = cv2.imread(mask_list[(frame_idx*subSampRate)/choice_Interval])
			ImgSlice = (mask_tensor[frame_idx*subSampRate,:].reshape((Ncols,Nrows))).transpose() #Careful!! Warning! It's transposed!
			maskgray = ImgSlice
			plt.imshow(np.uint8(ImgSlice),cmap = 'gray')
			plt.draw()
			plt.pause(0.0001)
			pdb.set_trace()
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
				index = (frame_idx*subSampRate)/trunclen+1
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




































