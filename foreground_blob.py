# find blobs in the incPCP foreground mask and code different blobs using different colors
import os
import cv2
import pdb
import pickle
import numpy as np
import glob as glob
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

"""blob detector doesn't work..."""
"""
def find_blob(ori_img):
	# doesn't work!! segmentation fault (core dumped)
	# Setup SimpleBlobDetector parameters.
	params = cv2.SimpleBlobDetector_Params()
	 
	# Change thresholds
	params.minThreshold = 10;
	params.maxThreshold = 200;
	 
	# Filter by Area.
	params.filterByArea = False
	params.minArea = 10 #1500
	 
	# Filter by Circularity
	params.filterByCircularity = False
	params.minCircularity = 0.1
	 
	# Filter by Convexity
	params.filterByConvexity = False
	params.minConvexity = 0.87
	 
	# Filter by Inertia
	params.filterByInertia = False
	params.minInertiaRatio = 0.01
	 
	# Create a detector with the parameters
	ver = (cv2.__version__).split('.')
	if int(ver[0]) < 3 :
		detector = cv2.SimpleBlobDetector(params)
	else: 
		detector = cv2.SimpleBlobDetector_create(params)

	keypoints = detector.detect(ori_img)
	# Draw detected blobs as red circles.
	# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
	im_with_keypoints = cv2.drawKeypoints(ori_img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	plt.imshow(im_with_keypoints)

	return keypoints
"""

def blobImg2blobmatrix(mask,ori_img):
	maskgray            = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
	ret,thresh          = cv2.threshold(maskgray,127,255,0)
	contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	# draw all the contours
	# ori_imgcopy         = ori_img.copy()
	# cv2.drawContours(ori_imgcopy, contours, -1, (0,255,0),thickness = 1) 
	# plt.imshow(ori_imgcopy[:,:,::-1])
	# plt.pause(0.00001)
	
	#filter the contours by area, only keep the first 20 
	cnts       = sorted(contours, key = cv2.contourArea, reverse = True)[:40] 
	blobImage  = np.zeros(ori_img.shape)
	blobMatrix = np.zeros(ori_img.shape[:-1])
	color      = np.array([np.random.randint(0,255) for _ in range(3*int(len(cnts)))]).reshape(int(len(cnts)),3)

	for k in range(len(cnts)):
		if cv2.contourArea(cnts[k])<20:
			break
		# draw in class number, starts from 1
		cv2.drawContours(blobImage, cnts, k,(k+1,k+1,k+1),thickness = cv2.cv.CV_FILLED) # fill the contour for the blob
		# draw in real color
		# cv2.drawContours(blobImage, cnts, k,(color[k]),thickness = cv2.cv.CV_FILLED) # fill the contour for the blob
		blobMatrix = np.sum(blobImage,2)/3
		# if want to save the colored img
		# blobname = '/media/TOSHIBA/DoTdata/CanalSt@BaxterSt-96.106/incPCP/Canal_blobImage/'+str(frame_idx).zfill(7)+'.jpg'
		# cv2.imwrite(blobname, blobImage)
	# plt.imshow(blobImagee)
	# plt.pause(0.00001)


	return blobMatrix




if __name__ == '__main__':
	mask_list = sorted(glob.glob('/media/TOSHIBA/DoTdata/CanalSt@BaxterSt-96.106/incPCP/Canal_filled_hole/*.jpg'))
	ori_list  = sorted(glob.glob('../DoT/CanalSt@BaxterSt-96.106/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/*.jpg'))
	nframe    = min(len(mask_list),len(ori_list))
	frame_idx = 0
	trunclen  = 600
	plt.figure()

	# temp_img   = cv2.imread(ori_list[0])
	# blobTensor = np.zeros((temp_img.shape[0],temp_img.shape[1],nframe))
	blob_sparse_list = []
	while frame_idx <nframe:
		print "frame_idx: ", frame_idx
		mask    = cv2.imread(mask_list[frame_idx])	
		ori_img = cv2.imread(ori_list[frame_idx])
		
		"""blob detector doesn't work..."""
		# keypoints = find_blob(ori_img)
		"""use cv2.findContours, and then fill the contours..."""
		blobMatrix = blobImg2blobmatrix(mask,ori_img)
		# blobTensor[:,:,frame_idx] = blobMatrix
		sparse_slice = csr_matrix(blobMatrix)
		blob_sparse_list.append(sparse_slice)

		frame_idx = frame_idx+1
		#end of while loop
		if frame_idx>0 and np.mod(frame_idx,trunclen)==0:
			print "Save the blob index tensor into a pickle file:"
			# savePath = '../DoT/CanalSt@BaxterSt-96.106/'
			savePath = '/media/TOSHIBA/DoTdata/CanalSt@BaxterSt-96.106/incPCP/'
			savename = os.path.join(savePath,'blobTensorList'+str(frame_idx/trunclen).zfill(3)+'.p')
			pickle.dump(blob_sparse_list, open( savename, "wb" ))
			blob_sparse_list = []



	"""directly dumping to pickle not allowed, memory error! """
	# savename = os.path.join(savePath,'blobTensor.p')
	# pickle.dump( blobTensor, open( savename, "wb" ) )






































