# blob detection 
import cv2
import pdb
import numpy as np
import glob as glob
import matplotlib.pyplot as plt


def find_blob():
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

	return detector



if __name__ == '__main__':
	# img = cv2.imread('/home/chengeli/CUSP/park/papers/BackGroundModelling/incPCP/AIG/Canal/Canal__0100.jpg')	 
	mask_list = sorted(glob.glob('/home/chengeli/CUSP/park/papers/BackGroundModelling/incPCP/AIG/Canal_filled_hole/*.jpg'))
	ori_list  = sorted(glob.glob('../DoT/CanalSt@BaxterSt-96.106/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/*.jpg'))
	nframe    = min(len(mask_list),len(ori_list))
	frame_idx = 0
	plt.figure()

	while frame_idx <nframe:
		print "frame_idx: ", frame_idx
		mask      = cv2.imread(mask_list[frame_idx])	
		ori_img   = cv2.imread(ori_list[frame_idx])
		
		"""blob detector doesn't work..."""
		# detector = find_blob()
		# keypoints = detector.detect(img)
		# # Draw detected blobs as red circles.
		# # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
		# im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		# plt.imshow(im_with_keypoints)


		maskgray            = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
		ret,thresh          = cv2.threshold(maskgray,127,255,0)
		contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		

		# cnt = contours[4]
		# cv2.drawContours(mask, [cnt], 0, (0,255,0), 3)

		cv2.drawContours(ori_img, contours, -1, (0,255,0),1)
		plt.imshow(ori_img[:,:,::-1])
		plt.pause(0.00001)
		frame_idx = frame_idx+1


		cnts = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
 		# fix me, sort based on area, or others????
 		# fill the contour for the blob
		cv2.drawContours(ori_img, cnts, -1, (0,255,0),1)
		plt.imshow(ori_img[:,:,::-1])






















