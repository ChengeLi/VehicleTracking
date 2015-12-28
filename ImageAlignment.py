import cv2
import pdb
import numpy as np
from matplotlib import pyplot as plt

def getPerspectiveMtx(img):

	# img = cv2.imread('../DoT/CanalSt@BaxterSt-96.106/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/00000601.jpg')
	img = np.asarray(img)
	# rows,cols,ch = img.shape

	# img_small = cv2.resize(img,(cols/2,rows/2))  # the resize function is x first, y after

	# pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
	# pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])


	# pts1 = np.float32([[328, 431],[483,435], [26,615],[797,621]])  # beatles 11.jpg
	# pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])


	# pts1 = np.float32([[161, 147],[224,147], [73,519],[399,397]])  # canal st, left lane
	# pts1 = np.float32([[229, 164],[340,151], [451,444],[703,331]])  # canal st, right lane
	#

	# pts2 = np.float32([[0,0],[350,0],[0,305],[350,305]])
	# pts2 = np.float32([[0,0],[350,0],[0,600],[350,600]])
	# pts2 = np.float32([[0,0],[380,0],[0,380],[380,380]]) #canal

	# pts1 = np.float32([[2593, 175],[2877,242], [1326,2148],[2163,2148]])  # new jayst whole
	# pts2 = np.float32([[0,0],[850,0],[0,2300],[850,2300]]) #new Jayst whole

	pts1 = np.float32([[2181, 1091], [2484,1161],[1610,2041],[2089,2157]])  # new jayst lower
	pts2 = np.float32([[0,0],[400,0],[0,1100],[400,1100]]) #new Jayst lower


	M = cv2.getPerspectiveTransform(pts1,pts2)
	return M

def perspectiveWarp(img, M,frame_idx,isSave):
	dst = cv2.warpPerspective(img,M,(400,1100))
	if isSave:
		# name = '../tempFigs/roi2/imgs/'+str(frame_idx).zfill(6)+'.jpg'
		name = '/media/My Book/CUSP/AIG/Jay&Johnson/roi2/imgs/'+str(frame_idx).zfill(6)+'.jpg'
		cv2.imwrite(name, dst)
	return dst


if __name__ == '__main__':
	linux_video_src = '/media/TOSHIBA/DoTdata/VideoFromCUSP/C0007.MP4' #complete
	# mac_video_src = '/Volumes/TOSHIBA/DoTdata/VideoFromCUSP/C0007.avi' #partial
	# test_video_src = '/Users/Chenge/Desktop/C0007.avi'

	cap       = cv2.VideoCapture(linux_video_src)
	nrows     = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
	ncols     = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
	nframe    = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
	framerate = cap.get(cv2.cv.CV_CAP_PROP_FPS)

	start_position = 114770  
	print 'reading buffer...'
	# for ii in range(start_position):
	# 	print(ii)
	# 	rval, img = cap.read()
	cap.set ( cv2.cv.CV_CAP_PROP_POS_FRAMES , start_position)
	print 'warp image...read frame',str(start_position)
	st, img = cap.read()
	warpMtx = getPerspectiveMtx(img)
	pdb.set_trace()
	for frame_idx in range(int(nframe/2)):
		print "frame_idx ",str(start_position+1+frame_idx)
		isSave        = 1
		status, frame = cap.read()
		dst           = perspectiveWarp(frame, warpMtx, start_position+1+frame_idx,isSave)
