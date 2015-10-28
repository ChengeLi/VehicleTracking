import cv2
import numpy as np
from matplotlib import pyplot as plt
import pdb


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

def perspectiveWarp(img, M,frame_idx):
	dst = cv2.warpPerspective(img,M,(400,1100))
	# plt.subplot(121),plt.imshow(img[:,:,::-1]),plt.title('Input')
	# plt.subplot(122),plt.imshow(dst[:,:,::-1]),plt.title('ROI Output')
	# plt.show()


	# plt.imshow(dst[:,:,::-1])
	name = './tempFigs/roi2/'+str(frame_idx).zfill(6)+'.jpg'
	# plt.savefig(name) ##save figure'
	cv2.imwrite(name, dst)

	'''
	# 2 step way, (not so good)
	pts1_step1 = np.float32([[161, 147],[224,147], [132,229],[271,220]])  # canal st
	pts2_step1 = np.float32([[0,0],[350,0],[0,200],[350,200]]) 

	pts1_step2 = np.float32([[132,229],[271,220],[76,513],[408,410]]) 
	pts2_step2 = np.float32([[0,200],[350,200],[0,1000],[350,1000]]) 

	M1 = cv2.getPerspectiveTransform(pts1_step1,pts2_step1)
	M2 = cv2.getPerspectiveTransform(pts1_step2,pts2_step2)

	# dst = cv2.warpPerspective(img_small,M,(300,300))
	dst1 = cv2.warpPerspective(img,M1,(350,200))
	dst2 = cv2.warpPerspective(img,M2,(350,1000))

	plt.subplot(121),plt.imshow(dst1),plt.title('step1')
	plt.subplot(122),plt.imshow(dst2),plt.title('step2')
	plt.show()


	'''





if __name__ == '__main__':
	linux_video_src = '/media/TOSHIBA/DoTdata/VideoFromCUSP/C0007.MP4'
	# mac_video_src = '/Volumes/TOSHIBA/DoTdata/VideoFromCUSP/C0007.avi'
	# test_video_src = '/Users/Chenge/Desktop/C0007.avi'

	cap       = cv2.VideoCapture(linux_video_src)
	nrows     = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
	ncols     = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
	nframe    = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
	    

	start_position = 4801
	print 'reading buffer...'
	for ii in range(start_position):
	    print(ii)
	    rval, img = cap.read()


	# status, frame = cap.read()
	# plt.imshow(frame[:,:,::-1])
	warpMtx = getPerspectiveMtx(img)
	
	for frame_idx in range(1801):
		status, frame = cap.read()
		perspectiveWarp(frame, warpMtx, start_position+frame_idx)
		# pdb.set_trace()
