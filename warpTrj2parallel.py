
import numpy as np
import cv2




def loadWarpMtx(homography_filename):
	# warpingMtx = np.loadtxt(homography_filename)


	pts1 = np.float32([[242, 192],[179,500],[289,190],[447,417]])  #([x,y]) canal st, right lane
	pts2 = np.float32([[125,97],[684,621],[179,48],[719,548]]) #canal world
	# pts2 = np.float32([[0,0],[0,81],[81,0],[81,736]]) #canal


	pts1 = np.float32([[161, 147],[224,147], [73,519],[399,397]])  # canal st, left lane
	pts1 = np.float32([[229, 164],[340,151], [451,444],[703,331]])  # canal st, right lane
	pts2 = np.float32([[0,0],[81,0],[0,736],[81,731]]) #canal

	M = cv2.getPerspectiveTransform(pts1,pts2)

	dst = cv2.warpPerspective(frame,M,(81,736))  # canal

	dst = np.zeros((81,736))
	for src_y in range(frame.shape[0]):
		for src_x in range(frame.shape[1]):
			# [src_x,src_y] = frame[ii,jj,0] 
			dst_x = np.dot([src_x,src_y,1],M[0,:])/np.dot([src_x,src_y,1],M[2,:])	
			dst_y = np.dot([src_x,src_y,1],M[1,:])/np.dot([src_x,src_y,1],M[2,:])
			dst[np.floor(dst_y),np.floor(dst_x)] = frame[src_y,src_x,0]


	return warpingMtx

def warpTrj2parallel(frame, warpingMtx):
	dst = cv2.warpPerspective(frame,warpingMtx*1000,(350,600))  
	# dst2 = cv2.warpPerspective(frame,M,(81,736))
	# dst2 = cv2.warpPerspective(frame,M,(719,548))
	dst2 = cv2.warpPerspective(frame,M,(719,548))






homography_filename = '/home/chengeli/CUSP/AIG/Saunier/Canal_homographyNEW.txt'
warpingMtx = loadWarpMtx(homography_filename)

frame = cv2.imread('/home/chengeli/CUSP/AIG/Saunier/00000012.jpg')
world_frame = cv2.imread('/home/chengeli/CUSP/AIG/Saunier/canal_baxter_world.jpg')

leftLane = frame[190:500,179:447,:]






















