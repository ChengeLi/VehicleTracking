
import numpy as np
import cv2

def perspectiveTransform_chenge():  # implement perspectiveTransform by Chenge, same as cv func

	warpped_xyTupleMtx = cv2.perspectiveTransform(np.array([xyTupleMtx.reshape((-1,2))],dtype='float32'), np.array(warpingMtx,dtype='float32'))[0,:,:].reshape((-1,600,2))

	xy1TupleMtx = np.zeros((x_smooth_mtx[goodTrj,:].shape[0],x_smooth_mtx[goodTrj,:].shape[1],3))
	xy1TupleMtx[:,:,0] = np.array(x_smooth_mtx[goodTrj,:],dtype='float32')  #first dim is X!
	xy1TupleMtx[:,:,1] = np.array(y_smooth_mtx[goodTrj,:],dtype='float32')
	xy1TupleMtx[:,:,2] = np.ones(x_smooth_mtx[goodTrj,:].shape)

	xy1TupleList = np.array(xy1TupleMtx.reshape((-1,3)))
	xyw_prime = np.matrix(warpingMtx)*np.matrix(xy1TupleList).T
	x_prime = xyw_prime[0,:]
	y_prime = xyw_prime[1,:]
	w_prime = xyw_prime[2,:]

	warpped_x = x_prime/w_prime
	warpped_y = y_prime/w_prime


def loadWarpMtx():
	# homography_filename = '/home/chengeli/CUSP/AIG/Saunier/Canal_homographyNEW.txt'
	# warpingMtxFromFile = loadtext(homography_filename) # not accurate

	ptsLeft = np.float32([[161, 147],[224,147], [73,519],[399,397]])  # canal st, left lane
	ptsRight = np.float32([[229, 164],[340,151], [451,444],[703,331]])  # canal st, right lane
	ptsinWorldScale_half = np.float32([[0,0],[81,0],[0,736],[81,736]]) 

	Mleft = cv2.getPerspectiveTransform(ptsLeft,ptsinWorldScale_half)
	Mright = cv2.getPerspectiveTransform(ptsRight,ptsinWorldScale_half)

	'NO NEED TO SPLIT INTO LEFT OR RIGHT!'
	ptsinImage = np.float32([[154, 145],[340,151], [73,519],[703,316]])  # canal st, left lane
	ptsinWorldScale = np.float32([[0,0],[81*2,0],[0,736],[81*2,736]]) 
	M = cv2.getPerspectiveTransform(ptsinImage,ptsinWorldScale)
	# dstFrm = cv2.warpPerspective(frame,M,(81*2,736))


	# dstleft = cv2.warpPerspective(frame,Mleft,(81,736))  # for image
	# dstright = cv2.warpPerspective(frame,Mright,(81,736))  
	(limitX, limitY) = (81*2,736)
	return Mleft, Mright, M, limitX, limitY





# frame = cv2.imread('/home/chengeli/CUSP/AIG/Saunier/00000012.jpg')
# # frame = cv2.imread('../DoT/CanalSt@BaxterSt-96.106/imgs/00000011.jpg')
# world_frame = cv2.imread('/home/chengeli/CUSP/AIG/Saunier/canal_baxter_world.jpg')

	


















