# warp trajectories based on image warping matrix
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pdb
import glob as glob
from scipy.io import loadmat,savemat
from scipy.sparse import csr_matrix
from Trj_class_and_func_definitions import *

import pickle


# color = np.array([np.random.randint(0,255) for _ in range(3*int(100))]).reshape(100,3)

# fakeTrj_y = [1,2,3,4,5,10,15,22,30,40,55]
# fakeTrj_x = range(1,12)

# x_re = fakeTrj_x
# y_re = fakeTrj_y

# fig888 = plt.figure()
# ax = plt.subplot(111)
# lines = ax.plot(x_re[:], y_re[:],color = (color[i-1].T)/255.,linewidth=2)
# fig888.canvas.draw()
# plt.pause(0.0001)








def perspectiveWarp(img, M,frame_idx):
	dst = cv2.warpPerspective(img,M,(350,600))
	# plt.subplot(121),plt.imshow(img[:,:,::-1]),plt.title('Input')
	# plt.subplot(122),plt.imshow(dst[:,:,::-1]),plt.title('ROI Output')
	# plt.show()


	# plt.imshow(dst[:,:,::-1])
	name = './tempFigs/roi2/'+str(frame_idx).zfill(6)+'.jpg'
	# plt.savefig(name) ##save figure'
	cv2.imwrite(name, dst)






if __name__ == '__main__':
	trunclen = 600
    # matfiles = sorted(glob('./tempFigs/roi2/len4' +'*.mat'))
	# img = cv2.imread('/Users/Chenge/Desktop/DoT/trun1+2_50_30.png')
	# realimg = img[70:533,100:720,:]
	
	pts1_left = np.float32([[161, 147],[224,147], [73,519],[399,397]])  # canal st, left lane
	pts1_right = np.float32([[229, 164],[340,151], [451,444],[703,331]])  # canal st, right lane
	# pts2 = np.float32([[0,0],[350,0],[0,305],[350,305]])
	pts2_right = np.float32([[0,0],[350,0],[0,600],[350,600]])
	pts2_left = np.float32([[0,0],[350,0],[0,600],[350,600]])

	# pts2 = np.float32([[0,0],[380,0],[0,380],[380,380]]) #canal

	# warpMtx = getPerspectiveMtx(pts1, pts2)
	warpMtxLeft = cv2.getPerspectiveTransform(pts1_left,pts2_left)
	warpMtxRight = cv2.getPerspectiveTransform(pts1_right,pts2_right)

	# perspectiveWarp(img, warpMtx, start_position+frame_idx)


	dstLeft = cv2.warpPerspective(realimg,warpMtxLeft,(350,600))
	dstRight = cv2.warpPerspective(realimg,warpMtxRight,(350,600))

	plt.imshow(realimg[:,:,::-1]), plt.title('ORIGINAL'),plt.axis('off')

	plt.figure(), plt.axis('off')
	plt.subplot(121),plt.imshow(dstLeft[:,:,::-1]), plt.title('left lane trajectories')
	plt.subplot(122),plt.imshow(dstRight[:,:,::-1]), plt.title('right lane trajectories')

	cv2.imwrite('originalImg.jpg',realimg)
	# cv2.imwrite('dstLeft.jpg',dstLeft)
	# cv2.imwrite('dstRight.jpg',dstRight)

	




def warpTrjs():
	trunclen = 600
	matfiles = sorted(glob.glob('../DoT/CanalSt@BaxterSt-96.106/adj/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/len50' +'*.mat'))
	frame_idx = 0
	trunkTrjFile = loadmat(matfiles[frame_idx//trunclen])
	xtrj = csr_matrix(trunkTrjFile['x_re'], shape=trunkTrjFile['x_re'].shape).toarray()
	ytrj = csr_matrix(trunkTrjFile['y_re'], shape=trunkTrjFile['y_re'].shape).toarray()
	ttrj = csr_matrix(trunkTrjFile['Ttracks'], shape=trunkTrjFile['Ttracks'].shape).toarray()

	xspd = csr_matrix(trunkTrjFile['xspd'], shape=trunkTrjFile['xspd'].shape).toarray()
	yspd = csr_matrix(trunkTrjFile['yspd'], shape=trunkTrjFile['yspd'].shape).toarray()

	IDintrunk = trunkTrjFile['mask'][0]
	Nsample = trunkTrjFile['x_re'].shape[0] # num of trjs in this trunk
	fnum   = trunkTrjFile['x_re'].shape[1] # 600


	print "build trj obj using raw trjs (x_re and y_re), non-clustered-yet result."

	raw_trj_time = {}
	raw_xtrj     = {}
	raw_ytrj     = {}
    for i in np.unique(IDintrunk):  #initialize
        vcxtrj[i]=[] 
        vcytrj[i]=[]
        vctime[i]=[]



	raw_trjObj = TrjObj(raw_xtrj, raw_ytrj, raw_trj_time)









# ======== combine x and y matrix, into the location matrix (x,y)

	
	xtrjwarpped = np.zeros(xtrj.shape)
	ytrjwarpped = np.zeros(ytrj.shape)

	for i in range(xtrj.shape[0]):
		for j in range(xtrj.shape[1]):
			temp = cv2.perspectiveTransform(np.array((xtrj[i,j],ytrj[i,j]), warpMtxLeft )
			xtrjwarpped[i,j] = temp[0]
			ytrjwarpped[i,j] = temp[1]

	



	dists = np.vstack(([xtrj.T], [ytrj.T])).T
	cv2.perspectiveTransform(np.array(dists[:,:,:]),warpMtxLeft)


	print "build trj obj using the clustered result."
	# vctime = pickle.load(open( "../DoT/CanalSt@BaxterSt-96.106/mat/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/vctime.p", "rb" ) )
	# vcxtrj = pickle.load(open( "../DoT/CanalSt@BaxterSt-96.106/mat/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/vcxtrj.p", "rb" ) )
	# vcytrj = pickle.load(open( "../DoT/CanalSt@BaxterSt-96.106/mat/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/vcytrj.p", "rb" ) )
	# clustered_Trj =TrjObj(vcxtrj, vcytrj, vctime)

	# # all dictionaries with trj ID as keys
	# clean_vctime = {key: value for key, value in vctime.items() 
	#              if key not in clustered_Trj.bad_IDs}
	# clean_vcxtrj = {key: value for key, value in vcxtrj.items() 
	#              if key not in clustered_Trj.bad_IDs}
	# clean_vcytrj = {key: value for key, value in vcytrj.items() 
	#              if key not in clustered_Trj.bad_IDs}


























