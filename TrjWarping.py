# warp trajectories based on image warping matrix



import cv2
import numpy as np
from matplotlib import pyplot as plt
import pdb

color = np.array([np.random.randint(0,255) for _ in range(3*int(100))]).reshape(100,3)

fakeTrj_y = [1,2,3,4,5,10,15,22,30,40,55]
fakeTrj_x = range(1,12)

x_re = fakeTrj_x
y_re = fakeTrj_y

fig888 = plt.figure()
ax = plt.subplot(111)
lines = ax.plot(x_re[:], y_re[:],color = (color[i-1].T)/255.,linewidth=2)
fig888.canvas.draw()
plt.pause(0.0001)








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
    matfiles = sorted(glob('./tempFigs/roi2/len4' +'*.mat'))

	trunkTrjFile = loadmat(matfiles[frame_idx//trunclen])
	xtrj = csr_matrix(trunkTrjFile['x_re'], shape=trunkTrjFile['x_re'].shape).toarray()
	ytrj = csr_matrix(trunkTrjFile['y_re'], shape=trunkTrjFile['y_re'].shape).toarray()
	IDintrunk = trunkTrjFile['mask'][0]
	sample = trunkTrjFile['x_re'].shape[0] # num of trjs in this trunk
	fnum   = trunkTrjFile['x_re'].shape[1] # 600


	pts1_left = np.float32([[161, 147],[224,147], [73,519],[399,397]])  # canal st, left lane
	pts1_right = np.float32([[229, 164],[340,151], [451,444],[703,331]])  # canal st, right lane
	# pts2 = np.float32([[0,0],[350,0],[0,305],[350,305]])
	pts2_right = np.float32([[0,0],[350,0],[0,600],[350,600]])
	pts2_left = np.float32([[0,0],[350,0],[0,600],[350,600]])

	# pts2 = np.float32([[0,0],[380,0],[0,380],[380,380]]) #canal





	img = cv2.imread('/Users/Chenge/Desktop/DoT/trun1+2_50_30.png')
	realimg = img[70:533,100:720,:]


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

	
















