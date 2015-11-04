# warp trajectories based on image warping matrix
import cv2
import pdb
import pickle
import glob as glob
import numpy as np
import os as os
from matplotlib import pyplot as plt
from scipy.io import loadmat,savemat
from scipy.sparse import csr_matrix
from Trj_class_and_func_definitions import *


def warpImage(warpMtxLeft,warpMtxRight):
	image_listing   = sorted(glob.glob('../DoT/CanalSt@BaxterSt-96.106/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/*.jpg'))
	# image_listing = sorted(glob('./tempFigs/roi2/*.jpg'))
	for kk in range(len(image_listing)):
		realimg  = cv2.imread(image_listing[kk])
		dstLeft  = cv2.warpPerspective(realimg,warpMtxLeft,(350,600))
		dstRight = cv2.warpPerspective(realimg,warpMtxRight,(350,600))

		# plt.imshow(realimg[:,:,::-1]), plt.title('ORIGINAL'),plt.axis('off')
		# plt.figure(), plt.axis('off')
		# plt.subplot(121),plt.imshow(dstLeft[:,:,::-1]), plt.title('left lane trajectories')
		# plt.subplot(122),plt.imshow(dstRight[:,:,::-1]), plt.title('right lane trajectories')

		leftname  = os.path.join('../DoT/CanalSt@BaxterSt-96.106/leftlane/',str(kk).zfill(8))+'.jpg'
		rightname = os.path.join('../DoT/CanalSt@BaxterSt-96.106/rightlane/',str(kk).zfill(8))+'.jpg'
		cv2.imwrite(leftname,dstLeft)
		cv2.imwrite(rightname,dstRight)

def saveWarpped(frame_idx,left_lane_ID,left_warpped,left_xtrj,left_ytrj,right_lane_ID,right_warpped,right_xtrj,right_ytrj):
	trunclen = 600
	if len(left_lane_ID)>0:
		warpped            = {}
		warpped['mask']    = left_lane_ID
		warpped['xtracks'] = left_warpped[:,:,0]# consider the left lane       
		warpped['ytracks'] = left_warpped[:,:,1]  
		warpped['Ttracks'] = []
		warpped['xspd']    = []
		warpped['yspd']    = []
		savePath = '../DoT/CanalSt@BaxterSt-96.106/leftlane/'
		savename = os.path.join(savePath,'warpped_'+str(frame_idx/trunclen).zfill(3))
		savemat(savename,warpped)
		

		original            = {}
		original['mask']    = left_lane_ID
		original['xtracks'] = left_xtrj # consider the left lane       
		original['ytracks'] = left_ytrj  
		original['Ttracks'] = []
		original['xspd']    = []
		original['yspd']    = []
		savePath = '../DoT/CanalSt@BaxterSt-96.106/leftlane/'
		savename = os.path.join(savePath,'original_'+str(frame_idx/trunclen).zfill(3))
		savemat(savename,original)
	else:
		print "No vehicles in left lane in truncation: ", str(frame_idx/trunclen)


	if len(right_lane_ID)>0:
		warpped            = {}
		warpped['mask']    = right_lane_ID
		warpped['xtracks'] = right_warpped[:,:,0]# consider the right lane       
		warpped['ytracks'] = right_warpped[:,:,1]  
		warpped['Ttracks'] = []
		warpped['xspd']    = []
		warpped['yspd']    = []
		savePath = '../DoT/CanalSt@BaxterSt-96.106/rightlane/'
		savename = os.path.join(savePath,'warpped_'+str(frame_idx/trunclen).zfill(3))
		savemat(savename,warpped)
		

		original            = {}
		original['mask']    = right_lane_ID
		original['xtracks'] = right_xtrj # consider the right lane       
		original['ytracks'] = right_ytrj  
		original['Ttracks'] = []
		original['xspd']    = []
		original['yspd']    = []
		savePath = '../DoT/CanalSt@BaxterSt-96.106/rightlane/'
		savename = os.path.join(savePath,'original_'+str(frame_idx/trunclen).zfill(3))
		savemat(savename,original)
	else:
		print "No vehicles in right lane in truncation: ", str(frame_idx/trunclen)




def warpTrjs(frame_idx, warpMtxLeft, warpMtxRight, isClustered = False,isSave = True):
	trunclen = 600
	while frame_idx < 1800:
		# print "frame = ", str(frame_idx)
		if (frame_idx % trunclen == 0):
			print "frame = ", str(frame_idx)
			if not isClustered:
				print "build trj obj using raw trjs (x_re and y_re), non-clustered-yet result."
				print "Load the dictionary into a pickle file, trunk:", str(frame_idx/trunclen)
				loadPath   = "../DoT/CanalSt@BaxterSt-96.106/mat/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/"
				loadnameT  = os.path.join(loadPath,'vctime_'+str(frame_idx/trunclen).zfill(3))+'.p'
				loadnameX  = os.path.join(loadPath,'vcxtrj_'+str(frame_idx/trunclen).zfill(3))+'.p'
				loadnameY  = os.path.join(loadPath,'vcytrj_'+str(frame_idx/trunclen).zfill(3))+'.p'
				vctime     = pickle.load(open( loadnameT, "rb" ) )
				vcxtrj     = pickle.load(open( loadnameX, "rb" ) )
				vcytrj     = pickle.load(open( loadnameY, "rb" ) )
				raw_trjObj = TrjObj(vcxtrj, vcytrj,vctime)
				trjObj     = raw_trjObj

			else:
				print "build trj obj using the clustered result."
				vctime           = pickle.load(open( "../DoT/CanalSt@BaxterSt-96.106/mat/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/vctime.p", "rb" ) )
				vcxtrj           = pickle.load(open( "../DoT/CanalSt@BaxterSt-96.106/mat/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/vcxtrj.p", "rb" ) )
				vcytrj           = pickle.load(open( "../DoT/CanalSt@BaxterSt-96.106/mat/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/vcytrj.p", "rb" ) )
				clustered_trjObj =TrjObj(vcxtrj, vcytrj, vctime)
				trjObj           = clustered_trjObj


			# cleaning
			print "Bad ids: ", str(trjObj.bad_IDs)
			trjObj.frame = {key: value for key, value in vctime.items() 
			             if key not in trjObj.bad_IDs}
			trjObj.xTrj = {key: value for key, value in vcxtrj.items() 
			             if key not in trjObj.bad_IDs}
			trjObj.yTrj = {key: value for key, value in vcytrj.items() 
			             if key not in trjObj.bad_IDs}
			trjObj.globalID = [ value for value in trjObj.globalID 
			             if value not in trjObj.bad_IDs]

			# pdb.set_trace()
			# core: warpping
			Nsample  = len(trjObj.globalID)
			left_lane_ID  = []
			right_lane_ID = []
			left_xtrj     = np.zeros((Nsample,trunclen))*np.NaN
			left_ytrj     = np.zeros((Nsample,trunclen))*np.NaN
			right_xtrj    = np.zeros((Nsample,trunclen))*np.NaN
			right_ytrj    = np.zeros((Nsample,trunclen))*np.NaN
			
			
			aliveRange_lower  = lambda key: max(trjObj.frame[key][0],frame_idx)%trunclen
			aliveRange_higher = lambda key: trjObj.frame[key][1]%trunclen+1
			fromLastTrunFlag  = lambda key: trjObj.frame[key][0]<frame_idx #from last truncation
			toNextTrunFlag    = lambda key: trjObj.frame[key][1]>frame_idx+trunclen # go across to next truncation

			for key in trjObj.globalID:
				if fromLastTrunFlag(key):
					print "trj ",str(key),"starts from last truncation!"
					lengthInLastTrun = frame_idx- trjObj.frame[key][0]
					thisX = trjObj.xTrj[key][lengthInLastTrun:,0] # only save the current truncation
					thisY = trjObj.yTrj[key][lengthInLastTrun:,0]
				elif toNextTrunFlag(key):
					print "trj ",str(key),"lasts to next truncation!"
					lengthInThisTrun = frame_idx+trunclen - trjObj.frame[key][0]
					thisX = trjObj.xTrj[key][0:lengthInThisTrun+1,0]
					thisY = trjObj.yTrj[key][0:lengthInThisTrun+1,0]
				else:
					thisX = trjObj.xTrj[key][:,0]
					thisY = trjObj.yTrj[key][:,0]
				
					
				# print "max y location: ",str(np.max(trjObj.yTrj[key]))	
				if trjObj.Ydir[key] == 1 and np.max(trjObj.xTrj[key])<= 400: #left lane criteria
					left_lane_ID.append(key)
					left_xtrj[len(left_lane_ID)-1,aliveRange_lower(key):aliveRange_higher(key)] = thisX
					left_ytrj[len(left_lane_ID)-1,aliveRange_lower(key):aliveRange_higher(key)] = thisY
				else:
					# print "max x location: ",str(np.max(trjObj.xTrj[key]))
					# print str(key)
					right_lane_ID.append(key) 
					right_xtrj[len(right_lane_ID)-1,aliveRange_lower(key):aliveRange_higher(key)] = thisX
					right_ytrj[len(right_lane_ID)-1,aliveRange_lower(key):aliveRange_higher(key)] = thisY
			
			# delete the empty rows
			left_xtrj  = left_xtrj[0:len(left_lane_ID),:]
			left_ytrj  = left_ytrj[0:len(left_lane_ID),:]
			right_xtrj = right_xtrj[0:len(right_lane_ID),:]
			right_ytrj = right_ytrj[0:len(right_lane_ID),:]

			# pdb.set_trace()
			left_locations  = np.vstack(([left_xtrj.T], [left_ytrj.T])).T
			right_locations = np.vstack(([right_xtrj.T], [right_ytrj.T])).T

			# ======== combine x and y matrix, into the location matrix (x,y)
			left_warpped  = cv2.perspectiveTransform(np.array(left_locations[:,:,:]),warpMtxLeft)
			right_warpped = cv2.perspectiveTransform(np.array(right_locations[:,:,:]),warpMtxRight)

			pdb.set_trace()
			if isSave:
				print "Save left and right warpped trjs."
				saveWarpped(frame_idx,left_lane_ID,left_warpped,left_xtrj,left_ytrj,right_lane_ID,right_warpped,right_xtrj,right_ytrj)
			else:
				pass


			"""put the warpped value back to create new trjs"""
			# warpped_xTrj = {}
			# warpped_yTrj = {}
			# for ll in range(len(left_lane_ID)):
			# 	leftID = left_lane_ID[ll]
			# 	warpped_xTrj[leftID] = left_warpped[ll,aliveRange_lower(key):aliveRange_higher(key)]
			# 	warpped_yTrj[leftID] = left_warpped[ll,aliveRange_lower(key):aliveRange_higher(key)]
				
			# for rr in range(len(right_lane_ID)):
			# 	righID = right_lane_ID[rr]
			# 	warpped_xTrj[righID] = right_warpped[rr,aliveRange_lower(key):aliveRange_higher(key)]
			# 	warpped_yTrj[righID] = right_warpped[rr,aliveRange_lower(key):aliveRange_higher(key)]

			# # add the warpped trjs to the objects' fields
			# trjObj.warpped_xTrj = warpped_xTrj
			# trjObj.warpped_yTrj = warpped_yTrj

		frame_idx = frame_idx+1 

	# return the Trj object, and the warpped results in matrix format, seperately
	return trjObj,left_warpped,right_warpped





# fix meeee.....
def unwarpTrjs(frame_idx, warpMtxLeftInv, warpMtxRightInv,isSave = False):
	pass 




if __name__ == '__main__':
    # matfiles = sorted(glob('./tempFigs/roi2/len4' +'*.mat'))
	# img = cv2.imread('/Users/Chenge/Desktop/DoT/trun1+2_50_30.png')
	# realimg = img[70:533,100:720,:]
	
	pts1_left    = np.float32([[161, 147],[224,147], [73,519],[399,397]])  # canal st, left lane
	pts1_right   = np.float32([[229, 164],[340,151], [451,444],[703,331]])  # canal st, right lane
	pts2_right   = np.float32([[0,0],[350,0],[0,600],[350,600]])
	pts2_left    = np.float32([[0,0],[350,0],[0,600],[350,600]])
	
	
	warpMtxLeft  = cv2.getPerspectiveTransform(pts1_left,pts2_left)
	warpMtxRight = cv2.getPerspectiveTransform(pts1_right,pts2_right)
	
	

	frame_idx = 0
	# warp the trjs based on left and right warpping matrixes
	raw_trjObj_ori,left_warpped,right_warpped  = warpTrjs(frame_idx, warpMtxLeft, warpMtxRight)



	# warp it back
	warpMtxLeftInv  = np.linalg.inv(warpMtxLeft)
	warpMtxRightInv = np.linalg.inv(warpMtxRight)
	# backLeft  = cv2.warpPerspective(dstLeft,warpMtxLeftInv,realimg.shape[:2][::-1])
	# backRight = cv2.warpPerspective(dstRight,warpMtxRightInv,realimg.shape[:2][::-1])
	# warpback  = backLeft + backRight
	# plt.figure()
	# plt.imshow(warpback[:,:,::-1])

	
	











