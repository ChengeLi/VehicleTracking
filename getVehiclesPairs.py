# consider the pair-wise relationship between each two cars
import os
import csv
import pdb
import cv2
import random
import numpy as np
import glob as glob
import cPickle as pickle
import matplotlib.pyplot as plt
from Trj_class_and_func_definitions import *
import sys


def prepare_data(isAfterWarpping,dataSource,isLeft=True):
	test_vctime = pickle.load( open(os.path.join(DataPathobj.dicpath,'final_vctime.p'), "rb" ) )
	test_vcxtrj = pickle.load( open(os.path.join(DataPathobj.dicpath,'final_vcxtrj.p'), "rb" ) )
	test_vcytrj = pickle.load( open(os.path.join(DataPathobj.dicpath,'final_vcytrj.p'), "rb" ) )			
	image_list  = DataPathobj.imagePath
	savePath    = DataPathobj.pairpath

	return test_vctime,test_vcxtrj,test_vcytrj,image_list,savePath



def in_this_frame(VehicleObj,frame_idx):
	### for each frame, get the IDs those are in this frame 
	In_fram_Status = 0
	appeartime = VehicleObj.frame[0]
	gonetime = VehicleObj.frame[1]

	if appeartime<= frame_idx: 
		if gonetime >= frame_idx:
			In_fram_Status = 1	

	return In_fram_Status

        

def get_Co_occur(VehicleObj1, VehicleObj2, subSampRate):
	if len(VehicleObj1.frame)==0 or len(VehicleObj2.frame)==0:
		print "error! trj time is empty!"
		return
	else:
		appeartime1 = VehicleObj1.frame[0]
		gonetime1   = VehicleObj1.frame[-1]
		appeartime2 = VehicleObj2.frame[0]
		gonetime2   = VehicleObj2.frame[-1]

	if appeartime1 >= appeartime2: ## swap 1 and 2, s.t. 1 always <=2
		temp        = appeartime2
		appeartime2 = appeartime1
		appeartime1 = temp

	if gonetime1 >= appeartime2:
		if gonetime1 <=gonetime2:
			cooccur_ran = range(appeartime2, gonetime1+subSampRate,subSampRate)
		else:
			cooccur_ran = range(appeartime2, gonetime2+subSampRate,subSampRate)
		cooccur_IDs    = [VehicleObj1.VehicleID, VehicleObj2.VehicleID]
		coorccurStatus = 1	


	if gonetime1 < appeartime2:
		print "no co-occurance!"
		coorccurStatus = 0
		cooccur_ran    = []
		cooccur_IDs    = []

	return coorccurStatus, cooccur_ran, cooccur_IDs


# def get_Co_location(cooccur_ran,cooccur_IDs,obj_pair2loop,subSampRate,isWrite=False):
# 	ID111      = cooccur_IDs[0]
# 	ID222      = cooccur_IDs[1]
	
# 	xTrj       = obj_pair2loop.xTrj[ID111]  
	
	
# 	fullrange1 = range(obj_pair2loop.frame[ID111][0], obj_pair2loop.frame[ID111][1]+1*subSampRate,subSampRate)
# 	startind1  = fullrange1.index(cooccur_ran[0])
# 	endind1    = fullrange1.index(cooccur_ran[-1])
	
# 	fullrange2 = range(obj_pair2loop.frame[ID222][0], obj_pair2loop.frame[ID222][1]+1*subSampRate,subSampRate)
# 	startind2  = fullrange2.index(cooccur_ran[0])
# 	endind2    = fullrange2.index(cooccur_ran[-1])	


# 	co1X = obj_pair2loop.xTrj[ID111][startind1:endind1+1]
# 	co1Y = obj_pair2loop.yTrj[ID111][startind1:endind1+1]

# 	co2X = obj_pair2loop.xTrj[ID222][startind2:endind2+1]
# 	co2Y = obj_pair2loop.yTrj[ID222][startind2:endind2+1]


"""use vehicle obj directly, no need to use trj obj"""
def get_Co_location(cooccur_ran,VehicleObj1,VehicleObj2,subSampRate,isWrite=False):
	fullrange1 = range(VehicleObj1.frame[0], VehicleObj1.frame[1]+1*subSampRate,subSampRate)
	startind1  = fullrange1.index(cooccur_ran[0])
	endind1    = fullrange1.index(cooccur_ran[-1])
	
	fullrange2 = range(VehicleObj2.frame[0], VehicleObj2.frame[1]+1*subSampRate,subSampRate)
	startind2  = fullrange2.index(cooccur_ran[0])
	endind2    = fullrange2.index(cooccur_ran[-1])	

	co1X = VehicleObj1.xTrj[startind1:endind1+1]
	co1Y = VehicleObj1.yTrj[startind1:endind1+1]

	co2X = VehicleObj2.xTrj[startind2:endind2+1]
	co2Y = VehicleObj2.yTrj[startind2:endind2+1]

	if isWrite:
		for gkk in range(np.size(cooccur_ran)):
			temp = []
			temp.append(ID111)
			temp.append(cooccur_ran[gkk])
			# print gkk
			temp.append(co1X[gkk])
			temp.append(co1Y[gkk])
			temp.append(obj_pair2loop.Ydir[ID111])
			temp.append(obj_pair2loop.Xdir[ID111])

			temp.append(ID222)
			temp.append(cooccur_ran[gkk])
			temp.append(co2X[gkk])
			temp.append(co2Y[gkk])
			temp.append(obj_pair2loop.Ydir[ID222])
			temp.append(obj_pair2loop.Xdir[ID222])
			
			writerCooccur.writerow(temp)
			writer2.writerow([ID111,ID222]) # just the pair ids
				
	return co1X, co2X, co1Y, co2Y



def visual_pair(co1X, co2X, co1Y, co2Y,cooccur_ran, color, k1,k2,saveflag = 0):
	vcxtrj1 = co1X
	vcytrj1 = co1Y
	vcxtrj2 = co2X
	vcytrj2 = co2Y
	dots = []

	for k in range(np.size(cooccur_ran)):
		frame_idx = cooccur_ran[k]
		# print "frame_idx: " ,frame_idx
		plt.title('frame '+str(frame_idx))

		if isVideo:
			cap.set ( cv2.cv.CV_CAP_PROP_POS_FRAMES ,frame_idx)
			status, frame = cap.read()
		else:
			tmpName= image_list[frame_idx]
			frame=cv2.imread(tmpName)

		im.set_data(frame[:,:,::-1])
		plt.draw()
		#    lines = axL.plot(vcxtrj1[k],vcytrj1[k],color = (color[k-1].T)/255.,linewidth=2)
		#    lines = axL.plot(vcxtrj2[k],vcytrj2[k],color = (color[k-1].T)/255.,linewidth=2)
		#    line_exist = 1
		# dots.append(axL.scatter(vx, vy, s=50, color=(color[k-1].T)/255.,edgecolor='black')) 
		# dots.append(axL.scatter(vcxtrj1[k], vcytrj1[k], s=50, color=(color[k-1].T)/255.,edgecolor='none')) 
		# dots.append(axL.scatter(vcxtrj2[k], vcytrj2[k], s=50, color=(color[k-1].T)/255.,edgecolor='none')) 
		# dots.append(axL.scatter(vcxtrj1[k], vcytrj1[k], s=50, color=(1,0,0),edgecolor='none'))
		# dots.append(axL.scatter(vcxtrj2[k], vcytrj2[k], s=50, color=(0,1,0),edgecolor='none'))
		dots.append(axL.scatter(vcxtrj1[k], vcytrj1[k], s=8, color=(color[k1].T)/255.,edgecolor='none')) 
		dots.append(axL.scatter(vcxtrj2[k], vcytrj2[k], s=8, color=(color[k2].T)/255.,edgecolor='none'))
		plt.draw()
		# plt.show()
		plt.pause(0.00001)


		# del dots[:]
		# plt.show()
		# pdb.set_trace()
		# for i in dots:
		#     i.remove()
		dots = []

		if saveflag == 1:
			name = '/media/My Book/CUSP/AIG/Jay&Johnson/roi2/PairFigures/'+str(frame_idx).zfill(6)+'.jpg'
			plt.savefig(name) ##save figure



def visual_givenID(loopVehicleID1, loopVehicleID2, obj_pair2loop,  color, isWrite, isVisualize = False, visualize_threshold=15, saveflag = 0,overlap_pair_threshold = 15):

	VehicleObj1 = VehicleObj(obj_pair2loop,loopVehicleID1)
	VehicleObj2 = VehicleObj(obj_pair2loop,loopVehicleID2)

	if abs(VehicleObj1.frame[0] - VehicleObj2.frame[0]) >=600:
		return
	[coorccurStatus, cooccur_ran, cooccur_IDs ] = get_Co_occur(VehicleObj1, VehicleObj2,subSampRate)
	if coorccurStatus and np.size(cooccur_ran)>=overlap_pair_threshold:
		# [co1X, co2X, co1Y, co2Y] = get_Co_location(cooccur_ran,cooccur_IDs,obj_pair2loop,subSampRate, isWrite) #get xy and write to file
		[co1X, co2X, co1Y, co2Y] = get_Co_location(cooccur_ran,VehicleObj1, VehicleObj2,subSampRate, isWrite) #get xy and write to file
		if np.size(cooccur_ran)>=visualize_threshold:
			# print "cooccur length: ", str(cooccur_ran)
			print "len(cooccur_ran):", len(cooccur_ran), cooccur_ran[0],'-',cooccur_ran[-1]
			sys.stdout.flush()
			saveflag = 0
			if isVisualize:
				print "trj",loopVehicleID1,"  trj",loopVehicleID2
				visual_pair(co1X, co2X, co1Y, co2Y,cooccur_ran, color,loopVehicleID1,loopVehicleID2,saveflag) #visualize



def assign_to_gt_box(trj,GTupperL_list,GTLowerR_list,GTcenterXY_list):
	"""assgin trj to the nearest GroundTruth bounding box"""

	"""find the overlapping time"""


	pass




if __name__ == '__main__':
	from DataPathclass import *
	DataPathobj = DataPath(dataSource, VideoIndex)
	from parameterClass import *
	Parameterobj = parameter(dataSource,VideoIndex)

	# fps = 5
	# subSampRate = 6
	fps = Parameterobj.targetFPS
	cap = cv2.VideoCapture(DataPathobj.video)
	subSampRate = int(np.round(cap.get(cv2.cv.CV_CAP_PROP_FPS)))/Parameterobj.targetFPS
	overlap_pair_threshold = 3*fps


	test_vctime,test_vcxtrj,test_vcytrj,image_list,savePath = prepare_data(isAfterWarpping,dataSource,isLeft)
	obj_pair = TrjObj(test_vcxtrj,test_vcytrj,test_vctime,subSampRate = subSampRate)
	# badkeys  = obj_pair.bad_IDs1+obj_pair.bad_IDs2+obj_pair.bad_IDs4
	badkeys  = obj_pair.bad_IDs1+obj_pair.bad_IDs2

	clean_vctime = {}
	clean_vcxtrj = {}
	clean_vcytrj = {}
	for key in test_vctime.keys():
		if key in badkeys:
			continue
		else: 
			clean_vctime[key] = test_vctime[key]
			clean_vcxtrj[key] = test_vcxtrj[key]
			clean_vcytrj[key] = test_vcytrj[key]

	# pickle.dump(clean_vctime, open('/media/My Book/CUSP/AIG/Jay&Johnson/roi2/subSamp/Pair_clean_vctime.p','wb'))
	# pickle.dump(clean_vcxtrj, open('/media/My Book/CUSP/AIG/Jay&Johnson/roi2/subSamp/Pair_clean_vcxtrj.p','wb'))
	# pickle.dump(clean_vcytrj, open('/media/My Book/CUSP/AIG/Jay&Johnson/roi2/subSamp/Pair_clean_vcytrj.p','wb'))
	
	# # for mac
	# pickle.dump(clean_vctime, open('../Jay&Johnson/roi2/Pair_clean_vctime.p','wb'))
	# pickle.dump(clean_vcxtrj, open('../Jay&Johnson/roi2/Pair_clean_vcxtrj.p','wb'))
	# pickle.dump(clean_vcytrj, open('../Jay&Johnson/roi2/Pair_clean_vcytrj.p','wb'))

	# clean_vctime = pickle.load(open('/media/My Book/CUSP/AIG/Jay&Johnson/roi2/Pair_clean_vctime.p','rb'))
	# clean_vcxtrj = pickle.load(open('/media/My Book/CUSP/AIG/Jay&Johnson/roi2/Pair_clean_vcxtrj.p','rb'))
	# clean_vcytrj = pickle.load(open('/media/My Book/CUSP/AIG/Jay&Johnson/roi2/Pair_clean_vcytrj.p','rb'))

	print "trj remaining: ", str(len(clean_vctime))
	# rebuild this object using filtered data, should be no bad_IDs
	obj_pair2 = TrjObj(clean_vcxtrj,clean_vcytrj,clean_vctime,subSampRate = subSampRate)
	# pickle.dump(obj_pair2,open('/media/My Book/CUSP/AIG/Jay&Johnson/roi2/subSamp/dic/complete_500-5-1/500-5-1clean_obj_pair2.p','wb'))
	# pickle.dump(obj_pair2,open('../Jay&Johnson/roi2/clean_obj_pair2.p','wb'))
	pickle.dump(obj_pair2,open(os.path.join(savePath,'obj_pair2.p'),'wb'))
	

	# pdb.set_trace()




	""" write clustered Trj infor(not-pairing): clusterID, virtual center X, vc Y, Y direction, X direction """
	# obj2write = obj_pair2
	# savename  = os.path.join(savePath,'Trj_with_ID_frm.csv')
	# writer    = csv.writer(open(savename,'wb'))
	# writer.writerow(['trj ID','frame','x','y','y direction','x direction'])
	# temp      = []
	# for kk in range(np.size(obj2write.Trj_with_ID_frm,0)):
	# 	temp   =  obj2write.Trj_with_ID_frm[kk]
	# 	curkey =  obj2write.Trj_with_ID_frm[kk][0]
	# 	temp.append(obj2write.Ydir[curkey])
	# 	temp.append(obj2write.Xdir[curkey])
	# 	writer.writerow(temp)



	# pickle.dump( obj_pair.Trj_with_ID_frm, open( "./mat/20150222_Mat/singleListTrj.p", "wb" ) ) 
	# singleListTrj = pickle.load(open( "./mat/20150222_Mat/singleListTrj.p", "rb" ) )


	#=======visualize the pair relationship==============================================
	isVideo  = True
	if isVideo:
		dataPath = DataPathobj.video #'My Book/CUSP/AIG/DoT/Convert3/CanalSt@BaxterSt-96.106/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms.avi'
		cap       = cv2.VideoCapture(dataPath)
		cap.set ( cv2.cv.CV_CAP_PROP_POS_FRAMES ,0)
		status, firstfrm = cap.read()
		framenum    = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

	else:
		image_list = sorted(glob.glob('../DoT/CanalSt@BaxterSt-96.106/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/*.jpg')) # only 2000 pictures
		# for plottting
		firstfrm =cv2.imread(image_list[0])
		framenum = int(len(image_list))


	nrows    = int(np.size(firstfrm,0))
	ncols    = int(np.size(firstfrm,1))
	
	# plt.figure(1,figsize =[10,12])
	# plt.figure()
	# axL     = plt.subplot(1,1,1)
	# frame   = np.zeros([nrows,ncols,3]).astype('uint8')
	# im      = plt.imshow(np.zeros([nrows,ncols,3]))
	# plt.axis('off')
	color_choice = np.array([np.random.randint(0,255) for _ in range(3*int(max(obj_pair2.globalID)))]).reshape(int(max(obj_pair2.globalID)),3)
	# colors  = lambda: np.random.rand(50)

	savenameCooccur = os.path.join(savePath,'pair_relationship_overlap3s.csv')
	writerCooccur   = csv.writer(open(savenameCooccur,'wb'))
	writerCooccur.writerow(['trj1 ID','frame','x','y','y direction','x direction','trj2 ID','frame','x','y','y direction','x direction'])
	obj_pair2loop   = obj_pair2

	savename2  = os.path.join(savePath,'pairs_ID_overlap3s.csv')
	writer2    = csv.writer(open(savename2,'wb'))
	writer2.writerow(['trj2 ID','trj2 ID'])

	isWrite     = True
	isVisualize = False
	plt.figure('testing')
	test_clusterSize  = pickle.load( open( os.path.join(DataPathobj.dicpath,'final_clusterSize.p'), "rb" ) )

	for ind1 in range(len(obj_pair2loop.globalID)-1):
		for ind2 in range(ind1+1, min(len(obj_pair2loop.globalID),ind1+500)):
			loopVehicleID1 = obj_pair2loop.globalID[ind1]
			loopVehicleID2 = obj_pair2loop.globalID[ind2]

			if (sum(test_clusterSize[loopVehicleID1])== len(test_clusterSize[loopVehicleID1]))\
				or  (sum(test_clusterSize[loopVehicleID2])== len(test_clusterSize[loopVehicleID2])):
				# print "single trj as cluster!"
				continue

			print "pairing: ",loopVehicleID1,' & ',loopVehicleID2
			plt.cla()
			axL   = plt.subplot(1,1,1)
			im    = plt.imshow(np.zeros([nrows,ncols,3]))
			plt.axis('off')
			visualize_threshold  = fps*10 # only if a pair shared more than this many frames, show them
			visual_givenID(loopVehicleID1, loopVehicleID2, obj_pair2loop,color_choice,isWrite, isVisualize, visualize_threshold,overlap_pair_threshold)





	# pdb.set_trace()
	# """use for signle stesting in the end, show pairs given IDs"""
	# plt.figure('testing2')
	# axL         = plt.subplot(1,1,1)
	# im          = plt.imshow(np.zeros([nrows,ncols,3]))
	# plt.axis('off')
	# isWrite     = False
	# isVisualize = True
	# visual_givenID(649, 703, obj_pair2loop,color_choice,isWrite, isVisualize ,visualize_threshold = 40)




	# """what IDs each frame has"""
	# IDs_in_frame = {}
	# for frame_idx in range(framenum):
	# 	IDs_in_frame[frame_idx] = []
	# 	for ind111 in range(len(obj_pair2loop.globalID)):
	# 		loopVehicleID111 = obj_pair2loop.globalID[ind111]
	# 		VehicleObj111 = VehicleObj(obj_pair2loop,loopVehicleID111)
	# 		In_fram_Status22 = in_this_frame(VehicleObj111, frame_idx)
			
	# 		if In_fram_Status22:
	# 			IDs_in_frame[frame_idx].append(loopVehicleID111)
	# 		else: continue

	# IDs_in_frame_filename = os.path.join(savePath,'IDs_in_frame.p' )
	# pickle.dump(IDs_in_frame, open(IDs_in_frame_filename,"wb"))







































