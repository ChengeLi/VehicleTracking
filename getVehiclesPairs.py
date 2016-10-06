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
import Trj_class_and_func_definitions
import sys

isVisualize = False
isWrite = True
isAfterWarpping = True

def robust_update(olddic, newdic):
	"""update dictionary robustly, no key duplicate"""
	if len(olddic)>1:
		maxkey = np.max(olddic.keys())
		for ii,oldkey in enumerate(newdic.keys()):
			newdic[oldkey+maxkey+1] = newdic[oldkey]
			del newdic[oldkey]
	olddic.update(newdic)
	return olddic

def prepare_data(isAfterWarpping,dataSource,folderName):
	image_list  = DataPathobj.imagePath
	
	dicFiles = sorted(glob.glob(DataPathobj.dicpath+'*final_vcxtrj*.p'))
	truncationNum = len(dicFiles)
	test_vctime = {}
	test_vcxtrj = {}
	test_vcytrj = {}
	test_clusterSize = {}
	for matidx in range(truncationNum):
		temp_vctime = pickle.load( open(os.path.join(DataPathobj.dicpath,'final_vctime_consecutive_frame'+str(matidx).zfill(3)+'.p'), "rb" ) )
		temp_vcxtrj = pickle.load( open(os.path.join(DataPathobj.dicpath,'final_vcxtrj'+str(matidx).zfill(3)+'.p'), "rb" ) )
		temp_vcytr = pickle.load( open(os.path.join(DataPathobj.dicpath,'final_vcytrj'+str(matidx).zfill(3)+'.p'), "rb" ) )
		temp_clusterSize = pickle.load( open( os.path.join(DataPathobj.dicpath,'final_clusterSize'+str(matidx).zfill(3)+'.p'), "rb" ))

		test_vctime = robust_update(test_vctime,temp_vctime)
		test_vcxtrj = robust_update(test_vcxtrj,temp_vcxtrj)
		test_vcytrj = robust_update(test_vcytrj,temp_vcytr)
		test_clusterSize = robust_update(test_clusterSize,temp_clusterSize)

	return test_vctime,test_vcxtrj,test_vcytrj,test_clusterSize,image_list


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

	if appeartime1 > appeartime2: ## swap 1 and 2, s.t. appeartime1 always <=appeartime2
		temp        = appeartime2
		appeartime2 = appeartime1
		appeartime1 = temp
		temp2     = gonetime2
		gonetime2 = gonetime1
		gonetime1 = temp2


	if gonetime1 >= appeartime2:
		if gonetime1 <=gonetime2:
			cooccur_ran = range(appeartime2, gonetime1+subSampRate,subSampRate)
		else:
			cooccur_ran = range(appeartime2, gonetime2+subSampRate,subSampRate)
		cooccur_IDs    = [VehicleObj1.VehicleID, VehicleObj2.VehicleID]
		coorccurStatus = 1	

	else:
		# print "no co-occurance!"
		coorccurStatus = 0
		cooccur_ran    = []
		cooccur_IDs    = []

	return coorccurStatus, cooccur_ran, cooccur_IDs


"""use vehicle obj directly, no need to use trj obj"""
def get_Co_location(cooccur_ran,cooccur_IDs, VehicleObj1,VehicleObj2,obj_pair2loop,subSampRate,isWrite):
	fullrange1 = range(VehicleObj1.frame[0], VehicleObj1.frame[-1]+1*subSampRate,subSampRate)
	startind1  = fullrange1.index(cooccur_ran[0])
	endind1    = fullrange1.index(cooccur_ran[-1])
	
	fullrange2 = range(VehicleObj2.frame[0], VehicleObj2.frame[-1]+1*subSampRate,subSampRate)
	startind2  = fullrange2.index(cooccur_ran[0])
	endind2    = fullrange2.index(cooccur_ran[-1])	

	co1X = VehicleObj1.fullxTrj[startind1:endind1+1]
	co1Y = VehicleObj1.fullyTrj[startind1:endind1+1]

	co2X = VehicleObj2.fullxTrj[startind2:endind2+1]
	co2Y = VehicleObj2.fullyTrj[startind2:endind2+1]
	if isWrite:
		ID111 = cooccur_IDs[0]
		ID222 = cooccur_IDs[1]
		for gkk in range(np.size(cooccur_ran)):
			temp = [ID111, cooccur_ran[gkk], co1X[gkk], co1Y[gkk], obj_pair2loop.Ydir[ID111], obj_pair2loop.Xdir[ID111],
					ID222, cooccur_ran[gkk], co2X[gkk], co2Y[gkk], obj_pair2loop.Ydir[ID222], obj_pair2loop.Xdir[ID222]]
			# print gkk
			writerCooccur.writerow(temp)
			writer2.writerow([ID111,ID222]) # just the pair ids
			# print "write..."
			# outputFile.write(str(temp)+'\n')
	return co1X, co2X, co1Y, co2Y



def visual_pair(co1X, co2X, co1Y, co2Y,cooccur_ran, color, k1,k2, isVideo = True,saveflag = 0):
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
		# else:
		# 	tmpName= image_list[frame_idx]
		# 	frame=cv2.imread(tmpName)

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



def pair_givenID_vis(loopVehicleID1, loopVehicleID2, obj_pair2loop,  color, isWrite =True, isVisualize = False, visualize_threshold=15, saveflag = 0, overlap_pair_threshold = 15, subSampRate = 6):
	"""generate pairwise relationship, write into csv file and visualize"""
	VehicleObj1 = Trj_class_and_func_definitions.VehicleObj(obj_pair2loop,loopVehicleID1)
	VehicleObj2 = Trj_class_and_func_definitions.VehicleObj(obj_pair2loop,loopVehicleID2)

	if abs(VehicleObj1.frame[0] - VehicleObj2.frame[0]) >=600:
		return
	[coorccurStatus, cooccur_ran, cooccur_IDs ] = get_Co_occur(VehicleObj1, VehicleObj2,subSampRate)
	if coorccurStatus and (np.size(cooccur_ran)>=overlap_pair_threshold):
		[co1X, co2X, co1Y, co2Y] = get_Co_location(cooccur_ran,cooccur_IDs,VehicleObj1, VehicleObj2,obj_pair2loop, subSampRate, isWrite) #get xy and write to file

		if np.size(cooccur_ran)>=visualize_threshold:
			# print "cooccur length: ", str(cooccur_ran)
			print "len(cooccur_ran):", len(cooccur_ran), cooccur_ran[0],'-',cooccur_ran[-1]
			sys.stdout.flush()
			if isVisualize:
				saveflag = 0
				print "trj",loopVehicleID1,"  trj",loopVehicleID2
				visual_pair(co1X, co2X, co1Y, co2Y,cooccur_ran, color, loopVehicleID1,loopVehicleID2, isVideo = True,saveflag = 0)


def assign_to_gt_box(trj,GTupperL_list,GTLowerR_list,GTcenterXY_list):
	"""assgin trj to the nearest GroundTruth bounding box"""

	"""find the overlapping time"""

	pass




# if __name__ == '__main__':
def pair_main(dataSource, VideoIndex,folderName):
	import DataPathclass 
	global DataPathobj
	DataPathobj = DataPathclass.DataPath(dataSource,VideoIndex)
	import parameterClass 
	global Parameterobj
	Parameterobj = parameterClass.parameter(dataSource,VideoIndex)

	# fps = 5
	# subSampRate = 6
	fps = Parameterobj.targetFPS
	# cap = cv2.VideoCapture(DataPathobj.video)
	# subSampRate = int(np.round(cap.get(cv2.cv.CV_CAP_PROP_FPS)))/Parameterobj.targetFPS
	subSampRate = int(30)/Parameterobj.targetFPS
	overlap_pair_threshold = 1*fps
	# if len(glob.glob(os.path.join(DataPathobj.pairpath,'obj_pair2.p')))!=0 and len(glob.glob(os.path.join(DataPathobj.pairpath,'all_final_clusterSize.p')))!=0:
	# 	print "already have obj_pair, loading...!"
	# 	obj_pair2 = pickle.load(open(os.path.join(DataPathobj.pairpath,'obj_pair2.p'),'rb'))
	# 	test_clusterSize = pickle.load( open( os.path.join(DataPathobj.pairpath,'all_final_clusterSize.p'), "rb" ) )
	# else:
	test_vctime,test_vcxtrj,test_vcytrj,test_clusterSize,image_list = prepare_data(isAfterWarpping,dataSource,folderName)
	obj_pair = Trj_class_and_func_definitions.TrjObj(test_vcxtrj,test_vcytrj,test_vctime,subSampRate = subSampRate)
	# badkeys  = obj_pair.bad_IDs1+obj_pair.bad_IDs2+obj_pair.bad_IDs4
	badkeys  = obj_pair.bad_IDs1+obj_pair.bad_IDs2

	for key in badkeys:
		del test_vctime[key]
		del test_vcxtrj[key]
		del test_vcytrj[key]
	clean_vctime = test_vctime
	clean_vcxtrj = test_vcxtrj
	clean_vcytrj = test_vcytrj
	print "trj remaining: ", str(len(clean_vctime))
	# rebuild this object using filtered data, should be no bad_IDs
	obj_pair2 = Trj_class_and_func_definitions.TrjObj(clean_vcxtrj,clean_vcytrj,clean_vctime,subSampRate = subSampRate)
	pickle.dump(obj_pair2,open(os.path.join(DataPathobj.pairpath,'obj_pair2.p'),'wb'))
	pickle.dump(test_clusterSize, open( os.path.join(DataPathobj.pairpath,'all_final_clusterSize.p'), "wb" ) )

	""" write clustered Trj infor(not-pairing): clusterID, virtual center X, vc Y, Y direction, X direction """
	# obj2write = obj_pair2
	# savename  = os.path.join(DataPathobj.pairpath,'Trj_with_ID_frm.csv')
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
	if isVisualize:
		isVideo  = True
		if isVideo:
			dataPath = DataPathobj.video 
			global cap
			cap       = cv2.VideoCapture(dataPath)
			cap.set ( cv2.cv.CV_CAP_PROP_POS_FRAMES ,0)
			status, firstfrm = cap.read()
			framenum = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
		else:
			image_list = sorted(glob.glob('../DoT/CanalSt@BaxterSt-96.106/CanalSt@BaxterSt-96.106_2015-06-16_16h03min52s762ms/*.jpg')) # only 2000 pictures
			firstfrm =cv2.imread(image_list[0])
			framenum = int(len(image_list))
		nrows = int(np.size(firstfrm,0))
		ncols = int(np.size(firstfrm,1))
		
		# plt.figure(1,figsize =[10,12])
		# plt.figure()
		# axL     = plt.subplot(1,1,1)
		# frame   = np.zeros([nrows,ncols,3]).astype('uint8')
		# im      = plt.imshow(np.zeros([nrows,ncols,3]))
		# plt.axis('off')
		# color_choice = np.array([np.random.randint(0,255) for _ in range(3*int(max(obj_pair2.globalID)))]).reshape(int(max(obj_pair2.globalID)),3)
		# colors  = lambda: np.random.rand(50)
		plt.figure('testing')


	color_choice = np.array([np.random.randint(0,255) for _ in range(3*int(max(obj_pair2.globalID)))]).reshape(int(max(obj_pair2.globalID)),3)
	savenameCooccur = os.path.join(DataPathobj.pairpath,'pair_relationship_overlap3s.csv')
	global writerCooccur
	writerCooccur   = csv.writer(open(savenameCooccur,'wb'))
	writerCooccur.writerow(['trj1 ID','frame','x','y','y direction','x direction','trj2 ID','frame','x','y','y direction','x direction'])

	savename2  = os.path.join(DataPathobj.pairpath,'pairs_ID_overlap3s.csv')
	global writer2
	writer2    = csv.writer(open(savename2,'wb'))
	writer2.writerow(['trj2 ID','trj2 ID'])

	# global outputFile
	# outputFile = open(savenameCooccur,'a')


	obj_pair2loop   = obj_pair2
	for ind1 in range(len(obj_pair2loop.globalID)-1):
		for ind2 in range(ind1+1, min(len(obj_pair2loop.globalID),ind1+500)):
			loopVehicleID1 = obj_pair2loop.globalID[ind1]
			loopVehicleID2 = obj_pair2loop.globalID[ind2]

			if (sum(test_clusterSize[loopVehicleID1])== len(test_clusterSize[loopVehicleID1]))\
				or  (sum(test_clusterSize[loopVehicleID2])== len(test_clusterSize[loopVehicleID2])):
				# print "single trj as cluster!"
				continue

			# print "pairing: ",loopVehicleID1,' & ',loopVehicleID2
			if isVisualize:
				plt.cla()
				axL   = plt.subplot(1,1,1)
				global im
				im    = plt.imshow(np.zeros([nrows,ncols,3]))
				plt.axis('off')
			visualize_threshold  = fps*10 # only if a pair shared more than this many frames, show them
			pair_givenID_vis(loopVehicleID1, loopVehicleID2, obj_pair2loop,color_choice, isWrite = isWrite, \
							 isVisualize = isVisualize, visualize_threshold = visualize_threshold,overlap_pair_threshold = overlap_pair_threshold)
	"""use for signle testing in the end, show pairs given IDs"""
	# plt.figure('testing2')
	# axL         = plt.subplot(1,1,1)
	# im          = plt.imshow(np.zeros([nrows,ncols,3]))
	# plt.axis('off')
	# isWrite     = False
	# isVisualize = True
	# pair_givenID_vis(649, 703, obj_pair2loop,color_choice,isWrite, isVisualize ,visualize_threshold = 40)


	"""what IDs each frame has"""
	# IDs_in_frame = {}
	# for frame_idx in range(framenum):
	# 	IDs_in_frame[frame_idx] = []
	# 	for ind111 in range(len(obj_pair2loop.globalID)):
	# 		loopVehicleID111 = obj_pair2loop.globalID[ind111]
	# 		VehicleObj111 = Trj_class_and_func_definitions.VehicleObj(obj_pair2loop,loopVehicleID111)
	# 		In_fram_Status22 = in_this_frame(VehicleObj111, frame_idx)
			
	# 		if In_fram_Status22:
	# 			IDs_in_frame[frame_idx].append(loopVehicleID111)
	# 		else: continue

	# IDs_in_frame_filename = os.path.join(DataPathobj.pairpath,'IDs_in_frame.p' )
	# pickle.dump(IDs_in_frame, open(IDs_in_frame_filename,"wb"))







































