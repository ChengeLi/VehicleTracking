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

def prepare_data(isAfterWarpping,isLeft=True):
    if isAfterWarpping:
        if isLeft:
			test_vctime = pickle.load( open( "../DoT/CanalSt@BaxterSt-96.106/leftlane/result/final_vctime.p", "rb" ) )
			test_vcxtrj = pickle.load( open( "../DoT/CanalSt@BaxterSt-96.106/leftlane/result/final_vcxtrj.p", "rb" ) )
			test_vcytrj = pickle.load( open( "../DoT/CanalSt@BaxterSt-96.106/leftlane/result/final_vcytrj.p", "rb" ) )

			left_image_list = sorted(glob.glob('../DoT/CanalSt@BaxterSt-96.106/leftlane/img/*.jpg'))
			image_list      = left_image_list
			savePath        = "../DoT/CanalSt@BaxterSt-96.106/leftlane/pair/"

        else:
			test_vctime      = pickle.load( open( "../DoT/CanalSt@BaxterSt-96.106/rightlane/result/final_vctime.p", "rb" ) )
			test_vcxtrj      = pickle.load( open( "../DoT/CanalSt@BaxterSt-96.106/rightlane/result/final_vcxtrj.p", "rb" ) )
			test_vcytrj      = pickle.load( open( "../DoT/CanalSt@BaxterSt-96.106/rightlane/result/final_vcytrj.p", "rb" ) )
			
			right_image_list = sorted(glob.glob('../DoT/CanalSt@BaxterSt-96.106/rightlane/img/*.jpg'))
			image_list       = right_image_list
			savePath         = "../DoT/CanalSt@BaxterSt-96.106/rightlane/pair/"   

    else:
		test_vctime  = pickle.load( open( "../tempFigs/roi2/dic/final_vctime.p", "rb" ) )
		test_vcxtrj  = pickle.load( open( "../tempFigs/roi2/dic/final_vcxtrj.p", "rb" ) )
		test_vcytrj  = pickle.load( open( "../tempFigs/roi2/dic/final_vcytrj.p", "rb" ) )
		# image_list = sorted(glob.glob('../tempFigs/roi2/*.jpg'))
		image_list   = sorted(glob.glob('/media/TOSHIBA/DoTdata/VideoFromCUSP/roi2/imgs/*.jpg'))
		savePath     = "../tempFigs/roi2/"

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

        

def get_Co_occur(VehicleObj1, VehicleObj2):
	if len(VehicleObj1.frame)==0 or len(VehicleObj2.frame)==0:
		print "error! trj time is empty!"
		return
	else:
		appeartime1 = VehicleObj1.frame[0]
		gonetime1   = VehicleObj1.frame[1]
		appeartime2 = VehicleObj2.frame[0]
		gonetime2   = VehicleObj2.frame[1]

	if appeartime1 >= appeartime2: ## swap 1 and 2, s.t. 1 always <=2
		temp        = appeartime2
		appeartime2 = appeartime1
		appeartime1 = temp

	if gonetime1 >= appeartime2:
		if gonetime1 <=gonetime2:
			cooccur_ran = range(appeartime2, gonetime1+1)
		else:
			cooccur_ran = range(appeartime2, gonetime2+1)
		cooccur_IDs    = [VehicleObj1.VehicleID, VehicleObj2.VehicleID]
		coorccurStatus = 1	


	if gonetime1 < appeartime2:
		print "no co-occurance!"
		coorccurStatus = 0
		cooccur_ran    = []
		cooccur_IDs    = []

	return coorccurStatus, cooccur_ran, cooccur_IDs


def get_Co_location(cooccur_ran,cooccur_IDs,obj_pair2loop):
	ID111      = cooccur_IDs[0]
	ID222      = cooccur_IDs[1]
	
	xTrj       = obj_pair2loop.xTrj[ID111]  
	
	
	fullrange1 = range(obj_pair2loop.frame[ID111][0], obj_pair2loop.frame[ID111][1]+1)
	startind1  = fullrange1.index(cooccur_ran[0])
	endind1    = fullrange1.index(cooccur_ran[-1])
	
	fullrange2 = range(obj_pair2loop.frame[ID222][0], obj_pair2loop.frame[ID222][1]+1)
	startind2  = fullrange2.index(cooccur_ran[0])
	endind2    = fullrange2.index(cooccur_ran[-1])	


	co1X = obj_pair2loop.xTrj[ID111][startind1:endind1+1]
	co1Y = obj_pair2loop.yTrj[ID111][startind1:endind1+1]

	co2X = obj_pair2loop.xTrj[ID222][startind2:endind2+1]
	co2Y = obj_pair2loop.yTrj[ID222][startind2:endind2+1]
	gkk  = 0
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
	return co1X, co2X, co1Y, co2Y





def visual_pair(co1X, co2X, co1Y, co2Y,cooccur_ran, color, k1,k2, saveflag = 0):
	vcxtrj1 = co1X
	vcytrj1 = co1Y
	vcxtrj2 = co2X
	vcytrj2 = co2Y
	dots = []
	# color1 = colors()
	# color2 = colors()
	for k in range(np.size(cooccur_ran)):
		frame_idx = cooccur_ran[k]
		# print "frame_idx: " ,frame_idx
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
		dots.append(axL.scatter(vcxtrj1[k], vcytrj1[k], s=10, color=(color[k1].T)/255.,edgecolor='none')) 
		dots.append(axL.scatter(vcxtrj2[k], vcytrj2[k], s=10, color=(color[k2].T)/255.,edgecolor='none'))

		plt.draw()
		plt.show()
		# pdb.set_trace()

		# del dots[:]
		# plt.show()
		# pdb.set_trace()
		# for i in dots:
		#     i.remove()
		dots = []
		plt.show()
		plt.pause(0.00001)

		if saveflag == 1:
			name = './figures/'+str(frame_idx).zfill(6)+'.jpg'
			# pdb.set_trace()
			plt.savefig(name) ##save figure



def visual_givenID(loopVehicleID1, loopVehicleID2, obj_pair2loop,  color , saveflag = 0):

	VehicleObj1 = VehicleObj(obj_pair2loop,loopVehicleID1)
	VehicleObj2 = VehicleObj(obj_pair2loop,loopVehicleID2)

	if abs(VehicleObj1.frame[0] - VehicleObj2.frame[0]) >=600:
		return
	[coorccurStatus, cooccur_ran, cooccur_IDs ] = get_Co_occur(VehicleObj1, VehicleObj2)
	if coorccurStatus and np.size(cooccur_ran)>=3:
		[co1X, co2X, co1Y, co2Y] = get_Co_location(cooccur_ran,cooccur_IDs,obj_pair2loop) #get xy and write to file
		if np.size(cooccur_ran)>=15:
			saveflag = 0
			writer2.writerow([loopVehicleID1,loopVehicleID2])
			visual_pair(co1X, co2X, co1Y, co2Y,cooccur_ran, color,loopVehicleID1,loopVehicleID2, saveflag) #visualize


if __name__ == '__main__':
		
	"""

	badkey2 = []
	for key, val in test_vctime.iteritems():
	    if not val==[]:
	        if np.size(test_vcxtrj[key])!= val[1]-val[0]+1:
	            badkey2.append(key)    ## 50 out of the rest (4340-1480-50 = 2810)
	                                   
	for badk in badkey2:
	    del test_vctime[badk]
	    del test_vcxtrj[badk]
	    del test_vcytrj[badk]   


	badkey = []
	for key, val in test_vcxtrj.iteritems():
	    if val ==[] or np.size(val)<= 10: #5*3:
	        badkey.append(key)   ## 1480 out of 4340

	for badk in badkey:
	    del test_vctime[badk]
	    del test_vcxtrj[badk]
	    del test_vcytrj[badk]


	"""

	
	isAfterWarpping = False
	isLeft          = True
	test_vctime,test_vcxtrj,test_vcytrj,image_list,savePath = prepare_data(isAfterWarpping,isLeft)

	obj_pair = TrjObj(test_vcxtrj,test_vcytrj,test_vctime)

	clean_vctime = {key: value for key, value in test_vctime.items() 
	             if key not in obj_pair.bad_IDs1+obj_pair.bad_IDs2+obj_pair.bad_IDs3}
	clean_vcxtrj = {key: value for key, value in test_vcxtrj.items() 
	             if key not in obj_pair.bad_IDs1+obj_pair.bad_IDs2+obj_pair.bad_IDs3}
	clean_vcytrj = {key: value for key, value in test_vcytrj.items() 
	             if key not in obj_pair.bad_IDs1+obj_pair.bad_IDs2+obj_pair.bad_IDs3}

	print "trj remaining: ", str(len(clean_vctime))

	# rebuild this object using filtered data, should be no bad_IDs
	obj_pair2 = TrjObj(clean_vcxtrj,clean_vcytrj,clean_vctime)
	print obj_pair2.globalID
	# pickle.dump(obj_pair2,open("./mat/20150222_Mat/obj_pair2.p","wb"))


	obj2write = obj_pair2
	savename  = os.path.join(savePath,'Trj_with_ID_frm.csv')
	writer    = csv.writer(open(savename,'wb'))
	writer.writerow(['trj ID','frame','x','y','y direction','x direction'])
	temp      = []
	for kk in range(np.size(obj2write.Trj_with_ID_frm,0)):
		temp   =  obj2write.Trj_with_ID_frm[kk]
		curkey =  obj2write.Trj_with_ID_frm[kk][0]
		temp.append(obj2write.Ydir[curkey])
		temp.append(obj2write.Xdir[curkey])
		writer.writerow(temp)



	# pickle.dump( obj_pair.Trj_with_ID_frm, open( "./mat/20150222_Mat/singleListTrj.p", "wb" ) ) 
	# singleListTrj = pickle.load(open( "./mat/20150222_Mat/singleListTrj.p", "rb" ) )


	#=======visualize the pair relationship==============================================
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

	savenameCooccur = os.path.join(savePath,'pair_relationship.csv')
	writerCooccur   = csv.writer(open(savenameCooccur,'wb'))
	writerCooccur.writerow(['trj1 ID','frame','x','y','y direction','x direction','trj2 ID','frame','x','y','y direction','x direction'])
	obj_pair2loop   = obj_pair2



	savename2  = os.path.join(savePath,'pairs_ID.csv')
	writer2    = csv.writer(open(savename2,'wb'))
	writer2.writerow(['trj2 ID','trj2 ID'])


	plt.figure('testing')
	for ind1 in range(len(obj_pair2loop.globalID)-1):
		for ind2 in range(ind1+1, min(len(obj_pair2loop.globalID),ind1+500)):
			loopVehicleID1 = obj_pair2loop.globalID[ind1]
			loopVehicleID2 = obj_pair2loop.globalID[ind2]
			print "pairing: ",loopVehicleID1,' & ',loopVehicleID2
			plt.cla()
			axL   = plt.subplot(1,1,1)
			frame = np.zeros([nrows,ncols,3]).astype('uint8')
			im    = plt.imshow(np.zeros([nrows,ncols,3]))
			plt.axis('off')
			visual_givenID(loopVehicleID1, loopVehicleID2, obj_pair2loop, color = color_choice)


	plt.figure('testing')
	axL     = plt.subplot(1,1,1)
	frame   = np.zeros([nrows,ncols,3]).astype('uint8')
	im      = plt.imshow(np.zeros([nrows,ncols,3]))
	plt.axis('off')
	visual_givenID(273, 214, obj_pair2loop, color = color_choice)


	# pdb.set_trace()
	# fix me  want to see this
	# # testing: visualize given ID1 and ID2
	# visual_givenID(3255,3255, obj_pair2loop, color_choice)
	# visual_givenID(2534,2538, obj_pair2loop, color_choice)
	# visual_givenID(4510,4510, obj_pair2loop, color_choice)
	# visual_givenID(1787,1787, obj_pair2loop, color_choice)
	# visual_givenID(2322,2322, obj_pair2loop, color_choice)




	IDs_in_frame = {}
	for frame_idx in range(framenum):
		IDs_in_frame[frame_idx] = []
		for ind111 in range(len(obj_pair2loop.globalID)):
			loopVehicleID111 = obj_pair2loop.globalID[ind111]
			VehicleObj111 = VehicleObj(obj_pair2loop,loopVehicleID111)
			In_fram_Status22 = in_this_frame(VehicleObj111, frame_idx)
			
			if In_fram_Status22:
				IDs_in_frame[frame_idx].append(loopVehicleID111)
			else: continue


	# # saving as csv is really slow, and the file is very big (4.9 G)
	# # writer = csv.writer(open('./mat/20150222_Mat/IDs_in_frame.csv', 'wb'))
	# # temp = []
	# # for kk in range(len(IDs_in_frame)):
	# #     temp.append(kk)
	# #     temp.append(IDs_in_frame[kk])
	# #     writer.writerow(temp)

	# pickle.dump(IDs_in_frame, open("./mat/20150222_Mat/IDs_in_frame.p","wb"))











































