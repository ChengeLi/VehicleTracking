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





def prepare_data(isAfterWarpping,dataSource,isLeft=True):
	if dataSource == 'DoT':
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

	if dataSource == 'Johnson':
		# test_vctime  = pickle.load( open( "../tempFigs/roi2/dic/final_vctime.p", "rb" ) )
		# test_vcxtrj  = pickle.load( open( "../tempFigs/roi2/dic/final_vcxtrj.p", "rb" ) )
		# test_vcytrj  = pickle.load( open( "../tempFigs/roi2/dic/final_vcytrj.p", "rb" ) )
		# image_list   = sorted(glob.glob('/media/TOSHIBA/DoTdata/VideoFromCUSP/roi2/imgs/*.jpg'))
		# savePath     = "../tempFigs/roi2/"
		"""complete"""
		# test_vctime  = pickle.load( open( "/media/My Book/CUSP/AIG/Jay&Johnson/roi2/dic/final_vctime.p", "rb" ) )
		# test_vcxtrj  = pickle.load( open( "/media/My Book/CUSP/AIG/Jay&Johnson/roi2/dic/final_vcxtrj.p", "rb" ) )
		# test_vcytrj  = pickle.load( open( "/media/My Book/CUSP/AIG/Jay&Johnson/roi2/dic/final_vcytrj.p", "rb" ) )

		test_vctime  = pickle.load( open( "/media/My Book/CUSP/AIG/Jay&Johnson/roi2/dic/0-290367/final_vctime.p", "rb" ) )
		test_vcxtrj  = pickle.load( open( "/media/My Book/CUSP/AIG/Jay&Johnson/roi2/dic/0-290367/final_vcxtrj.p", "rb" ) )
		test_vcytrj  = pickle.load( open( "/media/My Book/CUSP/AIG/Jay&Johnson/roi2/dic/0-290367/final_vcytrj.p", "rb" ) )

		image_list   = sorted(glob.glob('/media/My Book/CUSP/AIG/Jay&Johnson/roi2/imgs/*.jpg'))
		savePath     = "/media/My Book/CUSP/AIG/Jay&Johnson/roi2/pair_relationship/"

	return test_vctime,test_vcxtrj,test_vcytrj,image_list,savePath



def visual_pair(co1X, co2X, co1Y, co2Y,cooccur_ran, color, k1,k2,saveflag = 0):
	vcxtrj1 = co1X
	vcytrj1 = co1Y
	vcxtrj2 = co2X
	vcytrj2 = co2Y
	dots = []

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
		plt.pause(0.00001)


		# del dots[:]
		# plt.show()
		# pdb.set_trace()
		# for i in dots:
		#     i.remove()
		dots = []

		if saveflag == 1:
			name = './figures/'+str(frame_idx).zfill(6)+'.jpg'
			# pdb.set_trace()
			plt.savefig(name) ##save figure



def visual_givenID(loopVehicleID1, loopVehicleID2, obj_pair2loop,  color, isWrite, isVisualize = False, visualize_threshold=15, saveflag = 0,overlap_pair_threshold = 15):

	VehicleObj1 = VehicleObj(obj_pair2loop,loopVehicleID1)
	VehicleObj2 = VehicleObj(obj_pair2loop,loopVehicleID2)

	if abs(VehicleObj1.frame[0] - VehicleObj2.frame[0]) >=600:
		return
	[coorccurStatus, cooccur_ran, cooccur_IDs ] = get_Co_occur(VehicleObj1, VehicleObj2)
	if coorccurStatus and np.size(cooccur_ran)>=overlap_pair_threshold:
		[co1X, co2X, co1Y, co2Y] = get_Co_location(cooccur_ran,cooccur_IDs,obj_pair2loop,isWrite) #get xy and write to file
		if np.size(cooccur_ran)>=visualize_threshold:
			print "cooccur length: ", str(cooccur_ran)
			saveflag = 0
			if isWrite:
				writer2.writerow([loopVehicleID1,loopVehicleID2])
			if isVisualize:
				visual_pair(co1X, co2X, co1Y, co2Y,cooccur_ran, color,loopVehicleID1,loopVehicleID2,saveflag) #visualize


if __name__ == '__main__':
	
	isAfterWarpping = False
	isLeft          = True
	dataSource      = 'Johnson'
	fps = 30

	test_vctime,test_vcxtrj,test_vcytrj,image_list,savePath = prepare_data(isAfterWarpping,dataSource,isLeft)


































