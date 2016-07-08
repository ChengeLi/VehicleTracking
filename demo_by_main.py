
# from shell
# stdbuf -oL python demo_by_main.py 2 >../CanalVideos/log2 &
## don't use avi files (bc from matlab avi files' fps==1000???)
# from matlab asf files' fps==30, correct


import time
import sys
import numpy as np


def wholeProcess(dataSource,VideoIndex):
	from DataPathclass import *
	DataPathobj = DataPath(dataSource,VideoIndex)

	assert len(sorted(glob.glob(DataPathobj.blobPath+'/*.mat')))>0
	execfile('foreground_blob.py')

	print("running KLT...")
	t0 = time.time()
	execfile('klt_func.py')
	t1 = time.time() - t0
	print t1

	print("filter the trjs...")
	t0 = time.time()
	execfile('fit_extrapolate.py')
	t2 = time.time() - t0  # t2 = 2460.27871799469
	print t2  

	print("running trjcluster...")
	t0 = time.time()
	execfile('trjcluster_func_SBS.py')
	t3 = time.time() - t0
	print 't3 = ',t3
	print t3

	print("running subspace_cluster...")
	t0 = time.time()
	execfile('subspace_cluster.py')
	t4 = time.time() - t0  # t4 = 2334.37383294 
	print t4

	print("running unify_label...")
	t0 = time.time()
	execfile('unify_label_func.py')
	t5 = time.time() - t0
	print t5

	print "save the final result to dic format" 
	t0 = time.time()
	execfile('trj2dic.py')
	t6 = time.time() - t0
	print t6  # 172.96072793    

	print "get trj pairs" 
	t0 = time.time()
	execfile('getVehiclesPairs.py') # writing to csv takes more than 3 hours
	t7 = time.time() - t0
	print t7


	print str(t1), str(t2), str(t3), str(t4), str(t5), str(t6),str(t7)



if __name__ == '__main__':
	dataSource = 'DoT'
	# VideoIndex = np.int(sys.argv[1])
	# VideoIndex = 48
	for VideoIndex in range(51,124,1):
		print 'VideoIndex ', VideoIndex
		wholeProcess(dataSource,VideoIndex)

